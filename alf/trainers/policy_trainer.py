# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trainer for training an Algorithm on given environments."""

import abc
from absl import logging
import math
import os
import pprint
import signal
import sys
import time
import torch
import torch.nn as nn

import alf
from alf.algorithms.algorithm import Algorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.data_transformer import create_data_transformer
from alf.data_structures import StepType
from alf.environments.utils import create_environment
from alf.nest import map_structure
from alf.tensor_specs import TensorSpec
from alf.utils import common
from alf.utils import git_utils
from alf.utils import math_ops
from alf.utils.checkpoint_utils import Checkpointer
import alf.utils.datagen as datagen
from alf.utils.summary_utils import record_time
from alf.utils.video_recorder import VideoRecorder


class _TrainerProgress(nn.Module):
    def __init__(self):
        super(_TrainerProgress, self).__init__()
        self.register_buffer("_iter_num", torch.zeros((), dtype=torch.int64))
        self.register_buffer("_env_steps", torch.zeros((), dtype=torch.int64))
        self._num_iterations = None
        self._num_env_steps = None
        self._progress = None

    def set_termination_criterion(self, num_iterations, num_env_steps=0):
        self._num_iterations = float(num_iterations)
        self._num_env_steps = float(num_env_steps)
        # might be loaded from a checkpoint, so we update first
        self.update()

    def update(self, iter_num=None, env_steps=None):
        if iter_num is not None:
            self._iter_num.fill_(iter_num)
        if env_steps is not None:
            self._env_steps.fill_(env_steps)

        assert not (self._num_iterations is None
                    and self._num_env_steps is None), (
                        "You must first call set_terimination_criterion()!")
        iter_progress, env_steps_progress = 0, 0
        if self._num_iterations > 0:
            iter_progress = float(
                self._iter_num.to(torch.float64) / self._num_iterations)
        if self._num_env_steps > 0:
            env_steps_progress = float(
                self._env_steps.to(torch.float64) / self._num_env_steps)
        # If either criterion is met, the training ends
        self._progress = max(iter_progress, env_steps_progress)

    def set_progress(self, value: float):
        """Manually set the current progress.

        Args:
            value: a float number in [0, 1]
        """
        self._progress = value

    @property
    def progress(self):
        assert self._progress is not None, "Must call update() first!"
        return self._progress


class Trainer(object):
    """Base class for trainers.

    Trainer is responsible for creating algorithm and dataset/environment, setting up
    summary, checkpointing, running training iterations, and evaluating periodically.
    """

    _trainer_progress = _TrainerProgress()

    def __init__(self, config: TrainerConfig, ddp_rank: int = -1):
        """

        Args:
            config: configuration used to construct this trainer
            ddp_rank: process (and also device) ID of the process, if the
                process participates in a DDP process group to run distributed
                data parallel training. A value of -1 indicates regular single
                process training.
        """
        Trainer._trainer_progress = _TrainerProgress()
        root_dir = config.root_dir
        self._root_dir = root_dir
        self._train_dir = os.path.join(root_dir, 'train')
        self._eval_dir = os.path.join(root_dir, 'eval')

        self._algorithm_ctor = config.algorithm_ctor
        self._algorithm = None

        self._num_checkpoints = config.num_checkpoints
        self._checkpointer = None

        self._evaluate = config.evaluate
        self._eval_uncertainty = config.eval_uncertainty

        if config.num_evals is not None:
            self._eval_interval = common.compute_summary_or_eval_interval(
                config, config.num_evals)
        else:
            self._eval_interval = config.eval_interval

        if config.num_summaries is not None:
            self._summary_interval = common.compute_summary_or_eval_interval(
                config, config.num_summaries)
        else:
            self._summary_interval = config.summary_interval

        self._summaries_flush_secs = config.summaries_flush_secs
        self._summary_max_queue = config.summary_max_queue
        self._debug_summaries = config.debug_summaries
        self._summarize_grads_and_vars = config.summarize_grads_and_vars
        self._config = config
        self._random_seed = config.random_seed
        self._rank = ddp_rank

    def train(self):
        """Perform training."""
        self._restore_checkpoint()
        alf.summary.enable_summary()

        self._checkpoint_requested = False
        signal.signal(signal.SIGUSR2, self._request_checkpoint)
        # kill -12 PID
        logging.info("Use `kill -%s %s` to request checkpoint during training."
                     % (int(signal.SIGUSR2), os.getpid()))

        self._debug_requested = False
        # kill -10 PID
        signal.signal(signal.SIGUSR1, self._request_debug)
        logging.info("Use `kill -%s %s` to request debugging." % (int(
            signal.SIGUSR1), os.getpid()))

        checkpoint_saved = False
        try:
            if self._config.profiling:
                import cProfile, pstats, io
                pr = cProfile.Profile()
                pr.enable()

            common.run_under_record_context(
                self._train,
                summary_dir=self._train_dir,
                summary_interval=self._summary_interval,
                summarize_first_interval=self._config.summarize_first_interval,
                flush_secs=self._summaries_flush_secs,
                summary_max_queue=self._summary_max_queue)

            if self._config.profiling:
                pr.disable()
                s = io.StringIO()
                ps = pstats.Stats(pr, stream=s).sort_stats('time')
                ps.print_stats()
                ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
                ps.print_stats()
                ps.print_callees()

                logging.info(s.getvalue())
            self._save_checkpoint()
            checkpoint_saved = True
        finally:
            if (self._config.confirm_checkpoint_upon_crash
                    and not checkpoint_saved and self._rank <= 0):
                # Prompts for checkpoint only when running single process
                # training (rank is -1) or master process of DDP training (rank
                # is 0).
                ans = input("Do you want to save checkpoint? (y/n): ")
                if ans.lower().startswith('y'):
                    self._save_checkpoint()
            self._close()

    @staticmethod
    def progress():
        """A static method that returns the current training progress, provided
        that only one trainer will be used for training.

        Returns:
            float: a number in :math:`[0,1]` indicating the training progress.
        """
        return Trainer._trainer_progress.progress

    @staticmethod
    def current_iterations():
        return Trainer._trainer_progress._iter_num

    @staticmethod
    def current_env_steps():
        return Trainer._trainer_progress._env_step

    def _train(self):
        """Perform training according the the learning type. """
        pass

    def _close(self):
        """Closing operations after training. """
        pass

    def _summarize_training_setting(self):
        # We need to wait for one iteration to get the operative args
        # Right just give a fixed gin file name to store operative args
        common.write_config(self._root_dir)
        with alf.summary.record_if(lambda: True):

            def _markdownify(paragraph):
                return "    ".join(
                    (os.linesep + paragraph).splitlines(keepends=True))

            common.summarize_config()
            alf.summary.text('commandline', ' '.join(sys.argv))
            alf.summary.text(
                'optimizers',
                _markdownify(self._algorithm.get_optimizer_info()))
            alf.summary.text(
                'unoptimized_parameters',
                _markdownify(self._algorithm.get_unoptimized_parameter_info()))
            alf.summary.text('revision', git_utils.get_revision())
            alf.summary.text('diff', _markdownify(git_utils.get_diff()))
            alf.summary.text('seed', str(self._random_seed))

            if self._config.code_snapshots is not None:
                for f in self._config.code_snapshots:
                    path = os.path.join(
                        os.path.abspath(os.path.dirname(__file__)), "..", f)
                    if not os.path.isfile(path):
                        common.warning_once(
                            "The code file '%s' for summary is invalid" % path)
                        continue
                    with open(path, 'r') as fin:
                        code = fin.read()
                        # adding "<pre>" will make TB show raw text instead of MD
                        alf.summary.text('code/%s' % f,
                                         "<pre>" + code + "</pre>")

    def _request_checkpoint(self, signum, frame):
        self._checkpoint_requested = True

    def _request_debug(self, signum, frame):
        self._debug_requested = True

    def _save_checkpoint(self):
        # Saving checkpoint is only enabled when running single process training
        # (rank is -1) or master process of DDP training (rank is 0).
        if self._rank <= 0:
            global_step = alf.summary.get_global_counter()
            self._checkpointer.save(global_step=global_step)

    def _restore_checkpoint(self, checkpointer):
        """Retore from saved checkpoint.

            Args:
                checkpointer (Checkpointer):
        """
        if checkpointer.has_checkpoint():
            # Some objects (e.g. ReplayBuffer) are constructed lazily in algorithm.
            # They only appear after one training iteration. So we need to run
            # train_iter() once before loading the checkpoint
            self._algorithm.train_iter()

        try:
            recovered_global_step = checkpointer.load()
            self._trainer_progress.update()
        except RuntimeError as e:
            raise RuntimeError(
                ("Checkpoint loading failed from the provided root_dir={}. "
                 "Typically this is caused by using a wrong checkpoint. \n"
                 "Please make sure the root_dir is set correctly. "
                 "Use a new value for it if "
                 "planning to train from scratch. \n"
                 "Detailed error message: {}").format(self._root_dir, e))
        if recovered_global_step != -1:
            alf.summary.set_global_counter(recovered_global_step)

        self._checkpointer = checkpointer


class RLTrainer(Trainer):
    """Trainer for reinforcement learning. """

    def __init__(self, config: TrainerConfig, ddp_rank: int = -1):
        """

        Args:
            config (TrainerConfig): configuration used to construct this trainer
            ddp_rank (int): process (and also device) ID of the process, if the
                process participates in a DDP process group to run distributed
                data parallel training. A value of -1 indicates regular single
                process training.
        """
        super().__init__(config, ddp_rank)

        self._num_env_steps = config.num_env_steps
        self._num_iterations = config.num_iterations
        assert (self._num_iterations + self._num_env_steps > 0
                and self._num_iterations * self._num_env_steps == 0), \
            "Must provide #iterations or #env_steps exclusively for training!"
        self._trainer_progress.set_termination_criterion(
            self._num_iterations, self._num_env_steps)

        self._num_eval_episodes = config.num_eval_episodes
        alf.summary.should_summarize_output(config.summarize_output)

        env = alf.get_env()
        logging.info(
            "observation_spec=%s" % pprint.pformat(env.observation_spec()))
        logging.info("action_spec=%s" % pprint.pformat(env.action_spec()))
        data_transformer = create_data_transformer(
            config.data_transformer_ctor, env.observation_spec())
        self._config.data_transformer = data_transformer

        # keep compatibility with previous gin based config
        common.set_global_env(env)
        observation_spec = data_transformer.transformed_observation_spec
        common.set_transformed_observation_spec(observation_spec)
        logging.info("transformed_observation_spec=%s" %
                     pprint.pformat(observation_spec))

        self._algorithm = self._algorithm_ctor(
            observation_spec=observation_spec,
            action_spec=env.action_spec(),
            reward_spec=env.reward_spec(),
            env=env,
            config=self._config,
            debug_summaries=self._debug_summaries)
        self._algorithm.set_path('')
        if ddp_rank >= 0:
            if not self._algorithm.on_policy:
                raise RuntimeError(
                    'Mutli-GPU with DDP does not support off-policy training yet'
                )
            # Activate the DDP training
            self._algorithm.activate_ddp(ddp_rank)

        self._eval_env = None
        # Create a thread env to expose subprocess gin/alf configurations
        # which otherwise will be marked as "inoperative". Only created when
        # ``TrainerConfig.no_thread_env_for_conf=False``.
        self._thread_env = None

        # See ``alf/docs/notes/knowledge_base.rst```
        # (ParallelAlfEnvironment and ThreadEnvironment) for details.
        if config.no_thread_env_for_conf:
            if self._evaluate:
                self._eval_env = create_environment(
                    num_parallel_environments=1, seed=self._random_seed)
        else:
            if self._evaluate or isinstance(
                    env, alf.environments.parallel_environment.
                    ParallelAlfEnvironment):
                self._thread_env = create_environment(
                    nonparallel=True, seed=self._random_seed)
            if self._evaluate:
                self._eval_env = self._thread_env

        self._eval_metrics = None
        self._eval_summary_writer = None
        if self._evaluate:
            self._eval_metrics = [
                alf.metrics.AverageReturnMetric(
                    buffer_size=self._num_eval_episodes,
                    reward_shape=self._eval_env.reward_spec().shape),
                alf.metrics.AverageEpisodeLengthMetric(
                    buffer_size=self._num_eval_episodes),
                alf.metrics.AverageEnvInfoMetric(
                    example_env_info=self._eval_env.reset().env_info,
                    batch_size=self._eval_env.batch_size,
                    buffer_size=self._num_eval_episodes),
                alf.metrics.AverageDiscountedReturnMetric(
                    buffer_size=self._num_eval_episodes,
                    reward_shape=self._eval_env.reward_spec().shape),
            ]
            self._eval_summary_writer = alf.summary.create_summary_writer(
                self._eval_dir, flush_secs=config.summaries_flush_secs)

    def _close_envs(self):
        """Close all envs to release their resources."""
        alf.close_env()
        if self._eval_env is not None:
            self._eval_env.close()
        if (self._thread_env is not None
                and self._thread_env is not self._eval_env):
            self._thread_env.close()

    def _train(self):
        env = alf.get_env()
        env.reset()
        if self._eval_env:
            self._eval_env.reset()

        begin_iter_num = int(self._trainer_progress._iter_num)
        iter_num = begin_iter_num

        checkpoint_interval = math.ceil(
            (self._num_iterations
             or self._num_env_steps) / self._num_checkpoints)

        if self._num_iterations:
            time_to_checkpoint = self._trainer_progress._iter_num + checkpoint_interval
        else:
            time_to_checkpoint = self._trainer_progress._env_steps + checkpoint_interval

        while True:
            t0 = time.time()
            with record_time("time/train_iter"):
                train_steps = self._algorithm.train_iter()
            t = time.time() - t0
            logging.log_every_n_seconds(
                logging.INFO,
                '%s%s -> %s: %s time=%.3f throughput=%0.2f' %
                ('' if self._rank == -1 else f'[rank {self._rank:02d}] ',
                 common.get_conf_file(),
                 os.path.basename(self._root_dir.strip('/')), iter_num, t,
                 int(train_steps) / t),
                n_seconds=1)

            if self._evaluate and (iter_num + 1) % self._eval_interval == 0:
                self._eval()
            if iter_num == begin_iter_num:
                self._summarize_training_setting()

            # check termination
            env_steps_metric = self._algorithm.get_step_metrics()[1]
            total_time_steps = env_steps_metric.result()
            iter_num += 1

            self._trainer_progress.update(iter_num, total_time_steps)

            if ((self._num_iterations and iter_num >= self._num_iterations)
                    or (self._num_env_steps
                        and total_time_steps >= self._num_env_steps)):
                # Evaluate before exiting so that the eval curve shown in TB
                # will align with the final iter/env_step.
                if self._evaluate:
                    self._eval()
                break

            if ((self._num_iterations and iter_num >= time_to_checkpoint)
                    or (self._num_env_steps
                        and total_time_steps >= time_to_checkpoint)):
                self._save_checkpoint()
                time_to_checkpoint += checkpoint_interval
            elif self._checkpoint_requested:
                logging.info("Saving checkpoint upon request...")
                self._save_checkpoint()
                self._checkpoint_requested = False

            if self._debug_requested:
                self._debug_requested = False
                import pdb
                pdb.set_trace()

    def _close(self):
        """Closing operations after training. """
        self._close_envs()

    def _restore_checkpoint(self):
        checkpointer = Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, 'algorithm'),
            algorithm=self._algorithm,
            metrics=nn.ModuleList(self._algorithm.get_metrics()),
            trainer_progress=self._trainer_progress)

        super()._restore_checkpoint(checkpointer)

    @common.mark_eval
    def _eval(self):
        self._algorithm.eval()
        time_step = common.get_initial_time_step(self._eval_env)
        policy_state = self._algorithm.get_initial_predict_state(
            self._eval_env.batch_size)
        trans_state = self._algorithm.get_initial_transform_state(
            self._eval_env.batch_size)
        episodes = 0
        while episodes < self._num_eval_episodes:
            time_step, policy_step, trans_state = _step(
                algorithm=self._algorithm,
                env=self._eval_env,
                time_step=time_step,
                policy_state=policy_state,
                trans_state=trans_state,
                metrics=self._eval_metrics)
            policy_state = policy_step.state

            if time_step.is_last():
                episodes += 1

        step_metrics = self._algorithm.get_step_metrics()
        with alf.summary.push_summary_writer(self._eval_summary_writer):
            for metric in self._eval_metrics:
                metric.gen_summaries(
                    train_step=alf.summary.get_global_counter(),
                    step_metrics=step_metrics)

        common.log_metrics(self._eval_metrics)
        self._algorithm.train()


class SLTrainer(Trainer):
    """Trainer for supervised learning. """

    def __init__(self, config: TrainerConfig):
        """Create a SLTrainer

        Args:
            config (TrainerConfig): configuration used to construct this trainer
        """
        super().__init__(config)

        assert config.num_iterations > 0, \
            "Must provide num_iterations for training!"

        self._num_epochs = config.num_iterations
        self._trainer_progress.set_termination_criterion(self._num_epochs)
        self._algorithm = config.algorithm_ctor(config=config)
        self._algorithm.set_path('')

    def _train(self):
        begin_epoch_num = int(self._trainer_progress._iter_num)
        epoch_num = begin_epoch_num

        checkpoint_interval = math.ceil(
            self._num_epochs / self._num_checkpoints)
        time_to_checkpoint = begin_epoch_num + checkpoint_interval

        logging.info("==> Begin Training")
        while True:
            t0 = time.time()
            with record_time("time/train_iter"):
                train_steps = self._algorithm.train_iter()
                train_steps = train_steps or 1
            t = time.time() - t0
            logging.log_every_n_seconds(
                logging.INFO,
                '%s -> %s: %s time=%.3f throughput=%0.2f' %
                (common.get_conf_file(),
                 os.path.basename(self._root_dir.strip('/')), epoch_num, t,
                 int(train_steps) / t),
                n_seconds=1)

            if (epoch_num + 1) % self._eval_interval == 0:
                if self._evaluate:
                    self._algorithm.evaluate()
                if self._eval_uncertainty:
                    self._algorithm.eval_uncertainty()

            if epoch_num == begin_epoch_num:
                self._summarize_training_setting()

            # check termination
            epoch_num += 1
            self._trainer_progress.update(epoch_num)

            if (self._num_epochs and epoch_num >= self._num_epochs):
                if self._evaluate:
                    self._algorithm.evaluate()
                if self._eval_uncertainty:
                    self._algorithm.eval_uncertainty()
                break

            if self._num_epochs and epoch_num >= time_to_checkpoint:
                self._save_checkpoint()
                time_to_checkpoint += checkpoint_interval
            elif self._checkpoint_requested:
                logging.info("Saving checkpoint upon request...")
                self._save_checkpoint()
                self._checkpoint_requested = False

            if self._debug_requested:
                self._debug_requested = False
                import pdb
                pdb.set_trace()

    def _restore_checkpoint(self):
        checkpointer = Checkpointer(
            ckpt_dir=os.path.join(self._train_dir, 'algorithm'),
            algorithm=self._algorithm,
            trainer_progress=self._trainer_progress)

        super()._restore_checkpoint(checkpointer)


@torch.no_grad()
def _step(algorithm,
          env,
          time_step,
          policy_state,
          trans_state,
          metrics,
          render=False,
          recorder=None,
          sleep_time_per_step=0):
    policy_state = common.reset_state_if_necessary(
        policy_state, algorithm.get_initial_predict_state(env.batch_size),
        time_step.is_first())
    transformed_time_step, trans_state = algorithm.transform_timestep(
        time_step, trans_state)
    # save the untransformed time step in case that sub-algorithms need it
    transformed_time_step = transformed_time_step._replace(
        untransformed=time_step)
    policy_step = algorithm.predict_step(transformed_time_step, policy_state)

    if recorder:
        recorder.capture_frame(policy_step.info, time_step.is_last())
    elif render:
        env.render(mode='human')
        time.sleep(sleep_time_per_step)

    next_time_step = env.step(policy_step.output)
    for metric in metrics:
        metric(time_step.cpu())
    return next_time_step, policy_step, trans_state


@common.mark_eval
def play(root_dir,
         env,
         algorithm,
         checkpoint_step="latest",
         num_episodes=10,
         sleep_time_per_step=0.01,
         record_file=None,
         append_blank_frames=0,
         render=True,
         ignored_parameter_prefixes=[]):
    """Play using the latest checkpoint under `train_dir`.

    The following example record the play of a trained model to a mp4 video:
    .. code-block:: bash

        python -m alf.bin.play \
        --root_dir=~/tmp/bullet_humanoid/ppo2/ppo2-11 \
        --num_episodes=1 \
        --record_file=ppo_bullet_humanoid.mp4

    Args:
        root_dir (str): same as the root_dir used for `train()`
        env (AlfEnvironment): the environment
        algorithm (RLAlgorithm): the training algorithm
        checkpoint_step (int|str): the number of training steps which is used to
            specify the checkpoint to be loaded. If checkpoint_step is 'latest',
            the most recent checkpoint named 'latest' will be loaded.
        num_episodes (int): number of episodes to play
        sleep_time_per_step (float): sleep so many seconds for each step
        record_file (str): if provided, video will be recorded to a file
            instead of shown on the screen.
        append_blank_frames (int): If >0, wil append such number of blank frames
            at the end of the episode in the rendered video file. A negative
            value has the same effects as 0 and no blank frames will be appended.
            This option has no effects when displaying the frames on the screen
            instead of recording to a file.
        render (bool): If False, then this function only evaluates the trained
            model without calling rendering functions. This value will be ignored
            if a ``record_file`` argument is provided.
        ignored_parameter_prefixes (list[str]): ignore the parameters whose
            name has one of these prefixes in the checkpoint.
    """
    train_dir = os.path.join(root_dir, 'train')

    ckpt_dir = os.path.join(train_dir, 'algorithm')
    checkpointer = Checkpointer(ckpt_dir=ckpt_dir, algorithm=algorithm)
    checkpointer.load(
        checkpoint_step,
        ignored_parameter_prefixes=ignored_parameter_prefixes,
        including_optimizer=False,
        including_replay_buffer=False)

    recorder = None
    if record_file is not None:
        recorder = VideoRecorder(
            env, append_blank_frames=append_blank_frames, path=record_file)
    elif render:
        # pybullet_envs need to render() before reset() to enable mode='human'
        env.render(mode='human')
    env.reset()

    # The behavior of some algorithms is based by scheduler using training
    # progress (e.g. VisitSoftmaxTemperatureByProgress for MCTSAlgorithm). So we
    # need to set a valid value for progress.
    # TODO: we may want to use a different progress value based on the actual
    # progress of the checkpoint or user provided progress value.
    Trainer._trainer_progress.set_progress(1.0)

    time_step = common.get_initial_time_step(env)
    algorithm.eval()
    policy_state = algorithm.get_initial_predict_state(env.batch_size)
    trans_state = algorithm.get_initial_transform_state(env.batch_size)
    episode_reward = 0.
    episode_length = 0
    episodes = 0
    metrics = [
        alf.metrics.AverageReturnMetric(
            buffer_size=num_episodes, reward_shape=env.reward_spec().shape),
        alf.metrics.AverageEpisodeLengthMetric(buffer_size=num_episodes),
    ]
    while episodes < num_episodes:
        next_time_step, policy_step, trans_state = _step(
            algorithm=algorithm,
            env=env,
            time_step=time_step,
            policy_state=policy_state,
            trans_state=trans_state,
            metrics=metrics,
            render=render,
            recorder=recorder,
            sleep_time_per_step=sleep_time_per_step)

        if not time_step.is_first():
            episode_length += 1
            episode_reward += time_step.reward.view(-1).float().cpu().numpy()

        if time_step.is_last():
            logging.info("episode_length=%s episode_reward=%s" %
                         (episode_length, episode_reward))
            episode_reward = 0.
            episode_length = 0
            episodes += 1

        policy_state = policy_step.state
        time_step = next_time_step

    for m in metrics:
        logging.info(
            "%s: %s", m.name,
            map_structure(
                lambda x: x.cpu().numpy().item()
                if x.ndim == 0 else x.cpu().numpy(), m.result()))
    if recorder:
        recorder.close()
    env.reset()
