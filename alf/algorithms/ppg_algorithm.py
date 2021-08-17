# Copyright (c) 2021 Horizon Robotics and ALF Contributors. All Rights Reserved.
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
"""Phasic Policy Gradient Algorithm."""

import torch
import torch.distributions as td

from typing import Optional, Tuple

import alf
from alf.algorithms.off_policy_algorithm import OffPolicyAlgorithm
from alf.algorithms.ppo_algorithm import PPOAlgorithm
from alf.algorithms.config import TrainerConfig
from alf.algorithms.ppo_loss import PPOLoss
from alf.algorithms.algorithm import Loss
from alf.networks.encoding_networks import EncodingNetwork
from alf.networks.dual_actor_value_network import DualActorValueNetwork, DualActorValueNetworkState, ACNetwork
from alf.experience_replayers.experience_replay import OnetimeExperienceReplayer
from alf.data_structures import namedtuple, TimeStep, AlgStep, LossInfo
from alf.tensor_specs import TensorSpec

from alf.utils import common, dist_utils, value_ops, tensor_utils
from alf.utils.losses import element_wise_huber_loss

PPGRolloutInfo = namedtuple(
    'PPGRolloutInfo', [
        'action_distribution',
        'action',
        'value',
        'aux',
        'step_type',
        'discount',
        'reward',
        'reward_weights',
    ],
    default_value=())

PPGTrainInfo = namedtuple(
    'PPGTrainInfo',
    [
        'action',  # Sampled from the behavior policy
        'rollout_action_distribution',  # Evaluation of the behavior policy
        'action_distribution',  # Evaluation of the target policy
        'value',
        'aux',
        'step_type',
        'discount',
        'reward',
        'advantages',
        'returns',
        'reward_weights',
    ],
    default_value=())


def merge_rollout_into_train_info(rollout_info: PPGRolloutInfo,
                                  train_info: PPGTrainInfo) -> PPGTrainInfo:
    return train_info._replace(
        step_type=rollout_info.step_type,
        reward=rollout_info.reward,
        discount=rollout_info.discount,
        action_distribution=rollout_info.action_distribution,
        value=rollout_info.value,
        aux=rollout_info.aux,
        reward_weights=rollout_info.reward_weights)


PPGAuxPhaseLossInfo = namedtuple('PPGAuxPhaseLossInfo', [
    'td_loss_actual',
    'td_loss_aux',
    'policy_kl_loss',
])


@alf.configurable
class PPGAuxPhaseLoss(Loss):
    def __init__(self,
                 td_error_loss_fn=element_wise_huber_loss,
                 policy_kl_loss_weight: float = 1.0,
                 name='PPGAuxPhaseLoss'):
        super().__init__(name=name)
        self._td_error_loss_fn = td_error_loss_fn
        self._policy_kl_loss_weight = policy_kl_loss_weight

    def forward(self, info: PPGTrainInfo):
        td_loss_actual = self._td_error_loss_fn(info.returns.detach(),
                                                info.value)
        td_loss_aux = self._td_error_loss_fn(info.returns.detach(), info.aux)
        policy_kl_loss = td.kl_divergence(info.rollout_action_distribution,
                                          info.action_distribution)
        loss = td_loss_actual + td_loss_aux + self._policy_kl_loss_weight
        return LossInfo(
            loss=loss,
            extra=PPGAuxPhaseLossInfo(
                td_loss_actual=td_loss_actual,
                td_loss_aux=td_loss_aux,
                policy_kl_loss=policy_kl_loss))


@alf.configurable
class PPGAlgorithm(OffPolicyAlgorithm):
    """PPG Algorithm.
    """

    def __init__(self,
                 observation_spec: TensorSpec,
                 action_spec: TensorSpec,
                 reward_spec=TensorSpec(()),
                 encoding_network_ctor: callable = EncodingNetwork,
                 env=None,
                 config: Optional[TrainerConfig] = None,
                 debug_summaries: bool = False,
                 main_optimizer: Optional[torch.optim.Optimizer] = None,
                 aux_optimizer: Optional[torch.optim.Optimizer] = None,
                 name: str = "PPGAlgorithm"):
        """
        Args:
            observation_spec (nested TensorSpec): representing the observations.
            action_spec (nested BoundedTensorSpec): representing the actions.
            reward_spec (TensorSpec): a rank-1 or rank-0 tensor spec representing
                the reward(s).
            reward_weights (None|list[float]): this is only used when the reward is
                multidimensional. In that case, the weighted sum of the v values
                is used for training the actor if reward_weights is not None.
                Otherwise, the sum of the v values is used.
            env (Environment): The environment to interact with. env is a batched
                environment, which means that it runs multiple simulations
                simultateously. env only needs to be provided to the root
                Algorithm.
            epsilon_greedy (float): a floating value in [0,1], representing the
                chance of action sampling instead of taking argmax. This can
                help prevent a dead loop in some deterministic environment like
                Breakout. Only used for evaluation. If None, its value is taken
                from ``alf.get_config_value(TrainerConfig.epsilon_greedy)``
            config (TrainerConfig): config for training. config only needs to be
                provided to the algorithm which performs ``train_iter()`` by
                itself.
            actor_network_ctor (Callable): Function to construct the actor network.
                ``actor_network_ctor`` needs to accept ``input_tensor_spec`` and
                ``action_spec`` as its arguments and return an actor network.
                The constructed network will be called with ``forward(observation, state)``.
            value_network_ctor (Callable): Function to construct the value network.
                ``value_network_ctor`` needs to accept ``input_tensor_spec`` as
                its arguments and return a value netwrok. The contructed network
                will be called with ``forward(observation, state)`` and returns
                value tensor for each observation given observation and network
                state.
            loss (None|ActorCriticLoss): an object for calculating loss. If
                None, a default loss of class loss_class will be used.
            loss_class (type): the class of the loss. The signature of its
                constructor: ``loss_class(debug_summaries)``
            optimizer (torch.optim.Optimizer): The optimizer for training
            debug_summaries (bool): True if debug summaries should be created.
            name (str): Name of this algorithm.
        """

        # dual_actor_value_network = DualActorValueNetwork(
        #     observation_spec=observation_spec,
        #     action_spec=action_spec,
        #     encoding_network_ctor=encoding_network_ctor,
        #     is_sharing_encoder=False)

        dual_actor_value_network = ACNetwork(
            observation_spec=observation_spec, action_spec=action_spec)

        super().__init__(
            config=config,
            env=env,
            observation_spec=observation_spec,
            action_spec=action_spec,
            reward_spec=reward_spec,
            predict_state_spec=dual_actor_value_network.state_spec,
            # TODO(breakds): Value heads need state as well
            train_state_spec=dual_actor_value_network.state_spec)

        # TODO(breakds): Make this more flexible to allow recurrent networks
        # TODO(breakds): Make this more flexible to allow separate networks
        # TODO(breakds): Add other more complicated network parameters
        # TODO(breakds): Contiuous cases should be handled
        self._dual_actor_value_network = dual_actor_value_network
        self._shadow_network = dual_actor_value_network.copy()
        # TODO(breakds): Put this to the configuration
        self._loss = PPOLoss(debug_summaries=debug_summaries)
        self._aux_phase_loss = PPGAuxPhaseLoss()

        # TODO(breakds): Try not to maintain states in algorithm itself. The
        # less stateful the cleaner.
        self.add_optimizer(main_optimizer, [self._dual_actor_value_network])
        self.add_optimizer(aux_optimizer, [self._shadow_network])

        # HACK: Additional Exp Replayer
        self._in_aux_phase = False
        self._exp_replayer_backup = None
        self._aux_exp_replayer = OnetimeExperienceReplayer()
        self._observers.append(self._aux_exp_replayer.observe)

    @property
    def on_policy(self) -> bool:
        return False

    def evaluate_network(self,
                         inputs: TimeStep,
                         state: DualActorValueNetworkState,
                         use_shadow: bool = False) -> AlgStep:
        network = self._shadow_network if use_shadow else self._dual_actor_value_network
        action_distribution, value, aux, state = network(
            inputs.observation, state=state)

        action = dist_utils.sample_action_distribution(action_distribution)

        return AlgStep(
            output=action,
            state=state,
            info=PPGRolloutInfo(
                action_distribution=action_distribution,
                action=common.detach(action),
                value=value,
                aux=aux,
                step_type=inputs.step_type,
                discount=inputs.discount,
                reward=inputs.reward,
                reward_weights=()))

    def rollout_step(self, inputs: TimeStep,
                     state: DualActorValueNetworkState) -> AlgStep:
        return self.evaluate_network(inputs, state)

    def preprocess_experience(
            self,
            inputs: TimeStep,  # nest of [B, T, ...]
            rollout_info: PPGRolloutInfo,
            batch_info) -> Tuple[TimeStep, PPGTrainInfo]:
        # Initialize the update epoch at the beginning of each iteration

        # Here inputs is a nest of tensors representing a batch of trajectories.
        # Each tensor is expected to be of shape [B, T] or [B, T, ...], where T
        # stands for the temporal extent, where B is the the size of the batch.

        with torch.no_grad():
            if rollout_info.reward.ndim == 3:
                # [B, T, D] or [B, T, 1]
                discounts = rollout_info.discount.unsqueeze(
                    -1) * self._loss.gamma
            else:
                # [B, T]
                discounts = rollout_info.discount * self._loss.gamma

            advantages = value_ops.generalized_advantage_estimation(
                rewards=rollout_info.reward,
                values=rollout_info.value,
                step_types=rollout_info.step_type,
                discounts=discounts,
                td_lambda=self._loss._lambda,
                time_major=False)
            advantages = tensor_utils.tensor_extend_zero(advantages, dim=1)
            returns = rollout_info.value + advantages

        return inputs, merge_rollout_into_train_info(
            rollout_info,
            PPGTrainInfo(
                action=rollout_info.action,
                rollout_action_distribution=rollout_info.action_distribution,
                returns=returns,
                advantages=advantages))

    def train_step(self, inputs: TimeStep, state: DualActorValueNetworkState,
                   prev_train_info: PPGTrainInfo) -> AlgStep:
        if self._in_aux_phase:
            alg_step = self.evaluate_network(inputs, state, use_shadow=True)
        else:
            alg_step = self.evaluate_network(inputs, state)

        return alg_step._replace(
            info=merge_rollout_into_train_info(alg_step.info, prev_train_info))

    def calc_loss(self, info: PPGTrainInfo) -> LossInfo:
        if self._in_aux_phase:
            return self._aux_phase_loss(info)
        else:
            return self._loss(info)

    def switch_to_aux_phase(self):
        self._in_aux_phase = True
        self._shadow_network.load_state_dict(
            self._dual_actor_value_network.state_dict())
        self._exp_replayer_backup = self._exp_replayer
        self._exp_replayer = self._aux_exp_replayer

    def after_train_iter(self, root_inputs, rollout_info: PPGTrainInfo):
        if self._in_aux_phase:
            self._in_aux_phase = False
            self._dual_actor_value_network.load_state_dict(
                self._shadow_network.state_dict())
            self._exp_replayer = self._exp_replayer_backup
