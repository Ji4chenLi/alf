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

from functools import partial

import alf
from alf.examples import sac_conf
from alf.algorithms.one_step_qr_loss import OneStepQRLoss
from alf.algorithms.qrsac_algorithm import QRSacAlgorithm
from alf.networks import NormalProjectionNetwork, ActorDistributionNetwork, CriticNetwork
from alf.optimizers import AdamTF
from alf.utils.math_ops import clipped_exp
from alf.utils.losses import element_wise_squared_loss
from alf.tensor_specs import TensorSpec


# environment config
alf.config(
    'create_environment', env_name="Humanoid-v2", num_parallel_environments=1)

# algorithm config
fc_layer_params = (256, 256)

actor_network_cls = partial(
    ActorDistributionNetwork,
    fc_layer_params=fc_layer_params,
    continuous_projection_net_ctor=partial(
        NormalProjectionNetwork,
        state_dependent_std=True,
        scale_distribution=True,
        std_transform=partial(
            clipped_exp, clip_value_min=-10, clip_value_max=2)))

num_quatiles = 50

critic_network_cls = partial(
    CriticNetwork, joint_fc_layer_params=fc_layer_params, output_tensor_spec=TensorSpec((num_quatiles,)))

alf.config('calc_default_target_entropy', min_prob=0.184)

alf.config(
    'QRSacAlgorithm',
    critic_loss_ctor=OneStepQRLoss,
    actor_network_cls=actor_network_cls,
    critic_network_cls=critic_network_cls,
    target_update_tau=0.005,
    actor_optimizer=AdamTF(lr=3e-4),
    critic_optimizer=AdamTF(lr=3e-4),
    alpha_optimizer=AdamTF(lr=3e-4))

# training config
alf.config('Agent', rl_algorithm_cls=QRSacAlgorithm)

alf.config(
    'TrainerConfig',
    initial_collect_steps=10000,
    mini_batch_length=2,
    unroll_length=1,
    mini_batch_size=256,
    num_updates_per_train_iter=1,
    num_iterations=2500000,
    num_checkpoints=1,
    evaluate=True,
    eval_interval=1000,
    num_eval_episodes=5,
    debug_summaries=False,
    random_seed=0,
    summarize_grads_and_vars=False,
    summary_interval=1000,
    replay_buffer_length=1000000)
