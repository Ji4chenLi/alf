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
"""QR Soft Actor Critic algorithm."""

import torch

import alf
from alf.algorithms.sac_algorithm import SacAlgorithm, SacInfo, ActionType, \
    SacCriticState, SacActorInfo, SacCriticInfo
from alf.data_structures import TimeStep, LossInfo
from alf.nest import nest
import alf.nest.utils as nest_utils
from alf.utils import losses, math_ops


@alf.configurable
class QRSacAlgorithm(SacAlgorithm):
    """Quantile Soft Actor Critic algorithm
    """

    def _compute_critics(self,
                         critic_net,
                         observation,
                         action,
                         critics_state,
                         replica_min=True):
        assert  self._act_type == ActionType.Continuous
        observation = (observation, action)
        # continuous: critics shape [B, replicas, num_quantiles]
        critics, critics_state = critic_net(observation, state=critics_state)

        assert not self.has_multidim_reward()

        if replica_min:
            # Select the replica with the minimum mean of the q_value
            # shape [B, replicas]
            critic_mean = critics.mean(dim=-1)
            # shape [B]
            index = torch.min(critic_mean, dim=-1)[1]
            critics = critics[torch.arange(len(index)), index]

        # The returns have the following shapes in different circumstances:
        # [replica_min=True]
        #   continuous: critics shape [B, num_quantiles]
        # [replica_min=False]
        #   continuous: critics shape [B, replicas, num_quantiles]
        return critics, critics_state

    def _actor_train_step(self, inputs: TimeStep, state, action, critics,
                          log_pi, action_distribution):
        neg_entropy = sum(nest.flatten(log_pi))

        assert self._act_type == ActionType.Continuous
        q_value, critics_state = self._compute_critics(
            self._critic_networks, inputs.observation, action, state)
        continuous_log_pi = log_pi
        cont_alpha = torch.exp(self._log_alpha).detach()

        assert q_value.ndim == 2, q_value.shape
        # Expect the shape to be [B, n_quantiles].
        # We need to first take the mean across the quantiles to 
        # obtain the mean of q_value
        dqda = nest_utils.grad(action, q_value.mean(-1).sum())

        def actor_loss_fn(dqda, action):
            if self._dqda_clipping:
                dqda = torch.clamp(dqda, -self._dqda_clipping,
                                   self._dqda_clipping)
            loss = 0.5 * losses.element_wise_squared_loss(
                (dqda + action).detach(), action)
            return loss.sum(list(range(1, loss.ndim)))

        actor_loss = nest.map_structure(actor_loss_fn, dqda, action)
        actor_loss = math_ops.add_n(nest.flatten(actor_loss))
        actor_info = LossInfo(
            loss=actor_loss + cont_alpha * continuous_log_pi,
            extra=SacActorInfo(actor_loss=actor_loss, neg_entropy=neg_entropy))
        return critics_state, actor_info

    def _critic_train_step(self, inputs: TimeStep, state: SacCriticState,
                           rollout_info: SacInfo, action, action_distribution):
        critics, critics_state = self._compute_critics(
            self._critic_networks,
            inputs.observation,
            rollout_info.action,
            state.critics,
            replica_min=False)

        target_critics, target_critics_state = self._compute_critics(
            self._target_critic_networks,
            inputs.observation,
            action,
            state.target_critics)

        assert self._act_type == ActionType.Continuous

        target_critic = target_critics.detach()

        state = SacCriticState(
            critics=critics_state, target_critics=target_critics_state)
        info = SacCriticInfo(critics=critics, target_critic=target_critic)

        return state, info
