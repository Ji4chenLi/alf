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

import torch
import torch.nn as nn

import alf
from alf.data_structures import LossInfo, StepType
from alf.utils.losses import element_wise_huber_loss
from alf.utils import tensor_utils, value_ops
from alf.utils.summary_utils import safe_mean_hist_summary


@alf.configurable
class OneStepQRLoss(nn.Module):
    """Temporal difference loss."""

    def __init__(self,
                 gamma=0.99,
                 sum_over_quantiles=True,
                 debug_summaries=False,
                 name="OneStepQRLoss"):
        r"""

        Args:
            gamma (float|list[float]): A discount factor for future rewards. For
                multi-dim reward, this can also be a list of discounts, each
                discount applies to a reward dim.
            td_errors_loss_fn (Callable): A function for computing the TD errors
                loss. This function takes as input the target and the estimated
                Q values and returns the loss for each element of the batch.
            sum_over_quantiles (bool): Whether to sum over the quantiles
            debug_summaries (bool): True if debug summaries should be created.
            name (str): The name of this loss.
        """
        super().__init__()

        self._name = name
        self._gamma = torch.tensor(gamma)
        self._debug_summaries = debug_summaries
        self._sum_over_quantiles = sum_over_quantiles

    @property
    def gamma(self):
        """Return the :math:`\gamma` value for discounting future rewards.

        Returns:
            Tensor: a rank-0 or rank-1 (multi-dim reward) floating tensor.
        """
        return self._gamma.clone()

    def forward(self, info, value, target_value):
        """Calculate the loss.

        The first dimension of all the tensors is time dimension and the second
        dimesion is the batch dimension.

        Args:
            info (namedtuple): experience collected from ``unroll()`` or
                a replay buffer. All tensors are time-major. ``info`` should
                contain the following fields:
                - reward:
                - step_type:
                - discount:
            value (torch.Tensor): the time-major tensor for the value at each time
                step. The loss is between this and the calculated return.
            target_value (torch.Tensor): the time-major tensor for the value at
                each time step. This is used to calculate return. ``target_value``
                can be same as ``value``.
        Returns:
            LossInfo: with the ``extra`` field same as ``loss``.
        """
        if info.reward.ndim == 3:
            # [T, B, D] or [T, B, 1]
            discounts = info.discount.unsqueeze(-1) * self._gamma
        else:
            # [T, B]
            discounts = info.discount * self._gamma

        returns = value_ops.discounted_return(
            rewards=info.reward,
            values=target_value,
            step_types=info.step_type,
            discounts=discounts)
        value = value[:-1]

        if self._debug_summaries and alf.summary.should_record_summaries():
            mask = info.step_type[:-1] != StepType.LAST
            with alf.summary.scope(self._name):

                def _summarize(v, r, td, suffix):
                    alf.summary.scalar(
                        "explained_variance_of_return_by_value" + suffix,
                        tensor_utils.explained_variance(v, r, mask))
                    safe_mean_hist_summary('values' + suffix, v, mask)
                    safe_mean_hist_summary('returns' + suffix, r, mask)
                    safe_mean_hist_summary("td_error" + suffix, td, mask)

                if value.ndim == 2:
                    _summarize(value, returns, returns - value, '')
                else:
                    td = returns - value
                    for i in range(value.shape[2]):
                        suffix = '/' + str(i)
                        _summarize(value[..., i], returns[..., i], td[..., i],
                                   suffix)

        # Get the cummulative prob
        assert value.ndim == 2
        n_quantiles = value.shape[-1]
        # Cumulative probabilities to calculate quantiles.
        cum_prob = (torch.arange(
            n_quantiles, device=n_quantiles.device, dtype=torch.float
            ) + 0.5) / n_quantiles

        # For QR-DQN, current_quantiles have a shape
        # (T, B, n_quantiles), and make cum_prob
        # broadcastable to
        # (T, B, n_quantiles, n_target_quantiles)
        cum_prob = cum_prob.view(1, -1, 1)

        element_wise_delta = returns.detach().unsqueeze(-2) - value.unsqueeze(-1)
        huber_loss = element_wise_huber_loss(element_wise_delta)
        # (T, B, n_quantiles, n_target_quantiles)
        loss = torch.abs(cum_prob - (element_wise_delta.detach() < 0).float()) * huber_loss

        if self._sum_over_quantiles:
            loss = loss.sum(dim=-2)
        else:
            loss = loss.mean(dim=-2)

        loss.mean(dim=2)

        # The shape of the loss expected by Algorith.update_with_gradient is
        # [T, B], so we need to augment it with additional zeros.
        loss = tensor_utils.tensor_extend_zero(loss)
        return LossInfo(loss=loss, extra=loss)
