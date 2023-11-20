# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union
import numpy as np

import torch
import torch.nn as nn
from torch import Tensor

from mmdet3d.registry import MODELS
from mmdet.models import weight_reduce_loss


def inner_outer(t0, t1, y1):
    """Construct inner and outer measures on (t1, y1) for t0."""
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]),
                         torch.cumsum(y1, dim=-1)],
                        dim=-1)
    idx_left = torch.searchsorted(t1, t0, side='left')
    idx_right = torch.searchsorted(t1, t0, side='right')

    cy1_lo = torch.take_along_dim(cy1, idx_left, dim=-1)
    cy1_hi = torch.take_along_dim(cy1, idx_right, dim=-1)

    y0_outer = cy1_hi[..., 1:] - cy1_lo[..., :-1]

    return y0_outer


def lossfun_outer(t, w, t_hat, w_hat, eps=torch.finfo(torch.float32).eps):
  """The proposal weight should be an upper envelope on the nerf weight."""
  w_outer = inner_outer(t, t_hat, w_hat)
  # We assume w_inner <= w <= w_outer. We don't penalize w_inner because it's
  # more effective to pull w_outer up than it is to push w_inner down.
  # Scaled half-quadratic loss that gives a constant gradient at w_outer = 0.
  return torch.maximum(0, w - w_outer)**2 / (w + eps)


@MODELS.register_module()
class ProposalLoss(nn.Module):
    """Compute ProposalLoss loss.
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 eps=1e-6,
                 loss_name='loss_proposal'):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name

    def forward(
        self,
        t,
        w,
        t_hat,
        w_hat
    ):
        """
        t_hat, w_hat produced by proposal nerf head
        t, w produced by fine nerf head
        """

        loss = self.loss_weight * lossfun_outer(t, w, t_hat, w_hat)
        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name