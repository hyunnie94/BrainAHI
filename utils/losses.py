# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch
import torch.nn as nn
import numpy as np
import pdb
import torch.nn.functional as F
from torch import Tensor



class FocalLoss(nn.Module):
    """
    Focal Loss for classification (multi-class or binary)
    Args:
        alpha: class weight (float or list of floats)
        gamma: focusing parameter (default=2)
        reduction: "mean" or "sum"
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (N, C) logits
        targets: (N,) long
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)  # predicted prob of true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss



def compute_losses(outputs,
                   y_stage: Tensor | None = None,
                   y_event: Tensor | None = None,
                   w_stage: float = 1.0,
                   w_event: float = 1.0,
                   mask: Tensor | None = None):
    """
    outputs: dict from model.forward
      - stage_logits: (B,N,Ks) or None
      - event_logits: (B,N,Ke) or None
      - mask_expanded: (B,N) bool  (있으면 사용)
    y_stage, y_event: (B,N) with -1 padded
    """
    total = []
    logs = {}

    # (1) 마스크 준비
    if mask is None:
        mask = outputs.get("mask_expanded", None)
    if mask is None:
        if y_stage is not None:
            mask = (y_stage != -1)
        elif y_event is not None:
            mask = (y_event != -1)
    # mask가 없으면 이후 분모 0 방지를 위해 None 허용

    # (2) Stage loss
    if outputs.get("stage_logits", None) is not None and y_stage is not None and w_stage > 0:
        sl = outputs["stage_logits"]           # (B,N,Ks)
        B, N, Ks = sl.shape
        s_logits = sl.reshape(-1, Ks)          # (B*N, Ks)
        s_tgt = y_stage.reshape(-1).long()     # (B*N,)

        s_loss_all = F.cross_entropy(s_logits, s_tgt, ignore_index=-1, reduction='none')  # (B*N,)
        if mask is not None:
            m = mask.reshape(-1)
            s_loss = (s_loss_all[m]).sum() / m.sum().clamp_min(1)
        else:
            s_loss = s_loss_all.mean()

        if torch.isnan(s_loss) or not torch.isfinite(s_loss):
            s_loss = torch.zeros((), device=s_logits.device)
        logs["stage_loss"] = float(s_loss.detach().cpu())
        total.append(w_stage * s_loss)

    # (3) Event loss
    if outputs.get("event_logits", None) is not None and y_event is not None and w_event > 0:
        el = outputs["event_logits"]           # (B,N,Ke)
        B, N, Ke = el.shape
        e_logits = el.reshape(-1, Ke)          # (B*N, Ke)
        e_tgt = y_event.reshape(-1).long()     # (B*N,)

        e_loss_all = F.cross_entropy(e_logits, e_tgt, ignore_index=-1, reduction='none')
        if mask is not None:
            m = mask.reshape(-1)
            e_loss = (e_loss_all[m]).sum() / m.sum().clamp_min(1)
        else:
            e_loss = e_loss_all.mean()

        if torch.isnan(e_loss) or not torch.isfinite(e_loss):
            e_loss = torch.zeros((), device=e_logits.device)
        logs["event_loss"] = float(e_loss.detach().cpu())
        total.append(w_event * e_loss)

    # (4) 총합 + NaN guard
    if len(total) == 0:
        loss = torch.zeros((), device=(outputs.get("event_logits") or outputs.get("stage_logits")).device)
    else:
        loss = sum(total)
        if torch.isnan(loss) or not torch.isfinite(loss):
            loss = torch.zeros((), device=loss.device)

    logs["loss"] = float(loss.detach().cpu())
    return loss, logs



# ========================
# 손실 함수 in phase2 (Log + Weighted MAE)
# ========================

def u_weight(true, a_low=3.0, a_high=6.0, beta=0.05, high_knee=25.0):
    """Normal(low)과 Severe(high) 모두 강조"""
    w_low  = 1 + a_low  * torch.exp(-beta * true)
    w_high = 1 + a_high * torch.sigmoid(beta * (true - high_knee))
    return w_low + w_high

def weighted_mae_loss(pred_log, true_log):
    """
    pred_log, true_log : log(1+AHI)
    복원된 AHI 단위로 가중 적용
    """
    pred = torch.expm1(pred_log)
    true = torch.expm1(true_log)
    w = u_weight(true)
    return (w * torch.abs(pred - true)).mean()