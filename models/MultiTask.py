# MultiTask.py
# Ablation-friendly Multi-Task / Single-Task switchable model
# - task_mode: {"mtl", "stage", "event"}
# - use_stage_context: whether to feed stage logits to event head (MTL only)
# - forward returns a dict for consistent training loops

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.ModernTCN_Layer import Flatten_Head
from models.ModernTCN import ModernTCN


# ------------------------------
# Blocks
# ------------------------------
class SimpleAttentionBlock(nn.Module):
    """
    ModernTCN feature (B, M, D, N) -> (B, N, M*D) and run MultiheadAttention.
    Pools/reshapes time to target_N = num_epochs_per_window * 30s.
    Returns (B, M, D, N) features + attention weights.
    """
    def __init__(self, d_model, num_heads=4, dropout=0.1, num_epochs_per_window=1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads,
                                         dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.num_epochs_per_window = num_epochs_per_window
        self.seconds_per_epoch = 30

    def forward(self, x: Tensor, mask=None) -> tuple[Tensor, Tensor | None]:
        # x: (B, M, D, N)
        B, M, D, N = x.shape

        # Time pooling to fixed target_N
        target_N = self.num_epochs_per_window * self.seconds_per_epoch
        if N != target_N:
            if N % target_N != 0:
                raise ValueError(f"N ({N}) must be divisible by target_N ({target_N})")
            pool_size = N // target_N
            x = x.view(B, M, D, target_N, pool_size).mean(dim=-1)  # (B,M,D,target_N)
            N = target_N

        # (B,N,M*D)
        x_reshaped = x.permute(0, 3, 1, 2).contiguous().view(B, N, M * D)

        if mask is not None:
            out, attn_w = self.mha(x_reshaped, x_reshaped, x_reshaped,
                                   key_padding_mask=(mask == 0), need_weights=True)
        else:
            out, attn_w = self.mha(x_reshaped, x_reshaped, x_reshaped, need_weights=True)

        out = self.ln(out + x_reshaped)  # residual
        out = out.view(B, N, M, D).permute(0, 2, 3, 1).contiguous()  # (B,M,D,N)
        return out, attn_w


class StageHead(nn.Module):
    """
    Classify sleep stage per 30s epoch.
    Input: (B, M, D, N) -> project over time to (B, E, C) where E=num_epochs_per_window
    """
    def __init__(self, in_channels, num_stage_classes=5, dropout=0.1, num_epochs_per_window=1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_stage_classes = num_stage_classes
        self.num_epochs_per_window = num_epochs_per_window

        h1 = max(4, in_channels // 2)
        h2 = max(4, h1 // 2)
        self.linear1 = nn.Linear(in_channels, h1)
        self.act1 = nn.GELU()
        self.linear2 = nn.Linear(h1, h2)
        self.act2 = nn.GELU()
        self.linear3 = nn.Linear(h2, num_stage_classes)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, M, D, N)
        B, M, D, N = x.shape
        x_nd = x.permute(0, 3, 1, 2).contiguous().view(B, N, M * D)  # (B,N,M*D)
        x_nd = self.dropout(x_nd)
        z = self.act1(self.linear1(x_nd))
        z = self.act2(self.linear2(z))
        logits = self.linear3(z)  # (B,N,C)

        # Average-pool each epoch-length (30s) to one logit per epoch
        if N % self.num_epochs_per_window != 0:
            raise ValueError(f"N ({N}) must be divisible by num_epochs_per_window ({self.num_epochs_per_window})")
        seconds_per_epoch = N // self.num_epochs_per_window
        logits = logits.view(B, self.num_epochs_per_window, seconds_per_epoch, self.num_stage_classes).mean(dim=2)
        return logits  # (B,E,C)


class AdditiveAttention(nn.Module):
    """Additive attention between features and stage logits (context)."""
    def __init__(self, feat_dim, stage_dim, attention_dim):
        super().__init__()
        self.feat_proj = nn.Linear(feat_dim, attention_dim)
        self.stage_proj = nn.Linear(stage_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, x_feat, stage_logits, num_epochs_per_window=1):
        """
        x_feat: (B, N, feat_dim)
        stage_logits: (B, E, stage_dim) or (B, N, stage_dim) aggregated to (B,N,*)
        """
        original_shape = x_feat.shape
        if len(original_shape) == 2:
            x_feat = x_feat.unsqueeze(1)
        B, N, F = x_feat.shape

        # Expand stage logits from per-epoch to per-second
        seconds_per_epoch = N // num_epochs_per_window
        if N % num_epochs_per_window != 0:
            raise ValueError(f"N ({N}) must be divisible by num_epochs_per_window ({num_epochs_per_window})")
        stage_expanded = stage_logits.unsqueeze(2).repeat(1, 1, seconds_per_epoch, 1).view(B, N, -1)

        a = self.v(torch.tanh(self.feat_proj(x_feat) + self.stage_proj(stage_expanded)))  # (B,N,1)
        w = torch.softmax(a, dim=1)
        out = x_feat * w
        if len(original_shape) == 2:
            out = out.squeeze(1)
        return out, w


class EventHead(nn.Module):
    """
    Event detection head. Can use stage logits as context (MTL) via additive attention.
    Outputs per-epoch logits by default; set use_1sec_labels=True for per-second logits.
    """
    def __init__(self, feat_dim=160, stage_dim=5, num_event_classes=2,
                 attention_dim=128, dropout=0.1, num_epochs_per_window=1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention = AdditiveAttention(feat_dim, stage_dim, attention_dim)
        self.fusion = nn.Linear(feat_dim * 2, feat_dim)

        h1 = max(4, feat_dim // 2)
        h2 = max(4, h1 // 2)
        self.fc1 = nn.Linear(feat_dim, h1)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(h1, h2)
        self.act2 = nn.GELU()
        self.cls = nn.Linear(h2, num_event_classes)

        self.hidden_dim2 = h2
        self.num_epochs_per_window = num_epochs_per_window
        self.seconds_per_epoch = 30

    def forward(self, x_feat, stage_logits=None, use_1sec_labels=False, return_features=False):
        # x_feat: (B, M, D, N)
        B, M, D, N = x_feat.shape
        Fdim = M * D

        feat = x_feat.permute(0, 3, 1, 2).contiguous().view(B, N, Fdim)  # (B,N,F)

        attn_w = None
        if stage_logits is not None:
            stage_logits = torch.softmax(stage_logits, dim=-1)
            attended, attn_w = self.attention(feat, stage_logits, num_epochs_per_window=self.num_epochs_per_window)
            feat = self.fusion(torch.cat([feat, attended], dim=-1))  # (B,N,F)

        # time pooling to target_N = E*30
        target_N = self.num_epochs_per_window * self.seconds_per_epoch
        if N != target_N:
            if N % target_N != 0:
                raise ValueError(f"N ({N}) must be divisible by target_N ({target_N})")
            pool = N // target_N
            feat = feat.view(B, target_N, pool, Fdim).mean(dim=2)
            N = target_N

        z = self.act1(self.fc1(feat))
        z = self.act2(self.fc2(z))
        z = self.dropout(z)

        if use_1sec_labels:
            event_logits = self.cls(z)  # (B,N,2)
            if return_features:
                return z, event_logits, attn_w
            return event_logits, attn_w

        # Per-epoch logits
        if N % self.num_epochs_per_window != 0:
            raise ValueError(f"N ({N}) must be divisible by num_epochs_per_window ({self.num_epochs_per_window})")
        spe = N // self.num_epochs_per_window
        ep_feat = z.view(B, self.num_epochs_per_window, spe, self.hidden_dim2).mean(dim=2)  # (B,E,H)
        event_logits = self.cls(ep_feat)  # (B,E,2)

        if return_features:
            return z, event_logits, attn_w  # z: (B,N,H) for phase-2 (pool later)
        return event_logits, attn_w


# ------------------------------
# Model wrapper (Ablation-friendly)
# ------------------------------
class MultiTaskModel(nn.Module):
    """
    Shared backbone (ModernTCN) + optional attention
    Heads:
      - StageHead (epoch-level)
      - EventHead (epoch- or second-level)
    Ablation switches:
      - task_mode: {"mtl","stage","event"}
      - use_stage_context: whether to feed stage logits to event head (MTL only)
    """
    def __init__(self,
                 backbone_cfg,
                 num_stage_classes=5,
                 num_event_classes=2,
                 event_window='30s',
                 use_attention=True,
                 d_model_for_attn=None,
                 dropout=0.1,
                 num_epochs_per_window=1,
                 task_mode: str = "mtl",
                 use_stage_context: bool = True):
        super().__init__()
        self.event_window = event_window
        self.num_epochs_per_window = num_epochs_per_window
        self.seconds_per_epoch = 30
        self.task_mode = task_mode
        self.use_stage_context = use_stage_context

        # backbone without its built-in classifier
        bb_cfg = backbone_cfg.copy()
        bb_cfg['class_num'] = None
        self.shared_backbone = ModernTCN(**bb_cfg)

        # attention on backbone features (optional)
        if use_attention:
            if d_model_for_attn is None:
                D = backbone_cfg['dims'][-1]
                M = backbone_cfg['nvars']
                d_model_for_attn = M * D
            self.simple_attn = SimpleAttentionBlock(d_model=d_model_for_attn,
                                                    dropout=dropout,
                                                    num_epochs_per_window=num_epochs_per_window)
        else:
            self.simple_attn = None

        in_feat = backbone_cfg['dims'][-1] * backbone_cfg['nvars']
        self.stage_head = StageHead(in_channels=in_feat,
                                    num_stage_classes=num_stage_classes,
                                    dropout=dropout,
                                    num_epochs_per_window=num_epochs_per_window)

        self.event_head = EventHead(feat_dim=in_feat,
                                    stage_dim=num_stage_classes,
                                    num_event_classes=num_event_classes,
                                    attention_dim=128,
                                    dropout=dropout,
                                    num_epochs_per_window=num_epochs_per_window)

    # ---- convenience toggles for ablation ----
    def set_mode(self, mode: str, use_stage_context: bool | None = None):
        assert mode in {"mtl", "stage", "event"}
        self.task_mode = mode
        if use_stage_context is not None:
            self.use_stage_context = use_stage_context

    # ---- forward ----
    def forward(self, x_30s: Tensor, mask: Tensor | None = None,
                use_1sec_labels: bool = False, return_features: bool = False):
        """
        Returns dict:
          {
            "stage_logits": (B,E,C) or None,
            "event_logits": (B,E,2)/(B,N,2) or None,
            "backbone_attn": attn matrix or None,
            "event_attn": attn vector or None,
            "event_features": (B,N,H) if return_features else None,
            "mask_expanded": (B,N) or None
          }
        """
        # 1) backbone features (B,M,D,N0)
        shared = self.shared_backbone.forward_feature(x_30s)
        B, M, D, N = shared.shape

        # fix time length to E*30
        target_N = self.num_epochs_per_window * self.seconds_per_epoch
        if N != target_N:
            if N % target_N != 0:
                raise ValueError(f"N ({N}) must be divisible by target_N ({target_N})")
            pool = N // target_N
            shared = shared.view(B, M, D, target_N, pool).mean(dim=-1)
            N = target_N

        # expand mask to seconds
        mask_expanded = None
        if mask is not None:
            seconds_per_epoch = N // self.num_epochs_per_window
            mask_expanded = mask.unsqueeze(-1).repeat(1, 1, seconds_per_epoch).view(B, N)

        # optional attention on shared features
        if self.simple_attn is not None:
            shared, bb_attn = self.simple_attn(shared, mask=mask_expanded)
        else:
            bb_attn = None

        # 2) heads by mode
        stage_logits = None
        event_logits = None
        event_attn = None
        event_features = None

        if self.task_mode in {"mtl", "stage"}:
            stage_logits = self.stage_head(shared)  # (B,E,C)

        if self.task_mode in {"mtl", "event"}:
            stage_ctx = None
            if self.task_mode == "mtl" and self.use_stage_context:
                stage_ctx = stage_logits
            ev_out = self.event_head(shared, stage_ctx,
                                     use_1sec_labels=use_1sec_labels,
                                     return_features=return_features)
            if return_features:
                event_features, event_logits, event_attn = ev_out
            else:
                event_logits, event_attn = ev_out

        return {
            "stage_logits": stage_logits,
            "event_logits": event_logits,
            "backbone_attn": bb_attn,
            "event_attn": event_attn,
            "event_features": event_features,
            "mask_expanded": mask_expanded,
        }

