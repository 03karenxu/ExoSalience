"""Utilities for deriving 1-D saliency heatmaps from Astronet feature maps."""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _normalize_map(saliency: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize each sample to [0, 1]."""
    saliency = saliency - saliency.min(dim=-1, keepdim=True).values
    saliency = saliency / (saliency.max(dim=-1, keepdim=True).values + eps)
    return saliency


def _split_head_weights(head_weight: torch.Tensor, global_channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split the linear head weights into global/local channel segments."""
    if head_weight.dim() == 2:
        # assume [1, C] layout for binary classifier
        head_weight = head_weight.squeeze(0)
    if head_weight.dim() != 1:
        raise ValueError(f"head_weight must be 1-D or 2-D, got {head_weight.shape}")

    w_global = head_weight[:global_channels]
    w_local = head_weight[global_channels:]
    return w_global, w_local


def compute_saliency(
    logits: torch.Tensor,
    probs: torch.Tensor,
    global_feature_map: torch.Tensor,
    local_feature_map: torch.Tensor,
    head_weight: torch.Tensor,
    *,
    apply_relu: bool = True,
    normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compute Grad-CAM-like saliency heatmaps for Astronet outputs.

    Args:
        logits: Model logits of shape `[B]` or `[B, 1]`.
        probs: Sigmoid probabilities matching `logits`.
        global_feature_map: Global branch activations `[B, C_g, T_g]`.
        local_feature_map: Local branch activations `[B, C_l, T_l]`.
        head_weight: Flattened linear head weights `[C_g + C_l]` (or `[1, C]`).
        apply_relu: If true, clip negative contributions to zero (CAM post-processing).
        normalize: If true, scale each saliency curve to `[0, 1]`.

    Returns:
        Dictionary with saliency curves and intermediate channel weights.
    """

    if logits.dim() == 2:
        logits = logits.squeeze(-1)
    if probs.dim() == 2:
        probs = probs.squeeze(-1)

    batch_size, global_channels, _ = global_feature_map.shape
    _batch_local, local_channels, _ = local_feature_map.shape
    if _batch_local != batch_size:
        raise ValueError("Global and local feature maps must share the batch dimension")

    w_global, w_local = _split_head_weights(head_weight, global_channels)

    # expand weights for broadcasting over time dimension
    w_global = w_global.view(1, global_channels, 1)
    w_local = w_local.view(1, local_channels, 1)

    saliency_global = (global_feature_map * w_global).sum(dim=1)
    saliency_local = (local_feature_map * w_local).sum(dim=1)

    if apply_relu:
        saliency_global = F.relu(saliency_global)
        saliency_local = F.relu(saliency_local)

    # scale by confidence if desired (broadcast over time axis)
    confidence = probs.view(batch_size, 1)
    saliency_global = saliency_global * confidence
    saliency_local = saliency_local * confidence

    if normalize:
        saliency_global = _normalize_map(saliency_global)
        saliency_local = _normalize_map(saliency_local)

    return {
        "saliency_global": saliency_global,
        "saliency_local": saliency_local,
        "weights_global": w_global.squeeze(0),
        "weights_local": w_local.squeeze(0),
        "logits": logits,
        "probs": probs,
    }

