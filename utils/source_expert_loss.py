from __future__ import annotations

import torch


def smooth_six_reward(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mean_target: float,
    temperature: float,
) -> torch.Tensor:
    """Generation-weighted differentiable reward for capacity error <= 6%."""
    if mean_target <= 0.0:
        raise ValueError("mean_target must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    error = torch.abs(prediction - target)
    within_six = torch.sigmoid((0.06 - error) / float(temperature))
    return (target / float(mean_target)) * within_six


def group_balanced_pure_six_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    valid: torch.Tensor,
    mean_targets: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Average pure-six reward equally across groups with observed targets."""
    group_rewards = []
    for group_index in range(target.shape[-1]):
        mask = valid[..., group_index]
        if float(mask.sum().detach().cpu()) <= 0.0:
            continue
        reward = smooth_six_reward(
            target[..., group_index],
            prediction[..., group_index],
            float(mean_targets[group_index].detach().cpu()),
            temperature,
        )
        group_rewards.append((reward * mask).sum() / mask.sum())
    if not group_rewards:
        raise ValueError("Training batch has no scored target")
    return -torch.stack(group_rewards).mean()
