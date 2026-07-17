from __future__ import annotations

import numpy as np
import torch


def hard_score_reward_matrix(
    target: torch.Tensor,
    candidates: torch.Tensor,
    mean_target: float,
) -> torch.Tensor:
    if mean_target <= 0.0:
        raise ValueError("mean_target must be positive")
    error = torch.abs(target.unsqueeze(-1) - candidates)
    unit_price = torch.where(
        error <= 0.06,
        torch.full_like(error, 4.0),
        torch.where(error <= 0.08, torch.full_like(error, 3.0), 0.0),
    )
    return -error + (target.unsqueeze(-1) / float(mean_target)) * (
        unit_price / 4.0
    )


def hard_score_row_reward(
    target: np.ndarray,
    prediction: np.ndarray,
    mean_target: float,
) -> np.ndarray:
    if mean_target <= 0.0:
        raise ValueError("mean_target must be positive")
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    error = np.abs(prediction - target)
    unit_price = np.select(
        [error <= 0.06, error <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    return -error + (target / float(mean_target)) * (unit_price / 4.0)


def smooth_ficr_reward(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mean_target: float,
    temperature: float,
) -> torch.Tensor:
    """Differentiable FiCR reward with no point-error term."""
    if mean_target <= 0.0:
        raise ValueError("mean_target must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    error = torch.abs(prediction - target)
    within_eight = torch.sigmoid((0.08 - error) / float(temperature))
    within_six = torch.sigmoid((0.06 - error) / float(temperature))
    return (target / float(mean_target)) * (
        0.75 * within_eight + 0.25 * within_six
    )


def smooth_six_reward(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mean_target: float,
    temperature: float,
) -> torch.Tensor:
    """Differentiable generation-weighted reward for the <=6% band only."""
    if mean_target <= 0.0:
        raise ValueError("mean_target must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    error = torch.abs(prediction - target)
    within_six = torch.sigmoid((0.06 - error) / float(temperature))
    return (target / float(mean_target)) * within_six


def smooth_four_reward(
    target: torch.Tensor,
    prediction: torch.Tensor,
    mean_target: float,
    temperature: float,
) -> torch.Tensor:
    """Differentiable generation-weighted reward for the <=4% margin."""
    if mean_target <= 0.0:
        raise ValueError("mean_target must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    error = torch.abs(prediction - target)
    within_four = torch.sigmoid((0.04 - error) / float(temperature))
    return (target / float(mean_target)) * within_four
