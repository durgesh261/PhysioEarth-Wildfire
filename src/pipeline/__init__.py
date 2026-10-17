"""Pipeline package containing model training and rollout evaluation."""

from .trainer import (
    ModelRolloutOutput,
    train_single_model,
    evaluate_autoregressive_rollout,
    prepare_input_tensor,
)

__all__ = [
    "ModelRolloutOutput",
    "train_single_model",
    "evaluate_autoregressive_rollout",
    "prepare_input_tensor",
]
