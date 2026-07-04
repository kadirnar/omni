"""Training: distributed setup/strategies and the Trainer loop."""

from .distributed import (
    DistContext,
    autocast_ctx,
    cleanup,
    grad_sync_ctx,
    is_distributed_env,
    pick_strategy,
    setup_distributed,
    wrap_model,
)
from .loop import Trainer, build_lr_lambda

__all__ = [
    "DistContext",
    "Trainer",
    "autocast_ctx",
    "build_lr_lambda",
    "cleanup",
    "grad_sync_ctx",
    "is_distributed_env",
    "pick_strategy",
    "setup_distributed",
    "wrap_model",
]
