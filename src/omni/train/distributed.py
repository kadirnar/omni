"""Distributed setup and parallelism strategies for training.

Three strategies (docs/DESIGN.md §5, docs/research/training-systems.md):

- ``"none"``:  single process; the model just moves to the device.
- ``"ddp"``:   DistributedDataParallel; works on CPU (gloo) and CUDA (nccl).
- ``"fsdp2"``: per-Block ``fully_shard`` with ``MixedPrecisionPolicy``
  (bf16 params, fp32 reduce) then root ``fully_shard`` — CUDA only,
  fp32 master weights (torchtitan pattern).

The same code path runs single-process on Mac CPU/MPS and under
``torchrun --standalone --nproc_per_node=N`` on a CUDA node or on CPU/gloo.
No CUDA-only feature (nccl, set_device, bf16 autocast) is touched unless the
device really is CUDA.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..config import OmniConfig

STRATEGIES = ("none", "ddp", "fsdp2")


@dataclass
class DistContext:
    """Where this process sits in the (possibly single-process) job."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    is_main: bool  # rank == 0
    initialized: bool  # a torch.distributed process group is active


def is_distributed_env() -> bool:
    """True when launched by torchrun (RANK set in the environment)."""
    return "RANK" in os.environ


def _resolve_device(preference: str) -> torch.device:
    """Map a device preference to a concrete device.

    "auto" picks cuda when available, else cpu — MPS is used only when
    explicitly requested (its distributed/inductor support is immature).
    """
    if preference == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but CUDA is not available")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device 'mps' requested but MPS is not available")
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device preference {preference!r}; expected auto|cpu|cuda|mps")


def setup_distributed(device_preference: str = "auto") -> DistContext:
    """Initialize (or attach to) the process group and pick this rank's device.

    torchrun path (RANK in env): ``init_process_group`` with nccl when CUDA is
    used, else gloo; ``cuda.set_device(local_rank)`` only on CUDA. Plain
    ``python`` path: single process, no process group. Safe to call when a
    group is already initialized (attaches instead of re-initializing).
    """
    if is_distributed_env() or dist.is_initialized():
        use_cuda = torch.cuda.is_available() and device_preference in ("auto", "cuda")
        if not dist.is_initialized():
            dist.init_process_group("nccl" if use_cuda else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if use_cuda:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = _resolve_device("cpu" if device_preference == "auto" else device_preference)
        initialized = True
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = _resolve_device(device_preference)
        initialized = False
    return DistContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        is_main=rank == 0,
        initialized=initialized,
    )


def cleanup(ctx: DistContext) -> None:
    """Destroy the process group if this context is attached to one."""
    if ctx.initialized and dist.is_initialized():
        dist.destroy_process_group()


def pick_strategy(cfg: OmniConfig, ctx: DistContext, trainable_params: int) -> str:
    """Resolve cfg.train.strategy to one of "none" | "ddp" | "fsdp2".

    world_size == 1 always resolves to "none". A fixed (non-"auto") strategy
    is honored as-is (wrap_model raises if fsdp2 lacks CUDA). "auto": "fsdp2"
    iff trainable_params >= cfg.train.fsdp_threshold_params AND the device is
    CUDA, else "ddp". TRAINABLE params drive the choice: sharding pays for
    gradient/optimizer memory, so a frozen 8B backbone with small adapters
    trains fine (and simpler) under DDP, while a full finetune crosses the
    threshold and shards.
    """
    if ctx.world_size == 1:
        return "none"
    strategy = cfg.train.strategy
    if strategy != "auto":
        if strategy not in STRATEGIES:
            raise ValueError(
                f"unknown train.strategy {strategy!r}; expected auto|none|ddp|fsdp2"
            )
        return strategy
    if trainable_params >= cfg.train.fsdp_threshold_params and ctx.device.type == "cuda":
        return "fsdp2"
    return "ddp"


def _shardable_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    """The per-layer modules fully_shard wraps: ``model.blocks`` on the
    from-scratch OmniModel, the HF decoder's layer list on HFOmniModel
    (mirrors ``optim.perf.apply_compile``'s dispatch)."""
    blocks = getattr(model, "blocks", None)
    if blocks is not None:
        return list(blocks)
    backbone = getattr(model, "backbone", None)
    if backbone is not None:
        decoder = backbone.get_decoder() if hasattr(backbone, "get_decoder") else backbone
        layers = getattr(decoder, "layers", None)
        if layers is not None:
            return list(layers)
    raise RuntimeError(
        "fsdp2 wrapping found neither model.blocks nor an HF backbone with "
        "decoder .layers — cannot shard this model; use strategy 'ddp'"
    )


def wrap_model(model: torch.nn.Module, cfg: OmniConfig, ctx: DistContext) -> torch.nn.Module:
    """Move the model to the device and apply the resolved strategy.

    Must run BEFORE the optimizer is created (FSDP2 turns parameters into
    DTensors). Gradient checkpointing lives inside the model's forward, so it
    is automatically "applied before" any sharding here.

    - none:  ``model.to(device)``.
    - ddp:   ``DDP(model)`` after ``.to(device)``; no ``device_ids`` are
      passed so the CPU/gloo path works (on CUDA the device was already set
      via ``cuda.set_device``). DDP broadcasts rank0's weights at wrap time.
      Frozen parameters (v6 backbone) are simply never reduced.
    - fsdp2: CUDA only. Per-layer ``fully_shard`` over
      :func:`_shardable_blocks` (OmniModel Blocks or the HF decoder's layers)
      then root ``fully_shard`` (stays unsharded post-forward), with
      MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32) when
      cfg.train.precision == "bf16". Ranks must have constructed identical
      weights (fully_shard does not broadcast).
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    strategy = pick_strategy(cfg, ctx, trainable)
    model.to(ctx.device)
    if strategy == "none":
        return model
    if strategy == "ddp":
        return DDP(model)
    # fsdp2
    if ctx.device.type != "cuda":
        raise RuntimeError("strategy 'fsdp2' requires CUDA; use 'ddp' or 'none' on cpu/mps")
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    if cfg.train.precision == "bf16":
        mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    else:
        mp = MixedPrecisionPolicy()  # keep fp32 params/reduction
    for block in _shardable_blocks(model):
        fully_shard(block, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    return model


@contextmanager
def grad_sync_ctx(
    wrapped: torch.nn.Module, strategy: str, is_last_microbatch: bool
) -> Iterator[None]:
    """Context for one microbatch's forward+backward under grad accumulation.

    - ddp:   ``no_sync()`` on non-final microbatches (skip grad all-reduce).
    - fsdp2: ``set_requires_gradient_sync(is_last_microbatch)`` (skip
      reduce-scatter until the last microbatch; unsharded grads are held).
    - none:  no-op.
    """
    if strategy == "ddp" and not is_last_microbatch:
        with wrapped.no_sync():
            yield
    elif strategy == "fsdp2":
        wrapped.set_requires_gradient_sync(is_last_microbatch)
        yield
    else:
        yield


def autocast_ctx(
    cfg: OmniConfig, ctx: DistContext, strategy: str
) -> AbstractContextManager[object]:
    """bf16 autocast for the forward pass where it applies.

    Only CUDA + strategy in {ddp, none} + precision "bf16". FSDP2 casts via
    its MixedPrecisionPolicy instead, and CPU/MPS stay fp32.
    """
    if (
        ctx.device.type == "cuda"
        and strategy in ("ddp", "none")
        and cfg.train.precision == "bf16"
    ):
        return torch.autocast("cuda", torch.bfloat16)
    return nullcontext()
