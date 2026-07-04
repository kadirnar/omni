"""Training loop: LR schedules and the Trainer (fit / checkpoint / export).

The Trainer consumes an already-built model and dataloaders. Batches are the
collated DELAYED grids from ``omni.data.dataset.collate``:

    {"grid": long [B, S, T'], "loss_mask": bool [B, S, T'], "channel": long [B, T']}

Checkpoint layout under ``cfg.train.ckpt_dir``:

    latest.txt                     # name of the newest step dir (rank0)
    step_00000123/
        trainer_state.pt           # world_size == 1: model/optim/sched/step/rng/cfg
        <DCP shard files>          # world_size > 1: {"model", "optim"} via dcp.save
        extra.pt                   # world_size > 1, rank0: step/epoch/sched/rng/cfg
"""

from __future__ import annotations

import dataclasses
import math
import random
import shutil
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from ..config import OmniConfig, TrainConfig
from ..model import structural_diff
from ..model.hf_omni import HFOmniModel
from ..model.omni import OmniModel
from .distributed import (
    DistContext,
    autocast_ctx,
    grad_sync_ctx,
    pick_strategy,
    setup_distributed,
    wrap_model,
)


def _canonical_key(key: str) -> str:
    """Strip torch.compile's OptimizedModule wrapper infix from one FQN."""
    key = key.replace("._orig_mod.", ".")
    if key.startswith("_orig_mod."):
        key = key[len("_orig_mod.") :]
    return key


def _canonical_state_dict(sd: dict) -> dict:
    """State dict re-keyed to canonical (uncompiled) FQNs.

    Regional per-Block compile (``omni.optim.perf.apply_compile``) wraps each
    block in an ``OptimizedModule``, turning ``blocks.N.attn...`` into
    ``blocks.N._orig_mod.attn...``. Checkpoints and exports always store the
    canonical names so they load regardless of ``train.compile`` (the DCP path
    handles this itself via ``get_state_dict``/``set_state_dict``).
    """
    return {_canonical_key(k): v for k, v in sd.items()}


def build_lr_lambda(tc: TrainConfig) -> Callable[[int], float]:
    """LR multiplier per optimizer step (the LambdaLR factor on tc.lr).

    All schedules share a linear warmup 0 -> 1 over ``tc.warmup_steps``, then:

    - "cosine":   half-cosine from 1 down to ``tc.min_lr_ratio`` at
      ``tc.max_steps`` (clamped at the floor after).
    - "wsd":      constant 1.0 until ``0.8 * max_steps``, then linear down to
      ``min_lr_ratio`` at ``max_steps`` (clamped after).
    - "constant": 1.0 forever.
    """
    if tc.schedule not in ("cosine", "wsd", "constant"):
        raise ValueError(f"unknown train.schedule {tc.schedule!r}; expected cosine|wsd|constant")
    warmup = max(0, tc.warmup_steps)
    max_steps = max(1, tc.max_steps)
    floor = tc.min_lr_ratio
    schedule = tc.schedule

    def lr_lambda(step: int) -> float:
        if step < warmup:
            # (step + 1) / (warmup + 1), not step / warmup: LambdaLR applies
            # the factor for epoch 0 to the FIRST optimizer.step(), which
            # would otherwise run at lr=0 and discard the first global batch.
            return (step + 1) / (warmup + 1)
        if schedule == "constant":
            return 1.0
        if schedule == "cosine":
            span = max(1, max_steps - warmup)
            progress = min(1.0, (step - warmup) / span)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
        # wsd: stable at 1.0, then linear decay over the last 20% of steps.
        # Never start decaying before warmup ends (pathological configs with
        # warmup_steps >= 0.8*max_steps would otherwise jump discontinuously).
        decay_start = max(warmup, int(0.8 * max_steps))
        if step < decay_start:
            return 1.0
        span = max(1, max_steps - decay_start)
        progress = min(1.0, (step - decay_start) / span)
        return 1.0 + (floor - 1.0) * progress

    return lr_lambda


class Trainer:
    """Drives training: accumulation, clipping, LR schedule, logging, eval,
    checkpointing, resume, and consolidated export.

    Wraps ``model`` with the resolved strategy at construction so the
    optimizer sees the final (possibly DTensor) parameters. ``self.model`` is
    always the underlying OmniModel; ``self.wrapped`` is what forward passes
    go through (identical object for "none"/"fsdp2", a DDP wrapper for "ddp").
    """

    def __init__(
        self,
        cfg: OmniConfig,
        model: OmniModel,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        ctx: DistContext | None = None,
    ):
        self.cfg = cfg
        self.ctx = ctx if ctx is not None else setup_distributed()
        self.train_loader = train_loader
        self.val_loader = val_loader
        tc = cfg.train
        self.strategy = pick_strategy(
            cfg, self.ctx, sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
        self.model = model
        self.wrapped = wrap_model(model, cfg, self.ctx)  # before optimizer creation
        # Per-rank seed for anything stochastic from here on. Ranks must have
        # built identical weights BEFORE this (see scripts/train.py).
        torch.manual_seed(tc.seed + self.ctx.rank)
        # Both model classes expose optim_param_groups: lr tiers (new modules
        # vs unfrozen backbone) and weight-decay exclusion of norm gains and
        # embedding tables; frozen parameters are excluded entirely.
        params = self.model.optim_param_groups(tc)
        self.optimizer = torch.optim.AdamW(
            params,
            lr=tc.lr,
            betas=tuple(tc.betas),
            weight_decay=tc.weight_decay,
            fused=self.ctx.device.type == "cuda",  # fused kernel is CUDA-only
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, build_lr_lambda(tc))
        self.step = 0
        self._epoch = 0
        self._last_saved = -1
        self._loss_weights = (tc.text_loss_weight, tc.audio_loss_weight, tc.semantic_loss_weight)
        self._wandb = None

    # ------------------------------------------------------------------ data
    def _cycle(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield batches forever, bumping the epoch (and calling set_epoch on
        the loader's batch_sampler when it has one) at each loader restart."""
        while True:
            sampler = getattr(self.train_loader, "batch_sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self._epoch)
            got_any = False
            for batch in self.train_loader:
                got_any = True
                yield batch
            if not got_any:
                raise RuntimeError(
                    "train dataloader yielded no batches "
                    "(dataset smaller than batch_size * world_size with drop_last?)"
                )
            self._epoch += 1

    def _to_device(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """batch -> (grid [B, S, T'], loss_mask [B, S, T'], channel [B, T'])."""
        dev = self.ctx.device
        return (
            batch["grid"].to(dev, non_blocking=True),
            batch["loss_mask"].to(dev, non_blocking=True),
            batch["channel"].to(dev, non_blocking=True),
        )

    # ------------------------------------------------------------------- fit
    def fit(self) -> dict[str, float]:
        """Train until cfg.train.max_steps; returns the final metrics dict."""
        tc = self.cfg.train
        self.maybe_resume()
        if tc.wandb and self.ctx.is_main:
            self._wandb = self._init_wandb()
        if self.ctx.is_main:
            print(
                f"[train] strategy={self.strategy} device={self.ctx.device} "
                f"world_size={self.ctx.world_size} start_step={self.step} "
                f"max_steps={tc.max_steps}",
                flush=True,
            )
        self.wrapped.train()
        data = self._cycle()
        last_metrics: dict[str, float] = {}
        last_val: float | None = None
        window_t0 = time.perf_counter()
        window_steps = 0
        window_frames = 0

        while self.step < tc.max_steps:
            sums: dict[str, float] = {}
            counts: dict[str, int] = {}
            for micro in range(tc.accum_steps):
                grid, loss_mask, channel = self._to_device(next(data))
                is_last = micro == tc.accum_steps - 1
                with grad_sync_ctx(self.wrapped, self.strategy, is_last):
                    with autocast_ctx(self.cfg, self.ctx, self.strategy):
                        out = self.wrapped(grid, channel)
                        # multistream_loss carries its own exact-zero anchor, so
                        # every head gets a (zero) grad even when a batch has no
                        # targets for it — DDP requires that each backward.
                        total, metrics = self.model.loss(
                            out, grid, loss_mask, self._loss_weights
                        )
                    (total / tc.accum_steps).backward()
                for k, v in metrics.items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
                window_frames += grid.shape[0] * grid.shape[2]

            max_norm = tc.grad_clip if tc.grad_clip > 0 else float("inf")
            gn = torch.nn.utils.clip_grad_norm_(self.wrapped.parameters(), max_norm)
            grad_norm = float(gn.full_tensor()) if hasattr(gn, "full_tensor") else float(gn)
            lr_used = float(self.optimizer.param_groups[0]["lr"])
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.step += 1
            window_steps += 1

            means = {k: sums[k] / counts[k] for k in sums}
            last_metrics = {**means, "lr": lr_used, "grad_norm": grad_norm}

            # log_every <= 0 disables logging, mirroring eval_every/save_every.
            if tc.log_every > 0 and (self.step % tc.log_every == 0 or self.step == tc.max_steps):
                dt = max(time.perf_counter() - window_t0, 1e-9)
                steps_per_s = window_steps / dt
                frames_per_s = window_frames * self.ctx.world_size / dt
                if self.ctx.is_main:
                    heads = " ".join(
                        f"{k[5:]}={means[k]:.3f}" for k in sorted(means) if k.startswith("loss/")
                    )
                    print(
                        f"[train] step {self.step}/{tc.max_steps} "
                        f"loss={means['loss']:.4f} lr={lr_used:.3e} "
                        f"grad_norm={grad_norm:.3f} steps/s={steps_per_s:.2f} "
                        f"frames/s={frames_per_s:.0f}" + (f" | {heads}" if heads else ""),
                        flush=True,
                    )
                    if self._wandb is not None:
                        self._wandb.log(
                            {**means, "lr": lr_used, "grad_norm": grad_norm,
                             "steps_per_s": steps_per_s, "frames_per_s": frames_per_s,
                             "epoch": self._epoch},
                            step=self.step,
                        )
                window_t0 = time.perf_counter()
                window_steps = 0
                window_frames = 0

            if self.val_loader is not None and tc.eval_every > 0 and self.step % tc.eval_every == 0:
                val_loss = self._evaluate()
                last_val = val_loss
                if self.ctx.is_main:
                    print(f"[eval]  step {self.step} val_loss={val_loss:.4f}", flush=True)
                    if self._wandb is not None:
                        self._wandb.log({"val/loss": val_loss}, step=self.step)

            if tc.save_every > 0 and self.step % tc.save_every == 0:
                self.save_checkpoint(self.step)

        if self.step > 0 and self._last_saved != self.step:
            self.save_checkpoint(self.step)
        if self._wandb is not None:
            self._wandb.finish()
        if last_val is not None:
            last_metrics["val/loss"] = last_val
        last_metrics["step"] = float(self.step)
        return last_metrics

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Mean val loss (this rank's shard) over up to cfg.train.eval_steps batches."""
        self.wrapped.eval()
        losses: list[float] = []
        it = iter(self.val_loader)
        for _ in range(self.cfg.train.eval_steps):
            try:
                batch = next(it)
            except StopIteration:
                break
            grid, loss_mask, channel = self._to_device(batch)
            # same numerics as training (bf16 autocast where it applies) so
            # val loss is comparable and eval is not ~2x slower in fp32
            with autocast_ctx(self.cfg, self.ctx, self.strategy):
                out = self.wrapped(grid, channel)
                total, _ = self.model.loss(out, grid, loss_mask, self._loss_weights)
            losses.append(float(total))
        self.wrapped.train()
        return sum(losses) / max(1, len(losses))

    def _init_wandb(self):
        """Start (or resume) the rank0 W&B run.

        The run id is persisted at ``<ckpt_dir>/wandb_run_id.txt`` next to the
        checkpoints, so resuming training resumes the SAME wandb run with a
        continuous step history instead of starting a fresh one.
        """
        try:
            import wandb  # optional dep, imported only when cfg.train.wandb
        except ImportError as e:
            raise ImportError(
                "cfg.train.wandb=true but the 'wandb' package is not installed; "
                "pip install 'omni[wandb]' or set train.wandb=false"
            ) from e
        tc = self.cfg.train
        ckpt_dir = Path(tc.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        id_file = ckpt_dir / "wandb_run_id.txt"
        run_id = id_file.read_text().strip() if id_file.exists() else None
        run = wandb.init(
            project=tc.wandb_project,
            entity=tc.wandb_entity,
            name=tc.wandb_run_name,
            tags=list(tc.wandb_tags) or None,
            mode=tc.wandb_mode,
            dir=str(ckpt_dir),
            id=run_id or None,
            resume="allow" if run_id else None,
            config=dataclasses.asdict(self.cfg),
        )
        new_id = getattr(run, "id", None)
        if run_id is None and new_id:
            id_file.write_text(str(new_id))
        return wandb

    # ---------------------------------------------------------- checkpointing
    def _frozen_canonical_keys(self) -> set[str]:
        """Canonical FQNs of frozen parameters — never worth checkpointing
        (a frozen v6 backbone is reconstructed from the hub at build time;
        writing its ~8-16 GB every save_every would dominate checkpoint
        size). Empty for fully-trainable models (full save, as before)."""
        return {
            _canonical_key(n)
            for n, p in self.model.named_parameters()
            if not p.requires_grad
        }

    def _extra_state(self, step: int) -> dict:
        rng: dict[str, object] = {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {
            "step": step,
            "epoch": self._epoch,
            "sched": self.scheduler.state_dict(),
            "rng": rng,
            "cfg": dataclasses.asdict(self.cfg),
        }

    def save_checkpoint(self, step: int) -> None:
        """Write ``<ckpt_dir>/step_%08d/`` and update ``<ckpt_dir>/latest.txt``.

        world_size > 1: collective DCP save of {"model", "optim"} from
        ``get_state_dict(wrapped, optimizer)`` plus a rank0-only ``extra.pt``
        (step/epoch/sched/rank0-rng/cfg). world_size == 1: a single
        ``trainer_state.pt`` with the same content plus model/optim.
        """
        tc = self.cfg.train
        name = f"step_{step:08d}"
        step_dir = Path(tc.ckpt_dir) / name
        step_dir.mkdir(parents=True, exist_ok=True)
        frozen = self._frozen_canonical_keys()
        if self.ctx.world_size > 1:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict

            msd, osd = get_state_dict(self.wrapped, self.optimizer)
            if frozen:
                msd = {k: v for k, v in msd.items() if _canonical_key(k) not in frozen}
            dcp.save({"model": msd, "optim": osd}, checkpoint_id=str(step_dir))
            if self.ctx.is_main:
                torch.save(self._extra_state(step), step_dir / "extra.pt")
        else:
            model_sd = _canonical_state_dict(self.model.state_dict())
            if frozen:
                model_sd = {k: v for k, v in model_sd.items() if k not in frozen}
            state = {
                "model": model_sd,
                "optim": self.optimizer.state_dict(),
                **self._extra_state(step),
            }
            torch.save(state, step_dir / "trainer_state.pt")
        if self.ctx.is_main:
            (Path(tc.ckpt_dir) / "latest.txt").write_text(name + "\n")
            print(f"[ckpt]  saved {step_dir}", flush=True)
        if self.ctx.initialized:
            dist.barrier()  # every rank done writing before any pruning
        if self.ctx.is_main and tc.save_keep > 0:
            kept = sorted(d for d in Path(tc.ckpt_dir).glob("step_*") if d.is_dir())
            for old in kept[: -tc.save_keep]:
                shutil.rmtree(old, ignore_errors=True)
        self._last_saved = step

    def maybe_resume(self) -> int:
        """Restore model/optimizer/scheduler/step from the latest checkpoint.

        Active iff cfg.train.resume and ``<ckpt_dir>/latest.txt`` exists.
        Returns the restored step (0 when nothing was restored). The data
        position is restored only at epoch granularity; RNG state (saved on
        rank0) is restored on rank0 only.
        """
        tc = self.cfg.train
        if not tc.resume:
            return 0
        latest = Path(tc.ckpt_dir) / "latest.txt"
        if not latest.exists():
            return 0
        name = latest.read_text().strip()
        step_dir = Path(tc.ckpt_dir) / name
        # Read the light state FIRST and refuse structurally-incompatible
        # checkpoints with a clear message (a shared default ckpt_dir plus a
        # config change used to die in load_state_dict with a raw shape error).
        extra_file = (
            step_dir / ("extra.pt" if self.ctx.world_size > 1 else "trainer_state.pt")
        )
        extra = torch.load(extra_file, map_location="cpu", weights_only=False)
        saved_model_cfg = (extra.get("cfg") or {}).get("model") or {}
        diffs = structural_diff(saved_model_cfg, self.cfg.model)
        for k in ("freeze_backbone", "lora_rank"):  # optimizer-group structure
            cur = getattr(self.cfg.model, k)
            if k in saved_model_cfg and saved_model_cfg[k] != cur:
                diffs.append(f"{k}: saved={saved_model_cfg[k]!r} vs current={cur!r}")
        if diffs:
            raise ValueError(
                f"cannot resume from {step_dir}: the checkpoint was trained with a "
                f"different config ({'; '.join(diffs)}). Point train.ckpt_dir at a "
                "fresh directory, delete the stale checkpoints, or warm-start a new "
                "stage with scripts/train.py --init-from instead of resuming."
            )
        frozen = self._frozen_canonical_keys()
        if self.ctx.world_size > 1:
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_state_dict,
                set_state_dict,
            )

            # get_state_dict materializes optimizer state on a fresh optimizer,
            # so dcp.load has a matching structure to load into. Frozen keys
            # were never saved (lean checkpoints): request only what exists.
            msd, osd = get_state_dict(self.wrapped, self.optimizer)
            if frozen:
                msd = {k: v for k, v in msd.items() if _canonical_key(k) not in frozen}
            dcp.load({"model": msd, "optim": osd}, checkpoint_id=str(step_dir))
            set_state_dict(
                self.wrapped,
                self.optimizer,
                model_state_dict=msd,
                optim_state_dict=osd,
                options=StateDictOptions(strict=not frozen),
            )
        else:
            # Checkpoints store canonical FQNs; the live model's keys carry an
            # `_orig_mod.` infix when train.compile wrapped its blocks. Remap
            # so a checkpoint loads regardless of either run's compile setting
            # (also tolerates legacy checkpoints saved with compiled keys).
            remap = {_canonical_key(k): k for k in self.model.state_dict()}
            saved = _canonical_state_dict(extra["model"])
            result = self.model.load_state_dict(
                {remap.get(k, k): v for k, v in saved.items()}, strict=False
            )
            if result.unexpected_keys:
                raise ValueError(
                    f"checkpoint {step_dir} has unexpected keys: "
                    f"{result.unexpected_keys[:5]}"
                )
            live_frozen = {
                n for n, p in self.model.named_parameters() if not p.requires_grad
            }
            bad = [k for k in result.missing_keys if k not in live_frozen]
            if bad:
                raise ValueError(
                    f"checkpoint {step_dir} is missing trainable keys: {bad[:5]}"
                )
            self.optimizer.load_state_dict(extra["optim"])
        self.scheduler.load_state_dict(extra["sched"])
        self.step = int(extra["step"])
        self._epoch = int(extra.get("epoch", 0))
        self._last_saved = self.step
        rng = extra.get("rng")
        if rng is not None and self.ctx.is_main:
            torch.set_rng_state(rng["torch"])
            random.setstate(rng["python"])
            if "cuda" in rng and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
        if self.ctx.is_main:
            print(f"[ckpt]  resumed {step_dir} at step {self.step}", flush=True)
        return self.step

    # ----------------------------------------------------------------- export
    def export_model(self, out_dir: str | Path) -> None:
        """Write consolidated fp32 weights: ``<out_dir>/model.safetensors`` +
        ``config.yaml`` (flat ModelConfig), loadable via
        ``OmniModel.from_pretrained``. Collective under FSDP2 (every rank must
        call); only rank0 writes files.
        """
        if self.strategy == "fsdp2":
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            sd = get_model_state_dict(
                self.wrapped,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        else:
            # DDP shares storage with the raw module; canonicalize keys in case
            # train.compile wrapped the blocks in OptimizedModule.
            sd = _canonical_state_dict(self.model.state_dict())
        if self.ctx.is_main:
            if isinstance(self.model, HFOmniModel):  # v6 backbone path
                # Adapter-only export: the backbone is referenced by
                # backbone_id in config.yaml, so no fresh skeleton is needed
                # (and none could be built without re-downloading it).
                self.model.save_pretrained(out_dir, state_dict=sd)
            else:
                export = OmniModel(self.cfg.model)  # fresh CPU fp32 skeleton
                export.load_state_dict(sd)
                export.save_pretrained(out_dir)
            print(f"[export] wrote {Path(out_dir)}", flush=True)
        if self.ctx.initialized:
            dist.barrier()
