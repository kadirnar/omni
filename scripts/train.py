#!/usr/bin/env python
"""Train the omni model on prepared shard directories.

Single process (Mac CPU smoke test):
  python scripts/train.py --preset tiny --data data/shards/fake train.max_steps=50

Multi-GPU (or CPU/gloo) via torchrun:
  torchrun --standalone --nproc_per_node=8 scripts/train.py --preset base \
      --data data/shards/stageB:0.7 --data data/shards/textlm:0.3 [key=value ...]

--data DIR[:WEIGHT] is repeatable. Without weights every sample of every dir
is used once (natural mixing); with weights on ALL dirs the weights are
sampling PROPORTIONS of the epoch, honored by resampling (stage-recipe task
ratios / replay fractions). Mixing weighted and unweighted dirs is an error.
Positional key=value overrides are applied by omni.config.load_config;
--export DIR writes consolidated safetensors weights at the end. Exits
nonzero on failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys

import torch
import yaml

from omni.config import PRESETS, load_config
from omni.model import build_model
from omni.train.distributed import cleanup, setup_distributed
from omni.train.loop import Trainer


def _parse_data_arg(spec: str) -> tuple[str, float | None]:
    """"DIR[:WEIGHT]" -> (dir, weight | None). The suffix after the last ':'
    counts as a weight only when it parses as a float; otherwise it is path
    text. None means "no explicit weight" (natural mixing)."""
    head, sep, tail = spec.rpartition(":")
    if sep:
        try:
            return head, float(tail)
        except ValueError:
            pass
    return spec, None


def _shard_dirs(specs: list[str]) -> dict[str, float] | list[str]:
    """CLI --data specs -> build_dataloader input (all-weighted dict or plain
    list). Mixing weighted and unweighted dirs is ambiguous -> error."""
    pairs = [_parse_data_arg(s) for s in specs]
    weighted = [p for p in pairs if p[1] is not None]
    if not weighted:
        return [d for d, _w in pairs]
    if len(weighted) != len(pairs):
        missing = [d for d, w in pairs if w is None]
        raise SystemExit(
            f"--data mixes weighted and unweighted dirs (no weight on: {missing}); "
            "give every dir a weight (proportions) or none (natural mixing)"
        )
    return {d: float(w) for d, w in pairs}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--preset", default="tiny", choices=sorted(PRESETS),
                     help="config preset (default: tiny)")
    src.add_argument("--config", default=None, metavar="YAML",
                     help="config YAML file (may set 'preset:' inside)")
    p.add_argument("--data", action="append", required=True, metavar="DIR[:WEIGHT]",
                   help="training shard dir with optional mixture weight; repeatable")
    p.add_argument("--val-data", default=None, metavar="DIR",
                   help="optional validation shard dir")
    p.add_argument("--init-from", default=None, metavar="DIR",
                   help="warm-start weights from an exported checkpoint dir "
                        "(fresh optimizer/schedule — the stage-transition path, "
                        "e.g. stage 1 frozen -> stage 2 unfrozen/LoRA; unlike "
                        "train.resume, the freeze/LoRA config may differ)")
    p.add_argument("--export", default=None, metavar="DIR",
                   help="export consolidated weights here after training")
    p.add_argument("overrides", nargs="*", metavar="key=value",
                   help="dotted config overrides, e.g. train.max_steps=50")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = load_config(args.config if args.config else args.preset, args.overrides)
    ctx = setup_distributed()
    try:
        # Every rank must build identical weights: DDP broadcasts at wrap time
        # but FSDP2's fully_shard does not. (Pretrained-backbone models load
        # identical hub weights; their new modules follow this seed.)
        torch.manual_seed(cfg.train.seed)
        model = build_model(cfg.model)  # init_weights runs in __init__
        cfg.model = model.cfg  # adopt backbone-derived fields (d_model, vocab, ...)
        if args.init_from:
            from omni.model import load_weights

            load_weights(model, args.init_from)  # every rank, before wrap/compile
            if ctx.is_main:
                print(f"[init]  warm-started weights from {args.init_from}", flush=True)
        if ctx.is_main:
            counts = model.param_counts()
            print(
                f"params: total={counts['total']:,} "
                f"non_embedding={counts['non_embedding']:,}"
                + (f" trainable={counts['trainable']:,}" if "trainable" in counts else "")
            )
            print("config:")
            print(yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False), flush=True)
        if cfg.train.compile:
            # Regional per-Block compile must happen BEFORE any fully_shard
            # (torch >= 2.9 cannot trace through FSDP2 hooks).
            from omni.optim.perf import apply_compile

            model = apply_compile(model)

        from omni.data.dataset import build_dataloader  # peer module (agent B)

        shard_dirs = _shard_dirs(args.data)
        train_loader = build_dataloader(
            cfg, shard_dirs, rank=ctx.rank, world_size=ctx.world_size, epoch=0, shuffle=True
        )
        val_loader = None
        if args.val_data:
            val_loader = build_dataloader(
                cfg, [args.val_data], rank=ctx.rank, world_size=ctx.world_size,
                epoch=0, shuffle=False,
            )

        trainer = Trainer(cfg, model, train_loader, val_loader, ctx=ctx)
        metrics = trainer.fit()
        if ctx.is_main:
            print("final: " + " ".join(f"{k}={v:.4g}" for k, v in sorted(metrics.items())))
        if args.export:
            trainer.export_model(args.export)
    finally:
        cleanup(ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
