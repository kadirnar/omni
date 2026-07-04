#!/usr/bin/env python
"""Benchmark omni decode (steps/s, rtf) and forward+backward (tokens/s).

Examples:
  python scripts/benchmark.py --preset tiny --device cpu --decode-frames 50
  python scripts/benchmark.py --ckpt checkpoints/run/export --device cuda --compile --int8
  python scripts/benchmark.py --preset quality --device cuda --voice-frames 125
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from omni.config import load_config
from omni.model import build_model, load_model
from omni.optim.perf import apply_compile, benchmark_decode, benchmark_forward, quantize_int8


def _resolve_device(name: str, parser: argparse.ArgumentParser) -> torch.device:
    """auto -> cuda if available else cpu (mps only when asked explicitly)."""
    if name == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps requested but MPS is not available")
    return torch.device(name)


def _print_table(title: str, stats: dict) -> None:
    print(f"\n{title}")
    width = max(len(k) for k in stats)
    for k, v in stats.items():
        print(f"  {k:<{width}} : {v:,.3f}" if isinstance(v, float) else f"  {k:<{width}} : {v}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--preset", default="tiny", help="config preset (default: tiny)")
    parser.add_argument("--config", default=None, metavar="YAML", help="config YAML (overrides --preset)")
    parser.add_argument("--ckpt", default=None, metavar="DIR", help="exported model dir (default: random init)")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--compile", action="store_true", help="regional torch.compile of each Block")
    parser.add_argument("--compile-mode", default="default", help="torch.compile mode (default: default)")
    parser.add_argument("--int8", action="store_true", help="torchao int8 weight-only (skips fwd+bwd bench)")
    parser.add_argument("--decode-frames", type=int, default=100, help="timed decode steps (default: 100)")
    parser.add_argument("--decode-batch", type=int, default=1, help="decode batch size (default: 1)")
    parser.add_argument(
        "--voice-frames", type=int, default=0, metavar="N",
        help="prefill a random N-frame voice-reference prefix before decoding and "
             "report voice_prefill_ms (DESIGN_V5; try 127 grid cols via N=125; default: 0)",
    )
    parser.add_argument("--batch", type=int, default=2, help="forward+backward batch size (default: 2)")
    parser.add_argument("--frames", type=int, default=128, help="forward+backward sequence frames (default: 128)")
    parser.add_argument("--steps", type=int, default=10, help="timed forward+backward iterations (default: 10)")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="dotted config overrides, e.g. model.n_codebooks=2 (ignored with --ckpt)",
    )
    args = parser.parse_args(argv)
    for ov in args.overrides:
        if "=" not in ov:
            parser.error(f"override must look like a.b=c, got: {ov!r}")
    if args.voice_frames < 0:
        parser.error(f"--voice-frames must be >= 0, got {args.voice_frames}")

    device = _resolve_device(args.device, parser)
    try:
        cfg = load_config(args.config if args.config else args.preset, list(args.overrides))
    except (KeyError, ValueError, AssertionError) as e:
        parser.error(f"bad config/overrides: {e}")
    if args.ckpt is not None:
        ckpt = Path(args.ckpt)
        weights_ok = (ckpt / "model.safetensors").exists() or (
            ckpt / "adapters.safetensors"
        ).exists()
        if not (ckpt / "config.yaml").exists() or not weights_ok:
            parser.error(f"--ckpt {args.ckpt} needs config.yaml + weights")
        model = load_model(ckpt)
        cfg.model = model.cfg
        source = args.ckpt
    else:
        model = build_model(cfg.model)
        cfg.model = model.cfg
        source = f"random init ({cfg.preset})"
    model = model.to(device).eval()

    counts = model.param_counts()
    print(
        f"omni benchmark | model: {source} | params {counts['total']:,} "
        f"(non-emb {counts['non_embedding']:,}) | device {device.type} "
        f"| compile={args.compile} int8={args.int8}"
    )

    if args.int8:
        quantize_int8(model)  # before compile, per the gpt-fast recipe
    if args.compile:
        model = apply_compile(model, mode=args.compile_mode)

    dec = benchmark_decode(
        model, cfg, device, n_frames=args.decode_frames, batch=args.decode_batch,
        voice_frames=args.voice_frames,
    )
    title = f"decode loop (batch={args.decode_batch}"
    if args.voice_frames:
        title += f", voice prefix {args.voice_frames} frames"
    _print_table(title + ")", dec)
    print("  real-time budget: rtf > 1 means faster than 12.5 frames/s")
    if args.voice_frames:
        print(
            "  voice gates (DESIGN_V5): voice_prefill_ms < 40 at quality; "
            "compare ms_per_step vs a --voice-frames 0 run (delta < 1 ms)"
        )

    if args.int8:
        print("\nskipping forward+backward benchmark: int8 weights are inference-only")
    else:
        fwd = benchmark_forward(
            model, cfg, device, batch=args.batch, frames=args.frames, steps=args.steps
        )
        _print_table(f"forward+backward (batch={args.batch}, frames={args.frames})", fwd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
