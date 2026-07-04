#!/usr/bin/env python
"""Launch the omni streaming test console (browser UI + WebSocket API).

Examples:
  # offline wiring demo: random-init tiny model + FakeCodec
  python scripts/serve.py --preset tiny model.n_codebooks=2 model.text_vocab_size=320

  # a trained checkpoint with the real codec
  python scripts/serve.py --ckpt checkpoints/quality-export --codec mimi \\
      --tokenizer data/tokenizer/omni_bpe.json --port 7860

Then open http://127.0.0.1:7860 — pick a task, talk or type, and watch the
frame-budget meter while audio streams back. Local test tool: no auth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--ckpt", default=None, metavar="DIR",
                        help="exported model dir (default: random-init from --preset)")
    parser.add_argument("--preset", "--config", dest="config", default="tiny",
                        metavar="NAME|YAML", help="config preset or YAML (default: tiny)")
    parser.add_argument("--codec", default="fake", choices=("fake", "mimi"),
                        help="'mimi' downloads weights on first use (default: fake)")
    parser.add_argument("--tokenizer", default="byte", metavar="byte|PATH|hf:ID")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"),
                        help="inference device (default: auto = cuda if available else cpu); "
                             "the latency meter is only meaningful on the training-target GPU")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("the test console needs the 'serve' extra: pip install 'omni[serve]'",
              file=sys.stderr)
        return 1

    from omni.audio.codec import FakeCodec, build_codec
    from omni.config import load_config
    from omni.model import build_model, load_model
    from omni.serve.app import create_app
    from omni.text.tokenizer import build_tokenizer

    cfg = load_config(args.config, list(args.overrides))
    if args.ckpt is not None:
        ckpt = Path(args.ckpt)
        if not (ckpt / "config.yaml").exists():
            parser.error(f"--ckpt {args.ckpt} needs config.yaml + weights")
        model = load_model(ckpt)
        cfg.model = model.cfg
        print(f"loaded checkpoint {args.ckpt}", file=sys.stderr)
    else:
        model = build_model(cfg.model)
        cfg.model = model.cfg  # adopt backbone-derived fields
        print(f"no --ckpt: RANDOM-INIT '{cfg.preset}' model (wiring demo; expect noise)",
              file=sys.stderr)
    model.eval()

    if args.codec == "fake":
        codec = FakeCodec(n_codebooks=cfg.model.n_codebooks,
                          codec_vocab=cfg.model.audio_codec_vocab)
    else:
        codec = build_codec("mimi", model_id=cfg.codec_model_id,
                            n_codebooks=cfg.model.n_codebooks)
    tokenizer = build_tokenizer(args.tokenizer)

    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps requested but MPS is not available")

    app = create_app(
        cfg=cfg, model=model, codec=codec, tokenizer=tokenizer, ckpt=args.ckpt,
        device=device,
    )
    print(f"device: {device}", file=sys.stderr)
    print(f"omni test console -> http://{args.host}:{args.port}", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
