"""omni chat CLI: run one s2s / tts / asr generation from the command line.

Examples::

    python scripts/chat.py --task tts --text "hello omni" --out hello.wav \\
        --codec fake --tokenizer byte --preset tiny --max-frames 100
    python scripts/chat.py --task s2s --in user.wav --out reply.wav \\
        --ckpt checkpoints/run/export --codec mimi --tokenizer data/tokenizer/omni_bpe.json
    python scripts/chat.py --task asr --in user.wav --ckpt checkpoints/run/export
    python scripts/chat.py --task tts --text "bonjour" --voice ref.wav --lang fr \\
        --out clone.wav --ckpt checkpoints/run/export --codec mimi

Nothing downloads unless ``--codec mimi`` is passed explicitly. Without
``--ckpt`` the model is RANDOM-INIT (a wiring demo, not speech). The
transcript/text goes to stdout; diagnostics go to stderr.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from ..audio.codec import build_codec, load_wav, save_wav
from ..config import load_config
from ..model import build_model, load_model
from ..text.tokenizer import build_tokenizer
from .generate import DEFAULT_VOICE_FRAMES, OmniGenerator


def _resolve_device(name: str, parser: argparse.ArgumentParser) -> torch.device:
    """auto -> cuda if available else cpu (mps only when asked explicitly)."""
    if name == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        parser.error("--device mps requested but MPS is not available")
    return torch.device(name)


def _info(msg: str) -> None:
    print(msg, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omni-chat", description=__doc__.strip().splitlines()[0]
    )
    parser.add_argument(
        "--ckpt", default=None, metavar="DIR",
        help="model dir from Trainer.export_model / OmniModel.save_pretrained "
             "(default: random-init demo from --preset/--config)",
    )
    parser.add_argument("--preset", default="tiny", help="config preset (default: tiny)")
    parser.add_argument(
        "--config", default=None, metavar="YAML",
        help="config YAML path (overrides --preset)",
    )
    parser.add_argument("--task", required=True, choices=("s2s", "tts", "asr", "duplex"))
    parser.add_argument(
        "--in", dest="in_wav", default=None, metavar="IN.wav",
        help="input speech wav (s2s/asr; any sr, resampled to the codec)",
    )
    parser.add_argument("--text", default=None, help="text to speak (tts)")
    parser.add_argument(
        "--out", default=None, metavar="OUT.wav", help="output wav path (s2s/tts)"
    )
    parser.add_argument(
        "--codec", default="fake", choices=("fake", "mimi"),
        help="audio codec; 'mimi' downloads weights from the HF hub (default: fake)",
    )
    parser.add_argument(
        "--tokenizer", default="byte", metavar="byte|PATH|hf:ID",
        help='"byte", a tokenizer JSON from train_bpe, or "hf:<model_id>" for '
             "a backbone tokenizer (default: byte)",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    parser.add_argument(
        "--max-frames", type=int, default=None, metavar="N",
        help="cap on generated audio frames (default: sampling.max_frames)",
    )
    from ..streams import EMOTION_CLASSES, LANG_TAGS, turn_prefix

    parser.add_argument(
        "--emotion", default=None, choices=sorted(EMOTION_CLASSES),
        help="force the response style (s2s/tts; default: model decides)",
    )
    parser.add_argument(
        "--lang", default=None, choices=sorted(LANG_TAGS),
        help="force the response language tag (default: model decides)",
    )
    parser.add_argument(
        "--voice", default=None, metavar="REF.wav",
        help="reference speaker wav pinning the assistant voice (DESIGN_V5; "
             "tts/s2s/duplex, ignored with a warning for asr); composes with "
             "--emotion/--lang, which keep style/language authority",
    )
    args = parser.parse_args(argv)

    # ---- argument coherence -------------------------------------------------
    if args.task in ("s2s", "asr", "duplex"):
        if args.in_wav is None:
            parser.error(f"--task {args.task} requires --in IN.wav")
        if not Path(args.in_wav).exists():
            parser.error(f"input wav not found: {args.in_wav}")
    if args.task == "tts" and not args.text:
        parser.error("--task tts requires --text")
    if args.task in ("s2s", "tts", "duplex") and args.out is None:
        parser.error(f"--task {args.task} requires --out OUT.wav")
    if args.voice is not None and not Path(args.voice).exists():
        parser.error(f"voice reference wav not found: {args.voice}")

    device = _resolve_device(args.device, parser)

    # ---- config + model -----------------------------------------------------
    cfg = load_config(args.config if args.config else args.preset)
    if args.ckpt is not None:
        ckpt = Path(args.ckpt)
        weights_ok = (ckpt / "model.safetensors").exists() or (
            ckpt / "adapters.safetensors"
        ).exists()
        if not (ckpt / "config.yaml").exists() or not weights_ok:
            parser.error(
                f"--ckpt {args.ckpt} needs config.yaml + model.safetensors "
                "(or adapters.safetensors for backbone models)"
            )
        model = load_model(ckpt)
        cfg.model = model.cfg  # keep generator/codec checks coherent with the weights
        _info(f"loaded checkpoint {args.ckpt}")
    else:
        model = build_model(cfg.model)
        cfg.model = model.cfg  # adopt backbone-derived fields
        _info(
            f"no --ckpt: RANDOM-INIT '{cfg.preset}' model (wiring demo; expect noise)"
        )
    _info(
        f"model: {model.param_counts()['total']:,} params, device={device.type}, "
        f"task={args.task}, seed={args.seed}"
    )

    # ---- tokenizer + codec --------------------------------------------------
    if args.tokenizer != "byte" and not Path(args.tokenizer).exists():
        parser.error(f"tokenizer file not found: {args.tokenizer}")
    tokenizer = build_tokenizer(args.tokenizer)
    if tokenizer.vocab_size > cfg.model.text_vocab_size:
        _info(
            f"warning: tokenizer vocab {tokenizer.vocab_size} exceeds model "
            f"text_vocab_size {cfg.model.text_vocab_size}; high ids will fail"
        )

    # codec stays on CPU: encode/decode are cheap next to the decode loop.
    codec = build_codec(
        args.codec, model_id=cfg.codec_model_id, n_codebooks=cfg.model.n_codebooks
    )
    if codec.codec_vocab != cfg.model.audio_codec_vocab:
        parser.error(
            f"codec vocab {codec.codec_vocab} != model audio_codec_vocab "
            f"{cfg.model.audio_codec_vocab}"
        )

    if (args.task == "duplex") != cfg.model.duplex:
        parser.error(
            "--task duplex needs a duplex model (model.duplex=true) and duplex "
            f"models only run --task duplex; got task={args.task}, "
            f"model.duplex={cfg.model.duplex}"
        )

    # ---- voice reference (DESIGN_V5) ----------------------------------------
    # One <voice> segment of encoded reference codes rides the prompt after
    # <bos>, pinning the assistant timbre; --emotion/--lang tags stay in force.
    voice_codes = None
    if args.voice is not None:
        if args.task == "asr":
            _info("warning: --voice is ignored for --task asr (text-only transcription)")
        else:
            ref_wav = load_wav(args.voice, codec.sample_rate)
            voice_codes = codec.encode(ref_wav).to(device="cpu", dtype=torch.long)
            voice_codes = voice_codes[:, :DEFAULT_VOICE_FRAMES]  # 10 s cap
            if voice_codes.shape[1] < 1:
                parser.error(f"--voice {args.voice} too short: needs at least one 80 ms frame")
            _info(
                f"voice reference {args.voice}: {voice_codes.shape[1]} frames "
                f"({voice_codes.shape[1] / codec.frame_rate:.2f}s)"
            )

    # ---- run ------------------------------------------------------------
    if args.task == "duplex":
        from .duplex import DuplexGenerator

        wav_in = load_wav(args.in_wav, codec.sample_rate)
        _info(f"encoded {args.in_wav}: {wav_in.shape[0] / codec.sample_rate:.2f}s")
        dgen = DuplexGenerator(
            model, cfg, device=device, tokenizer=tokenizer, seed=args.seed,
            voice_codes=voice_codes,
        )
        text, wav_out = dgen.run_file(wav_in, codec)
        print(text if text else "")
        save_wav(args.out, wav_out, codec.sample_rate)
        _info(
            f"wrote {args.out}: {wav_out.shape[0] / codec.sample_rate:.2f}s "
            "(assistant track, same timeline as the input)"
        )
        return 0

    generator = OmniGenerator(model, cfg, device=device, tokenizer=tokenizer)
    kw = {"seed": args.seed, "max_frames": args.max_frames}
    # DESIGN_V4: --lang/--emotion force the monologue control tags; the model's
    # own perceived-emotion field stays free (it is skipped when forcing).
    prefix = turn_prefix(lang=args.lang, response_style=args.emotion)
    if prefix:
        kw["prefix_ids"] = prefix
    if voice_codes is not None:  # tts/s2s (asr warned + dropped above)
        kw["voice_codes"] = voice_codes
    if args.task == "tts":
        result = generator.tts(args.text, codec, **kw)
    elif args.task == "s2s":
        wav_in = load_wav(args.in_wav, codec.sample_rate)
        _info(f"encoded {args.in_wav}: {wav_in.shape[0] / codec.sample_rate:.2f}s")
        result = generator.s2s(wav_in, codec, **kw)
    else:  # asr
        wav_in = load_wav(args.in_wav, codec.sample_rate)
        _info(f"encoded {args.in_wav}: {wav_in.shape[0] / codec.sample_rate:.2f}s")
        result = generator.asr(wav_in, codec, **kw)

    print(result.text if result.text else "")
    if args.task in ("s2s", "tts"):
        if result.frames == 0:
            _info("warning: 0 audio frames generated; writing an empty wav")
        wav_out = codec.decode(result.audio_codes)
        save_wav(args.out, wav_out, codec.sample_rate)
        _info(
            f"wrote {args.out}: {result.frames} frames "
            f"({result.frames / codec.frame_rate:.2f}s)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
