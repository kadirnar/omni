#!/usr/bin/env python
"""Prepare training shards (binary shard dirs readable by omni.data.dataset).

Every subcommand takes --preset/--config, --out DIR and trailing KEY=VALUE
config overrides. Only `fake` is fully offline; the other subcommands stream
Hugging Face datasets (network!) and `--codec mimi` / non-sine `--tts` download
model weights on first use.

Examples:
  python scripts/prepare_data.py fake --n 200 --out data/shards/fake \\
      --preset tiny model.text_vocab_size=320
  python scripts/prepare_data.py textlm --dataset HuggingFaceFW/fineweb-edu \\
      --name sample-10BT --split train --tokenizer data/tokenizer/omni_bpe.json \\
      --max-samples 10000 --out data/shards/textlm
  python scripts/prepare_data.py asr --dataset openslr/librispeech_asr \\
      --name clean --split train.100 --codec mimi --tokenizer data/tokenizer/omni_bpe.json \\
      --max-samples 5000 --out data/shards/asr \\
      --voice-p 0.1 --speaker-column speaker_id
  python scripts/prepare_data.py s2s --dialogues soda --tts vibevoice --codec mimi \\
      --tokenizer data/tokenizer/omni_bpe.json --max-samples 2000 --out data/shards/s2s \\
      --align uniform --voice-p 0.5
  python scripts/prepare_data.py duplex --n 200 --out data/shards/duplex \\
      --preset tiny model.text_vocab_size=320
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path

from omni.config import OmniConfig, load_config

_DIALOGUE_ALIASES = {
    "soda": ("allenai/soda", "train"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "train_sft"),
}
# "soda-emotional" maps SODA's xReact field to emotion/style labels (DESIGN_V4 §2)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--preset", "--config", dest="config", default="tiny", metavar="NAME|YAML",
        help="config preset (tiny/small/base) or YAML path (default: tiny)",
    )
    p.add_argument("--out", required=True, metavar="DIR", help="output shard directory")
    p.add_argument(
        "overrides", nargs="*", metavar="KEY=VALUE",
        help="dotted config overrides, e.g. model.n_codebooks=4 data.max_sample_frames=250",
    )


def _add_tokenizer(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--tokenizer", default="byte", metavar="byte|PATH|hf:ID",
        help="'byte' for the 320-id ByteTokenizer, a trained BPE JSON path, or "
             "'hf:<model_id>' for a backbone tokenizer (default: byte)",
    )


def _add_align(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--align", default="none", choices=("none", "uniform", "whisper"),
        help="word-align the assistant monologue (default: none = packed layout; "
        "'whisper' downloads Whisper weights on first use)",
    )


def _add_voice_p(p: argparse.ArgumentParser, what: str) -> None:
    p.add_argument(
        "--voice-p", type=float, default=0.0, metavar="P",
        help=f"probability a {what} carries a reference-voice segment "
        "(docs/DESIGN_V5_VOICE.md; default: 0.0 = no segments, v4-identical)",
    )


def _add_codec(p: argparse.ArgumentParser, *, required: bool = True) -> None:
    help_ = "audio codec; 'mimi' downloads kyutai/mimi weights on first use"
    if required:
        p.add_argument("--codec", required=True, choices=("fake", "mimi"), help=help_)
    else:
        p.add_argument(
            "--codec", default="fake", choices=("fake", "mimi"),
            help=help_ + " (default: fake, offline)",
        )
    p.add_argument(
        "--device", default="cpu", metavar="cpu|cuda|mps",
        help="device for codec encoding (default: cpu)",
    )


def _build_codec(name: str, cfg: OmniConfig, device: str):
    """Codec matching cfg.model; FakeCodec honors a non-default audio_codec_vocab."""
    from omni.audio.codec import FakeCodec, build_codec

    if name == "fake":
        return FakeCodec(
            n_codebooks=cfg.model.n_codebooks, codec_vocab=cfg.model.audio_codec_vocab
        )
    return build_codec(
        name,
        model_id=cfg.codec_model_id,
        n_codebooks=cfg.model.n_codebooks,
        device=device,
    )


def _dialogue_source(spec: str, split: str | None, max_samples: int, seed: int) -> Iterator[dict]:
    """'fake' | 'soda' | 'ultrachat' | any HF dataset id -> dialogue dict iterator."""
    from omni.data.synthesize import fake_dialogues, load_text_dialogues

    if spec == "fake":
        return fake_dialogues(max_samples, seed=seed)
    if spec == "soda-emotional":
        from omni.data.synthesize import load_soda_emotional

        return load_soda_emotional(split or "train", 2 * max_samples)
    dataset_id, default_split = _DIALOGUE_ALIASES.get(spec, (spec, "train"))
    # headroom: prepare_s2s may skip dialogues that cannot fit max_sample_frames
    return load_text_dialogues(dataset_id, split or default_split, 2 * max_samples)


def _report(out_dir: str) -> None:
    with open(Path(out_dir) / "meta.json") as f:
        meta = json.load(f)
    duplex = ", duplex" if meta.get("duplex") else ""
    print(
        f"wrote {meta['n_samples']} samples in {meta['n_shards']} shard(s) to {out_dir} "
        f"(n_codebooks={meta['n_codebooks']}, codec_vocab={meta['codec_vocab']}, "
        f"text_vocab_size={meta['text_vocab_size']}{duplex})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.partition("Examples:")[2] and "Examples:" + __doc__.partition("Examples:")[2],
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fake", help="offline mixture of all 5 tasks (SineTTS + FakeCodec + bytes)")
    p.add_argument("--n", type=int, default=200, help="number of samples (default: 200)")
    p.add_argument("--seed", type=int, default=0)
    _add_common(p)

    p = sub.add_parser("textlm", help="pack an HF text dataset into full-length textlm rows")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu", metavar="HF_ID")
    p.add_argument("--name", default=None, help="dataset config name, e.g. sample-10BT")
    p.add_argument("--split", default="train")
    p.add_argument("--max-samples", type=int, required=True, metavar="N")
    p.add_argument("--no-streaming", action="store_true", help="download instead of streaming")
    p.add_argument("--lang", default=None, metavar="XX", help="prepend <lang_XX> to every row")
    _add_tokenizer(p)
    _add_common(p)

    p = sub.add_parser("asr", help="(audio, transcript) HF rows -> asr/tts/audiolm grids")
    p.add_argument("--dataset", default="openslr/librispeech_asr", metavar="HF_ID")
    p.add_argument("--name", default=None, help="dataset config name, e.g. clean")
    p.add_argument("--split", default="train.100")
    p.add_argument("--max-samples", type=int, required=True, metavar="N")
    p.add_argument(
        "--tasks", default="asr,tts,alm",
        help="comma-separated cycle over asr/tts/alm (default: asr,tts,alm)",
    )
    p.add_argument("--lang", default=None, metavar="XX", help="prepend <lang_XX> to asr/tts text")
    p.add_argument(
        "--lang-column", default=None, metavar="COL",
        help="dataset column with a per-row language label (Common Voice 'locale', "
             "FLEURS-style fields); normalized to <lang_XX>, rows outside the "
             "12-language inventory are skipped; exclusive with --lang",
    )
    p.add_argument(
        "--emotion-column", default=None, metavar="COL",
        help="dataset column with an emotion label -> SER-tagged ASR (DESIGN_V4)",
    )
    _add_voice_p(p, "row")
    p.add_argument(
        "--speaker-column", default=None, metavar="COL",
        help="dataset speaker-id column: a previous same-speaker row's codes become "
        "the voice reference (required when --voice-p > 0)",
    )
    p.add_argument("--seed", type=int, default=0, help="voice-pairing rng seed (default: 0)")
    _add_align(p)
    _add_codec(p)
    _add_tokenizer(p)
    _add_common(p)

    p = sub.add_parser("s2s", help="dialogues -> TTS -> codec -> s2s grids")
    p.add_argument(
        "--dialogues", default="fake", metavar="soda|ultrachat|fake|HF_ID",
        help="dialogue source (default: fake, offline)",
    )
    p.add_argument("--split", default=None, help="dialogue dataset split (source-specific default)")
    p.add_argument("--max-samples", type=int, required=True, metavar="N")
    p.add_argument(
        "--tts", default="sine", choices=("sine", "vibevoice", "cosyvoice3", "soulx", "chatterbox"),
        help="TTS backend; non-sine backends need their optional packages and a GPU "
             "(see docs/DESIGN_V3_AUDIO.md; default: sine)",
    )
    p.add_argument("--seed", type=int, default=0)
    _add_voice_p(p, "dialogue")
    _add_align(p)
    _add_codec(p)
    _add_tokenizer(p)
    _add_common(p)

    p = sub.add_parser(
        "duplex",
        help="synthetic full-duplex conversations (SineTTS + FakeCodec by default, "
             "offline; shards always carry duplex=true meta)",
    )
    p.add_argument("--n", type=int, default=100, help="number of conversations (default: 100)")
    p.add_argument(
        "--tts", default="sine", choices=("sine", "vibevoice", "cosyvoice3", "soulx", "chatterbox"),
        help="TTS backend; non-sine backends need their optional packages and a GPU "
             "(see docs/DESIGN_V3_AUDIO.md; default: sine)",
    )
    p.add_argument("--seed", type=int, default=0)
    _add_voice_p(p, "conversation")
    _add_codec(p, required=False)
    _add_tokenizer(p)
    _add_common(p)

    args = parser.parse_args(argv)
    if args.cmd == "asr" and args.voice_p > 0 and not args.speaker_column:
        # fail before any codec/dataset download; prepare_asr_tts re-checks
        print("error: --voice-p > 0 requires --speaker-column (DESIGN_V5 voice pairing)")
        return 1
    cfg = load_config(args.config, list(args.overrides))
    print(
        f"preparing '{args.cmd}' shards with preset={cfg.preset!r}: "
        f"n_codebooks={cfg.model.n_codebooks}, codec_vocab={cfg.model.audio_codec_vocab}, "
        f"text_vocab_size={cfg.model.text_vocab_size}, "
        f"max_sample_frames={cfg.data.max_sample_frames}"
    )

    from omni.data import prepare  # heavy import (torch) after arg validation
    from omni.text.tokenizer import build_tokenizer

    try:
        if args.cmd == "fake":
            prepare.prepare_fake(args.out, n_samples=args.n, cfg=cfg, seed=args.seed)
        elif args.cmd == "textlm":
            prepare.prepare_textlm(
                args.out,
                dataset_id=args.dataset,
                name=args.name,
                split=args.split,
                tokenizer=build_tokenizer(args.tokenizer),
                cfg=cfg,
                max_samples=args.max_samples,
                streaming=not args.no_streaming,
                lang=args.lang,
            )
        elif args.cmd == "asr":
            from omni.data.synthesize import build_aligner

            prepare.prepare_asr_tts(
                args.out,
                dataset_id=args.dataset,
                name=args.name,
                split=args.split,
                codec=_build_codec(args.codec, cfg, args.device),
                tokenizer=build_tokenizer(args.tokenizer),
                cfg=cfg,
                max_samples=args.max_samples,
                tasks=tuple(t.strip() for t in args.tasks.split(",") if t.strip()),
                aligner=build_aligner(args.align, lang=args.lang),
                lang=args.lang,
                lang_column=args.lang_column,
                emotion_column=args.emotion_column,
                voice_p=args.voice_p,
                speaker_column=args.speaker_column,
                seed=args.seed,
            )
        elif args.cmd == "s2s":
            from omni.data.synthesize import build_aligner, build_tts

            prepare.prepare_s2s(
                args.out,
                dialogues=_dialogue_source(args.dialogues, args.split, args.max_samples, args.seed),
                tts=build_tts(args.tts),
                codec=_build_codec(args.codec, cfg, args.device),
                tokenizer=build_tokenizer(args.tokenizer),
                cfg=cfg,
                max_samples=args.max_samples,
                seed=args.seed,
                aligner=build_aligner(args.align),
                voice_p=args.voice_p,
            )
        else:  # duplex
            from omni.data.synthesize import build_tts

            prepare.prepare_duplex(
                args.out,
                n_conversations=args.n,
                cfg=cfg,
                tts=build_tts(args.tts),
                codec=_build_codec(args.codec, cfg, args.device),
                tokenizer=build_tokenizer(args.tokenizer),
                seed=args.seed,
                voice_p=args.voice_p,
            )
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(f"error: {e}")
        return 1
    _report(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
