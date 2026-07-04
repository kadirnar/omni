#!/usr/bin/env python
"""Standalone dialogue -> TTS wav -> codec codes -> s2s shard pipeline.

Thin wrapper over `omni.data.prepare.prepare_s2s`, plus `--dump-wav DIR` to
write every synthesized utterance as a WAV for listening/inspection. Fully
offline with the defaults (--dialogues fake --tts sine --codec fake); `soda`
/ `ultrachat` dialogues stream from the HF hub, and non-sine `--tts` backends /
`--codec mimi` download model weights on first use.

Examples:
  python scripts/synthesize_data.py --codec fake --max-samples 50 \\
      --out data/shards/s2s_fake --dump-wav /tmp/s2s_wavs \\
      --preset tiny model.text_vocab_size=320
  python scripts/synthesize_data.py --dialogues soda --tts vibevoice --codec mimi \\
      --tokenizer data/tokenizer/omni_bpe.json --max-samples 5000 \\
      --out data/shards/s2s_soda
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from omni.config import load_config

if TYPE_CHECKING:  # annotation-only; the script defers heavy imports to main()
    import torch

_DIALOGUE_ALIASES = {
    "soda": ("allenai/soda", "train"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", "train_sft"),
}


def _dialogue_source(spec: str, split: str | None, max_samples: int, seed: int) -> Iterator[dict]:
    """'fake' | 'soda' | 'ultrachat' | any HF dataset id -> dialogue dict iterator."""
    from omni.data.synthesize import fake_dialogues, load_text_dialogues

    if spec == "fake":
        return fake_dialogues(max_samples, seed=seed)
    dataset_id, default_split = _DIALOGUE_ALIASES.get(spec, (spec, "train"))
    # headroom: prepare_s2s may skip dialogues that cannot fit max_sample_frames
    return load_text_dialogues(dataset_id, split or default_split, 2 * max_samples)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.strip().splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dialogues", default="fake", metavar="soda|ultrachat|fake|HF_ID",
        help="dialogue source (default: fake, offline)",
    )
    parser.add_argument("--split", default=None, help="dialogue split (source-specific default)")
    parser.add_argument("--max-samples", type=int, default=100, metavar="N")
    parser.add_argument(
        "--tts", default="sine", choices=("sine", "vibevoice", "cosyvoice3", "soulx", "chatterbox"),
        help="TTS backend; non-sine backends need their optional packages and a GPU "
             "(see docs/DESIGN_V3_AUDIO.md; default: sine)",
    )
    parser.add_argument(
        "--codec", required=True, choices=("fake", "mimi"),
        help="audio codec; 'mimi' downloads kyutai/mimi weights on first use",
    )
    parser.add_argument("--device", default="cpu", help="device for codec encoding (default: cpu)")
    parser.add_argument(
        "--tokenizer", default="byte", metavar="byte|PATH",
        help="'byte' for the ByteTokenizer, else a trained BPE JSON path (default: byte)",
    )
    parser.add_argument(
        "--dump-wav", default=None, metavar="DIR",
        help="also write every synthesized utterance to DIR as utt-NNNNNN-<voice>.wav",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--preset", "--config", dest="config", default="tiny", metavar="NAME|YAML",
        help="config preset (tiny/small/base) or YAML path (default: tiny)",
    )
    parser.add_argument("--out", required=True, metavar="DIR", help="output shard directory")
    parser.add_argument(
        "overrides", nargs="*", metavar="KEY=VALUE",
        help="dotted config overrides, e.g. model.text_vocab_size=320",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config, list(args.overrides))

    from omni.audio.codec import FakeCodec, build_codec
    from omni.data.prepare import prepare_s2s
    from omni.data.synthesize import TTSBackend, build_tts
    from omni.text.tokenizer import build_tokenizer

    class _DumpTTS(TTSBackend):
        """Pass-through TTS that also writes each utterance under --dump-wav."""

        def __init__(self, inner: TTSBackend, dump_dir: str):
            self.inner = inner
            self.voices = list(inner.voices)
            self.dump_dir = Path(dump_dir)
            self.n = 0

        def synth(self, text: str, voice: str) -> tuple[torch.Tensor, int]:
            from omni.audio.codec import save_wav

            wav, sr = self.inner.synth(text, voice)
            save_wav(self.dump_dir / f"utt-{self.n:06d}-{voice}.wav", wav, sr)
            self.n += 1
            return wav, sr

    tts: TTSBackend = build_tts(args.tts)
    if args.dump_wav is not None:
        tts = _DumpTTS(tts, args.dump_wav)
    if args.codec == "fake":
        codec = FakeCodec(
            n_codebooks=cfg.model.n_codebooks, codec_vocab=cfg.model.audio_codec_vocab
        )
    else:
        codec = build_codec(
            args.codec,
            model_id=cfg.codec_model_id,
            n_codebooks=cfg.model.n_codebooks,
            device=args.device,
        )

    print(
        f"synthesizing s2s shards: dialogues={args.dialogues!r} tts={args.tts!r} "
        f"codec={args.codec!r} max_samples={args.max_samples} -> {args.out}"
    )
    try:
        prepare_s2s(
            args.out,
            dialogues=_dialogue_source(args.dialogues, args.split, args.max_samples, args.seed),
            tts=tts,
            codec=codec,
            tokenizer=build_tokenizer(args.tokenizer),
            cfg=cfg,
            max_samples=args.max_samples,
            seed=args.seed,
        )
    except (ValueError, FileNotFoundError, ImportError) as e:
        print(f"error: {e}")
        return 1

    with open(Path(args.out) / "meta.json") as f:
        meta = json.load(f)
    print(f"wrote {meta['n_samples']} samples in {meta['n_shards']} shard(s) to {args.out}")
    if args.dump_wav is not None:
        print(f"dumped {tts.n} utterance wav(s) to {args.dump_wav}")  # type: ignore[union-attr]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
