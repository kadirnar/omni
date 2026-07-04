#!/usr/bin/env python
"""Train the omni byte-level BPE text tokenizer.

Corpus sources (combinable; at least one required):
  --dataset HF_ID[:config[:split]][@WEIGHT]   stream an HF dataset (downloads!);
                                              REPEATABLE — multiple datasets are
                                              interleaved by WEIGHT (default 1.0),
                                              the DESIGN_V4 multilingual mix
  --text-file PATH [PATH ...]                 local text files, one document/line

For the multilingual 48k BPE (DESIGN_V4 SS3) mix per-language corpora with
weights, e.g. En35/Zh20/Fr,De,Tr10/Es5/misc10 — keep >=~20% Chinese so common
CJK characters land in-vocab rather than byte-fallback (the 12.5 tok/s
monologue budget cannot afford 3 bytes/char).

Examples:
  python scripts/train_tokenizer.py --text-file corpus.txt --vocab-size 4096
  python scripts/train_tokenizer.py \
      --dataset HuggingFaceFW/fineweb-edu:sample-10BT:train@35 \
      --dataset uonlp/CulturaX:zh@20 --dataset uonlp/CulturaX:tr@10 \
      --field text --max-docs 2000000 --vocab-size 48000 \
      --out data/tokenizer/omni_bpe_48k.json
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from omni.text.tokenizer import train_bpe


def _parse_dataset_spec(spec: str) -> tuple[str, float]:
    """"HF_ID[:config[:split]][@WEIGHT]" -> (source spec, weight)."""
    head, sep, tail = spec.rpartition("@")
    if sep:
        try:
            return head, float(tail)
        except ValueError:
            pass  # '@' was part of the id/path text
    return spec, 1.0


def _iter_dataset(spec: str, field: str, max_docs: int | None) -> Iterator[str]:
    """Yield the `field` column of a streamed HF dataset (user-invoked download)."""
    import datasets  # heavy import + network: only on the explicit --dataset path

    parts = spec.split(":")
    if len(parts) > 3:
        raise ValueError(f"--dataset must be HF_ID[:config[:split]], got {spec!r}")
    dataset_id = parts[0]
    name = parts[1] if len(parts) > 1 and parts[1] else None
    split = parts[2] if len(parts) > 2 and parts[2] else "train"
    ds = datasets.load_dataset(dataset_id, name, split=split, streaming=True)
    n = 0
    for row in ds:
        if max_docs is not None and n >= max_docs:
            return
        text = row.get(field)
        if isinstance(text, str) and text.strip():
            n += 1
            yield text


def _interleave(sources: "list[Iterator[str]]", weights: list[float]) -> Iterator[str]:
    """Deterministic weighted round-robin over document iterators.

    Each round grants every live source credit proportional to its weight and
    yields floor(credit) documents from it (credit carries over), so long-run
    document proportions match the weights without any rng. Exhausted sources
    drop out; iteration ends when all are exhausted.
    """
    live = list(range(len(sources)))
    credit = [0.0] * len(sources)
    scale = min(w for w in weights) if weights else 1.0
    while live:
        for i in list(live):
            credit[i] += weights[i] / scale  # slowest source ~1 doc/round
            while credit[i] >= 1.0:
                credit[i] -= 1.0
                try:
                    yield next(sources[i])
                except StopIteration:
                    live.remove(i)
                    break


def _iter_files(paths: list[str], max_docs: int | None) -> Iterator[str]:
    """Yield non-empty lines (one document each) from local text files."""
    n = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if max_docs is not None and n >= max_docs:
                    return
                line = line.strip()
                if line:
                    n += 1
                    yield line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        metavar="HF_ID[:config[:split]][@WEIGHT]",
        help="stream this HF dataset (downloads; split defaults to 'train'); "
             "repeatable — multiple datasets interleave by @WEIGHT (default 1.0) "
             "for a multilingual mix",
    )
    parser.add_argument(
        "--text-file",
        nargs="+",
        action="extend",
        default=None,
        metavar="PATH",
        help="local text file(s), one document per line (repeatable)",
    )
    parser.add_argument("--field", default="text", help="dataset text column (default: text)")
    parser.add_argument(
        "--max-docs", type=int, default=None, metavar="N",
        help="cap the number of training documents (default: all)",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=32_768,
        help="total vocab incl. 64 reserved specials (default: 32768, min 320)",
    )
    parser.add_argument(
        "--out", default="data/tokenizer/omni_bpe.json",
        help="output tokenizer JSON path (default: data/tokenizer/omni_bpe.json)",
    )
    args = parser.parse_args(argv)

    if args.dataset is None and args.text_file is None:
        parser.error("give --dataset (repeatable) and/or --text-file")

    sources: list[Iterator[str]] = []
    weights: list[float] = []
    names: list[str] = []
    for spec in args.dataset or []:
        ds_spec, w = _parse_dataset_spec(spec)
        # per-source cap: the global --max-docs applies to the mixed stream
        sources.append(_iter_dataset(ds_spec, args.field, None))
        weights.append(w)
        names.append(f"{ds_spec}@{w:g}")
    if args.text_file:
        for p in args.text_file:
            if not Path(p).exists():
                parser.error(f"text file not found: {p}")
        sources.append(_iter_files(args.text_file, None))
        weights.append(1.0)
        names.append(f"files{args.text_file}")

    if len(sources) == 1:
        mixed: Iterator[str] = sources[0]
    else:
        mixed = _interleave(sources, weights)
    if args.max_docs is not None:
        import itertools

        mixed = itertools.islice(mixed, args.max_docs)
    texts = mixed
    source = " + ".join(names) + f" (field {args.field!r})"

    from tqdm import tqdm

    print(f"training byte-level BPE (vocab_size={args.vocab_size}) on {source}")
    tok = train_bpe(
        tqdm(texts, desc="docs", unit="doc"),
        vocab_size=args.vocab_size,
        out_path=args.out,
    )
    sample = "Hello omni, 12 speech tokens please!"
    ids = tok.encode(sample)
    print(f"saved tokenizer to {args.out}")
    print(f"vocab_size={tok.vocab_size} (specials 0..63 pinned, BPE ids from 64)")
    print(f"sanity: {sample!r} -> {len(ids)} ids -> {tok.decode(ids)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
