"""Text tokenizers for the omni text stream.

Both tokenizers honor the frozen id layout of ``omni.streams``: ids 0..63 are
exactly ``streams.SPECIAL_TOKENS`` (ids 0..10) plus ``<reserved_11>``..
``<reserved_63>`` fillers, and real text ids start at 64.

- :class:`TextTokenizer`: byte-level BPE (HF ``tokenizers``), trained with
  :func:`train_bpe`; specials are pinned by handing them to the trainer first.
- :class:`ByteTokenizer`: zero-artifact stand-in mapping byte ``b -> 64 + b``
  (vocab 320), used for tiny/CI runs (``model.text_vocab_size=320``).
- :class:`HFTextTokenizer` (v6, DESIGN_V6 §3): a pretrained backbone's HF
  tokenizer with every id shifted up by 64, so the reserved special layout
  survives unchanged: omni id ``i < 64`` is a special, ``i >= 64`` is backbone
  id ``i - 64``.

No tokenizer ever auto-adds specials (or the backbone's own BOS/EOS) while
encoding.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import tokenizers

from ..streams import N_RESERVED_SPECIALS, RESERVED_SPECIAL_FORMAT, SPECIAL_TOKENS

_N_BYTES = 256  # byte-level BPE initial alphabet size


def special_token_strings() -> list[str]:
    """The 64 reserved special-token strings, index == token id (0..63)."""
    by_id = {i: name for name, i in SPECIAL_TOKENS.items()}
    assert len(by_id) == len(SPECIAL_TOKENS), "duplicate ids in SPECIAL_TOKENS"
    return [
        by_id.get(i, RESERVED_SPECIAL_FORMAT.format(i=i))
        for i in range(N_RESERVED_SPECIALS)
    ]


class TextTokenizer:
    """Byte-level BPE over ``tokenizers.Tokenizer`` with pinned special ids.

    Construction validates that ids 0..63 are exactly the reserved specials;
    BPE ids (256 byte symbols + merges) start at 64.
    """

    def __init__(self, tok: tokenizers.Tokenizer):
        self._tok = tok
        for i, s in enumerate(special_token_strings()):
            got = tok.token_to_id(s)
            if got != i:
                raise ValueError(
                    f"special token {s!r} must have id {i}, got {got}; "
                    "this tokenizer was not trained with train_bpe"
                )

    @property
    def vocab_size(self) -> int:
        """Total vocab including the 64 reserved specials (== max id + 1)."""
        return int(self._tok.get_vocab_size(with_added_tokens=True))

    def encode(self, text: str) -> list[int]:
        """Text -> BPE ids (>= 64 unless the text spells a special literally).

        Never auto-adds <bos>/<eos> or any other special.
        """
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        """Ids -> text. ``skip_specials=False`` renders "<bos>"-style strings."""
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=skip_specials)

    def save(self, path: str | Path) -> None:
        """Serialize to a single tokenizers-JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(p))

    @classmethod
    def load(cls, path: str | Path) -> "TextTokenizer":
        return cls(tokenizers.Tokenizer.from_file(str(path)))


class ByteTokenizer:
    """UTF-8 byte tokenizer: byte ``b -> id 64 + b``; no artifacts to load.

    Duck-typed like :class:`TextTokenizer` (vocab_size / encode / decode).
    ``vocab_size`` is 320 (64 reserved specials + 256 bytes).
    """

    def __init__(self) -> None:
        self.vocab_size: int = N_RESERVED_SPECIALS + _N_BYTES  # 320

    def encode(self, text: str) -> list[int]:
        """Text -> ids in [64, 320). Never auto-adds specials."""
        return [N_RESERVED_SPECIALS + b for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        """Ids -> text via UTF-8 (errors="replace").

        Special ids (< 64) are skipped, or rendered as their "<name>" strings
        when ``skip_specials=False``. Out-of-range ids are always skipped.
        """
        names = special_token_strings()
        parts: list[str] = []
        buf = bytearray()
        for raw in ids:
            i = int(raw)
            if N_RESERVED_SPECIALS <= i < self.vocab_size:
                buf.append(i - N_RESERVED_SPECIALS)
            elif 0 <= i < N_RESERVED_SPECIALS and not skip_specials:
                if buf:
                    parts.append(buf.decode("utf-8", errors="replace"))
                    buf = bytearray()
                parts.append(names[i])
        if buf:
            parts.append(buf.decode("utf-8", errors="replace"))
        return "".join(parts)


class HFTextTokenizer:
    """A backbone's Hugging Face tokenizer offset into the omni id space.

    Duck-typed like :class:`TextTokenizer` (``vocab_size`` / ``encode`` /
    ``decode``). Backbone token ``b`` becomes omni id ``64 + b``; ids 0..63
    stay the reserved omni specials, which the backbone tokenizer never sees
    (the model embeds them through its own 64-row special table, DESIGN_V6 §3).

    ``hf_tok`` needs only ``encode(text, add_special_tokens=False) -> list[int]``
    and ``decode(ids, skip_special_tokens=...) -> str`` plus a length
    (``len()`` or ``vocab_size``) — a real ``transformers`` tokenizer or a
    test stand-in both qualify.

    NOTE: ``vocab_size`` here is 64 + the TOKENIZER's size; the model's
    ``text_vocab_size`` is 64 + the backbone's EMBEDDING rows, which may be
    larger (padded embeddings, e.g. Qwen). Tokenizer <= model always holds.
    """

    def __init__(self, hf_tok, model_id: str | None = None):
        self._tok = hf_tok
        self.model_id = model_id
        try:
            n = len(hf_tok)
        except TypeError:
            n = int(hf_tok.vocab_size)
        self.vocab_size: int = N_RESERVED_SPECIALS + n

    def encode(self, text: str) -> list[int]:
        """Text -> omni ids (all >= 64). Never adds omni or backbone specials."""
        return [
            N_RESERVED_SPECIALS + i
            for i in self._tok.encode(text, add_special_tokens=False)
        ]

    def decode(self, ids: Sequence[int], skip_specials: bool = True) -> str:
        """Omni ids -> text. ``skip_specials=False`` renders "<bos>"-style
        strings for ids < 64; out-of-range ids are always skipped. The
        backbone tokenizer's own specials are kept verbatim (they can only
        appear if the model sampled them)."""
        names = special_token_strings()
        parts: list[str] = []
        buf: list[int] = []

        def flush() -> None:
            if buf:
                parts.append(self._tok.decode(buf, skip_special_tokens=False))
                buf.clear()

        for raw in ids:
            i = int(raw)
            if N_RESERVED_SPECIALS <= i < self.vocab_size:
                buf.append(i - N_RESERVED_SPECIALS)
            elif 0 <= i < N_RESERVED_SPECIALS and not skip_specials:
                flush()
                parts.append(names[i])
        flush()
        return "".join(parts)

    @classmethod
    def from_pretrained(cls, model_id: str) -> "HFTextTokenizer":
        """Load a backbone tokenizer from the HF hub (downloads on first use;
        only reachable from explicit user paths, per the no-network rule)."""
        try:
            from transformers import AutoTokenizer
        except ImportError as e:  # transformers is a core dep, but be explicit
            raise ImportError(
                "HFTextTokenizer.from_pretrained needs the 'transformers' package"
            ) from e
        return cls(AutoTokenizer.from_pretrained(model_id), model_id=model_id)


def train_bpe(
    texts: Iterable[str],
    vocab_size: int = 32_768,
    out_path: str | Path | None = None,
) -> TextTokenizer:
    """Train a byte-level BPE on `texts` with specials pinned at ids 0..63.

    GPT-2-style byte-level pretokenization (whitespace regex) plus individual
    digit splitting; the full 256-byte alphabet is always included, so encode
    never produces <unk>. `vocab_size` is the TOTAL size (64 specials + 256
    bytes + merges), hence must be >= 320. Saves to `out_path` when given.
    """
    from tokenizers import decoders, models, pre_tokenizers, trainers

    assert vocab_size >= N_RESERVED_SPECIALS + _N_BYTES, (
        f"vocab_size must be >= {N_RESERVED_SPECIALS + _N_BYTES} "
        "(64 specials + 256 byte symbols)"
    )
    tok = tokenizers.Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        # BpeTrainer assigns vocabulary ids in order: specials first (0..63),
        # then the initial alphabet (64..319), then learned merges.
        special_tokens=special_token_strings(),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tt = TextTokenizer(tok)  # validates the pinned specials
    if out_path is not None:
        tt.save(out_path)
    return tt


def build_tokenizer(path: str | None) -> "TextTokenizer | ByteTokenizer | HFTextTokenizer":
    """None or "byte" -> ByteTokenizer(); "hf:<model_id>" ->
    HFTextTokenizer.from_pretrained (downloads); anything else ->
    TextTokenizer.load(path)."""
    if path is None or path == "byte":
        return ByteTokenizer()
    if path.startswith("hf:"):
        return HFTextTokenizer.from_pretrained(path[len("hf:"):])
    return TextTokenizer.load(path)
