"""Tests for omni.text.tokenizer: ByteTokenizer, train_bpe, save/load, specials."""

from __future__ import annotations

import random
import re

import pytest

from omni.streams import BOS, EOS, PAD, SPECIAL_TOKENS, TASK_ASR
from omni.text.tokenizer import (
    ByteTokenizer,
    TextTokenizer,
    build_tokenizer,
    train_bpe,
)

_WORDS = (
    "the quick brown fox jumps over a lazy dog while zephyrs blow vexing "
    "daft jim pack my box with five dozen liquor jugs speech model token "
    "audio frame turn hello world data train test omni sound wave code text "
    "stream delay grid mask learning neural network sample rate frequency "
    "convert whisper listen speak answer question dialogue assistant user "
    "system prompt zero one two three four five six seven eight nine Big "
    "Cat Dog Time Life Work Year Day Way Thing Man World School State "
    "family student group country problem hand part place case week company "
    "number point government night water room mother area money story fact "
    "month lot right study book eye job word business issue side kind head "
    "house service friend father power hour game line end member law car "
    "city community name team minute idea body information back parent face "
    "others level office door health person art war history party result"
).split()


def _corpus(n_lines: int = 500) -> list[str]:
    rng = random.Random(0)
    return [" ".join(rng.choice(_WORDS) for _ in range(10)) for _ in range(n_lines)]


@pytest.fixture(scope="module")
def bpe_tok() -> TextTokenizer:
    return train_bpe(_corpus(), vocab_size=384)


def test_byte_tokenizer_roundtrip() -> None:
    bt = ByteTokenizer()
    assert bt.vocab_size == 320
    s = "héllo wörld ☃ 123"
    ids = bt.encode(s)
    assert ids == [64 + b for b in s.encode("utf-8")]
    assert min(ids) >= 64 and max(ids) < 320
    assert bt.decode(ids) == s
    # specials are skipped on decode by default
    assert bt.decode([BOS] + ids + [EOS, PAD]) == s


def test_byte_tokenizer_ascii_ids() -> None:
    bt = ByteTokenizer()
    assert bt.encode("A") == [64 + ord("A")]
    assert bt.encode("") == []
    assert bt.decode([]) == ""


def test_bpe_vocab_size_and_base_ids(bpe_tok: TextTokenizer) -> None:
    assert bpe_tok.vocab_size == 384, "requested vocab_size must be honored"
    ids = bpe_tok.encode("the quick brown fox")
    assert ids, "plain text must tokenize to something"
    assert all(i >= 64 for i in ids), "BPE ids start at 64; encode never adds specials"


def test_bpe_roundtrip_and_special_skip(bpe_tok: TextTokenizer) -> None:
    s = "the quick brown fox jumps over the lazy dog"
    ids = bpe_tok.encode(s)
    assert bpe_tok.decode(ids).strip() == s
    assert bpe_tok.decode([BOS] + ids + [EOS, PAD]).strip() == s


def test_bpe_specials_at_pinned_ids(bpe_tok: TextTokenizer) -> None:
    for name, i in SPECIAL_TOKENS.items():
        shown = bpe_tok.decode([i], skip_specials=False)
        assert name in shown, f"id {i} must render as {name!r}, got {shown!r}"
        assert bpe_tok.decode([i]).strip() == "", f"id {i} must be skippable"
    # ids len(SPECIAL_TOKENS)..63 are reserved filler specials (v4 names 0..48)
    for i in (len(SPECIAL_TOKENS), 55, 63):
        shown = bpe_tok.decode([i], skip_specials=False).strip()
        assert re.fullmatch(r"<reserved_\d+>", shown), f"id {i} -> {shown!r}"


def test_bpe_save_load_roundtrip(tmp_path, bpe_tok: TextTokenizer) -> None:
    p = tmp_path / "tok.json"
    bpe_tok.save(p)
    assert p.exists()
    tok2 = TextTokenizer.load(p)
    assert tok2.vocab_size == bpe_tok.vocab_size
    for s in ("hello world", "the assistant answers the user question"):
        assert tok2.encode(s) == bpe_tok.encode(s)
        assert tok2.decode(tok2.encode(s)) == bpe_tok.decode(bpe_tok.encode(s))
    assert "<asr>" in tok2.decode([TASK_ASR], skip_specials=False)


def test_train_bpe_out_path(tmp_path) -> None:
    p = tmp_path / "bpe.json"
    tok = train_bpe(_corpus(200), vocab_size=340, out_path=p)
    assert p.exists(), "train_bpe(out_path=...) must write the artifact"
    assert tok.vocab_size == 340
    loaded = build_tokenizer(str(p))
    assert isinstance(loaded, TextTokenizer)
    assert loaded.encode("hello world") == tok.encode("hello world")


def test_build_tokenizer_byte_paths() -> None:
    assert isinstance(build_tokenizer(None), ByteTokenizer)
    assert isinstance(build_tokenizer("byte"), ByteTokenizer)
