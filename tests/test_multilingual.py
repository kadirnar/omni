"""Multilingual-throughout coverage (DESIGN_V4 §3 + 2026-07 multilingual pass):
language normalization, non-ASCII text through every tokenizer and grid,
language-tagged fake data, language-aware synthesis, mixture recipes.

The architecture premise under test: language rides the text stream as
reserved ``<lang_XX>`` ids (audio path is language-agnostic), so multilingual
support must hold at every layer — prep, shards, batches, generation."""

from __future__ import annotations

import inspect

import pytest
import torch

from omni.data.dataset import ShardDataset, build_dataloader
from omni.data.prepare import _lang_label, prepare_fake
from omni.data.synthesize import (
    FAKE_LANGS,
    SineTTS,
    UniformAligner,
    WhisperAligner,
    build_aligner,
    fake_dialogues,
)
from omni.streams import LANG_TAGS, TEXT_STREAM, turn_prefix
from omni.text.tokenizer import ByteTokenizer, train_bpe

MULTI = "merhaba dünya 你好世界 こんにちは şeker çiçek"


# ---------------------------------------------------------------- normalization
def test_lang_label_normalization() -> None:
    assert _lang_label("en") == "en"
    assert _lang_label("en-US") == "en"
    assert _lang_label("en_GB") == "en"
    assert _lang_label("English") == "en"
    assert _lang_label("cmn") == "zh"
    assert _lang_label("zh-CN") == "zh"
    assert _lang_label("zh-Hant") == "zh"
    assert _lang_label("pt_BR") == "pt"
    assert _lang_label("es-419") == "es"
    assert _lang_label("Turkish") == "tr"
    assert _lang_label("tur") == "tr"
    assert _lang_label("deu") == "de"
    # outside the 12-tag inventory / garbage -> None
    assert _lang_label("sw") is None
    assert _lang_label("klingon") is None
    assert _lang_label("") is None
    assert _lang_label(None) is None
    # every LANG_TAGS key normalizes to itself
    for key in LANG_TAGS:
        assert _lang_label(key) == key


def test_lang_and_lang_column_are_exclusive(test_cfg, byte_tok, tmp_path) -> None:
    from omni.audio.codec import FakeCodec
    from omni.data.prepare import prepare_asr_tts

    with pytest.raises(ValueError, match="lang_column"):
        prepare_asr_tts(
            tmp_path / "x", dataset_id="d", name=None, split="train",
            codec=FakeCodec(n_codebooks=2), tokenizer=byte_tok, cfg=test_cfg,
            max_samples=1, lang="en", lang_column="locale",
        )


# ------------------------------------------------------------------ tokenizers
def test_byte_tokenizer_full_unicode_roundtrip() -> None:
    tok = ByteTokenizer()
    ids = tok.encode(MULTI)
    assert ids and all(64 <= i < 320 for i in ids)  # UTF-8 bytes, never specials
    assert tok.decode(ids) == MULTI


def test_train_bpe_multilingual_corpus(tmp_path) -> None:
    """A BPE trained on a mixed corpus keeps the frozen special layout and
    round-trips every script losslessly (byte fallback)."""
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "merhaba dünya bugün hava çok güzel şarkı söylüyoruz",
        "你好世界 今天天气很好 我们一起唱歌",
        "bonjour le monde la lumière est déjà là",
    ] * 40
    tok = train_bpe(corpus, vocab_size=512, out_path=tmp_path / "bpe.json")
    # vocab_size caps merges; a tiny corpus may exhaust merges below the cap
    assert 320 <= tok.vocab_size <= 512
    # frozen special layout survives multilingual training
    assert tok.encode("<s2s>")[0] == 9
    assert tok.encode("<lang_tr>")[0] == LANG_TAGS["tr"]
    assert tok.encode("<lang_zh>")[0] == LANG_TAGS["zh"]
    for text in ("şeker ve çiçek", "你好世界", "déjà vu"):
        ids = tok.encode(text)
        assert all(i >= 64 for i in ids)
        assert tok.decode(ids) == text


# ------------------------------------------------------------------- fake data
def test_fake_dialogues_rotate_languages() -> None:
    dialogues = list(fake_dialogues(8, seed=0))
    labeled = [d for k, d in enumerate(dialogues) if k % 2 == 1]
    langs = {t["lang"] for d in labeled for t in d["turns"]}
    assert langs >= {"en", "tr", "zh"}  # rotation visible within 8 dialogues
    assert langs <= set(FAKE_LANGS)
    # the zh-labeled dialogues really contain CJK text
    zh_turns = [t for d in labeled for t in d["turns"] if t["lang"] == "zh"]
    assert zh_turns and any(
        any("一" <= ch <= "鿿" for ch in t["user"]) for t in zh_turns
    )
    # unlabeled dialogues carry no lang key (v1 shape preserved)
    unlabeled = [d for k, d in enumerate(dialogues) if k % 2 == 0]
    assert all("lang" not in t for d in unlabeled for t in d["turns"])


def test_prepare_fake_emits_lang_tags(tmp_path, test_cfg) -> None:
    prepare_fake(tmp_path / "shards", n_samples=40, cfg=test_cfg, seed=0)
    ds = ShardDataset(tmp_path / "shards")
    lang_ids = set(LANG_TAGS.values())
    seen: set[int] = set()
    untagged = 0
    for i in range(len(ds)):
        row = ds[i].grid[TEXT_STREAM].tolist()
        hits = {v for v in row if v in lang_ids}
        seen |= hits
        if not hits:
            untagged += 1
    # rotation (None, en, tr, zh): three tagged languages plus untagged rows
    assert {LANG_TAGS["en"], LANG_TAGS["tr"], LANG_TAGS["zh"]} <= seen
    assert untagged > 0, "the None slot must keep untagged coverage"


def test_multilingual_batches_reach_the_model(tmp_path, test_cfg) -> None:
    """Shards -> dataloader -> a real forward/loss step on lang-tagged,
    non-ASCII multilingual grids (the whole offline training path)."""
    from omni.model.omni import OmniModel

    prepare_fake(tmp_path / "shards", n_samples=20, cfg=test_cfg, seed=1)
    test_cfg.data.num_workers = 0
    loader = build_dataloader(test_cfg, [str(tmp_path / "shards")])
    torch.manual_seed(0)
    model = OmniModel(test_cfg.model)
    lang_ids = set(LANG_TAGS.values())
    saw_lang = False
    for batch in loader:
        out = model(batch["grid"], batch["channel"])
        total, _ = model.loss(out, batch["grid"], batch["loss_mask"])
        assert torch.isfinite(total)
        saw_lang = saw_lang or bool(
            torch.isin(
                batch["grid"][:, TEXT_STREAM],
                torch.tensor(sorted(lang_ids)),
            ).any()
        )
    assert saw_lang, "no language tag ever reached the model"


# ------------------------------------------------------------------- synthesis
def test_sinetts_lang_is_deterministic_accent() -> None:
    tts = SineTTS()
    a1, _ = tts.synth("hello world", "alto")
    a2, _ = tts.synth("hello world", "alto", lang=None)
    assert torch.equal(a1, a2)  # lang=None is bit-identical to the old path
    b1, _ = tts.synth("hello world", "alto", lang="tr")
    b2, _ = tts.synth("hello world", "alto", lang="tr")
    assert torch.equal(b1, b2)  # deterministic per language
    assert not torch.equal(a1, b1)  # observable language effect


def test_whisper_aligner_defaults_multilingual() -> None:
    """Regression pin: the aligner default must be the MULTILINGUAL Whisper
    checkpoint (the old whisper-tiny.en silently broke 11 of 12 languages)."""
    default = inspect.signature(WhisperAligner.__init__).parameters["model_id"].default
    assert default == "openai/whisper-tiny"
    assert not default.endswith(".en")
    assert isinstance(build_aligner("uniform", lang="tr"), UniformAligner)
    assert build_aligner("none", lang="tr") is None


# ------------------------------------------------------------------- inference
def test_generation_carries_forced_lang_tag(test_cfg, byte_tok, fake_codec) -> None:
    """chat --lang / serve lang dropdown path: a forced <lang_XX> prefix lands
    in the monologue record for a non-Latin language."""
    from omni.infer.generate import OmniGenerator
    from omni.model.omni import OmniModel

    torch.manual_seed(0)
    gen = OmniGenerator(
        OmniModel(test_cfg.model), test_cfg, device="cpu", tokenizer=byte_tok
    )
    wav = torch.sin(torch.linspace(0, 300, fake_codec.samples_per_frame * 4))
    r = gen.s2s(wav, fake_codec, prefix_ids=turn_prefix(lang="zh"), max_frames=6, seed=0)
    assert LANG_TAGS["zh"] in r.text_ids


# ------------------------------------------------------------------ mixture
def test_per_language_mixture_recipe(tmp_path, test_cfg) -> None:
    """The documented multilingual data layout: one shard dir per language,
    mixed with explicit proportions — counts must follow the recipe."""
    from omni.data.dataset import MixDataset

    dirs = []
    for i in range(2):
        d = tmp_path / f"lang{i}"
        prepare_fake(d, n_samples=15 if i == 0 else 5, cfg=test_cfg, seed=i)
        dirs.append(ShardDataset(d))
    mix = MixDataset(dirs, [0.5, 0.5], seed=0)  # En-heavy corpus, 50/50 recipe
    assert mix.counts == [10, 10]  # small language oversampled to its share
