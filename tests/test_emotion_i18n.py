"""Extensions v4: emotion tags + multilingual support (docs/DESIGN_V4_EMOTION_I18N.md)."""

from __future__ import annotations

import pytest
import torch

from omni import streams
from omni.streams import (
    EMO_PCV,
    EMO_RSP,
    EMOTION_CLASSES,
    LANG_TAGS,
    N_PERCEIVED_EMOTIONS,
    PARALING_TAGS,
    SPECIAL_TOKENS,
    TASK_SER,
    turn_prefix,
)


def test_reserved_token_plan_exact() -> None:
    """Pin the DESIGN_V4 allocation: ids 11..48 named, 49..63 reserved."""
    assert TASK_SER == 11 and EMO_PCV == 12 and EMO_RSP == 13
    assert EMOTION_CLASSES["angry"] == 20 and EMOTION_CLASSES["serious"] == 28
    assert PARALING_TAGS["laugh"] == 29 and PARALING_TAGS["yawn"] == 36
    assert LANG_TAGS["en"] == 37 and LANG_TAGS["tr"] == 44 and LANG_TAGS["nl"] == 48
    assert N_PERCEIVED_EMOTIONS == 8
    ids = list(SPECIAL_TOKENS.values())
    assert len(ids) == len(set(ids)) == 52  # 0..51 named (v5 adds voice markers)
    assert sorted(ids) == list(range(52))
    assert streams.VOICE == 49 and streams.VOICE_END == 50 and streams.ACCENT_KEEP == 51
    assert all(i < streams.N_RESERVED_SPECIALS for i in ids)
    assert streams.TASK_TAGS["ser"] == TASK_SER


def test_tokenizer_specials_follow_plan(byte_tok) -> None:
    from omni.text.tokenizer import special_token_strings

    names = special_token_strings()
    assert names[20] == "<angry>" and names[29] == "<laugh>" and names[44] == "<lang_tr>"
    assert names[49] == "<voice>" and names[50] == "<voice_end>" and names[51] == "<accent_keep>"
    assert names[52] == "<reserved_52>" and names[63] == "<reserved_63>"
    assert len(names) == 64
    # ByteTokenizer geometry is unchanged by the new names
    assert byte_tok.vocab_size == 320
    assert byte_tok.decode(byte_tok.encode("merhaba dünya")) == "merhaba dünya"


def test_trained_bpe_pins_tag_ids(tmp_path) -> None:
    from omni.text.tokenizer import train_bpe

    tok = train_bpe(
        ["hello world", "angry voices carry", "merhaba"] * 30,
        vocab_size=400,
        out_path=tmp_path / "tok.json",
    )
    assert tok.encode("<angry>")[0] == EMOTION_CLASSES["angry"]
    assert tok.encode("<lang_tr>")[0] == LANG_TAGS["tr"]
    assert tok.encode("<laugh> ha")[0] == PARALING_TAGS["laugh"]


def test_turn_prefix() -> None:
    assert turn_prefix() == []
    assert turn_prefix(lang="tr") == [LANG_TAGS["tr"]]
    assert turn_prefix(perceived="angry") == [EMO_PCV, EMOTION_CLASSES["angry"]]
    full = turn_prefix(lang="en", perceived="angry", response_style="calm", intensity="hi")
    assert full == [
        LANG_TAGS["en"],
        EMO_PCV, EMOTION_CLASSES["angry"],
        EMO_RSP, EMOTION_CLASSES["calm"],
        streams.INTENSITY_HI,
    ]
    # intensity only applies to a response style: passing it alone is an error
    # (silently dropping it would hide a caller bug)
    with pytest.raises(ValueError, match="response style"):
        turn_prefix(intensity="hi")
    with pytest.raises(KeyError):
        turn_prefix(perceived="melancholic")


def test_sine_tts_style_is_audible() -> None:
    from omni.data.synthesize import SineTTS

    t = SineTTS()
    a, _ = t.synth("hello world", "alto")
    b, _ = t.synth("hello world", "alto", style="angry")
    c, _ = t.synth("hello world", "alto", style="angry")
    assert a.shape == b.shape and not torch.equal(a, b)
    assert torch.equal(b, c), "style shift must stay deterministic"


def test_fake_dialogues_carry_labels() -> None:
    from omni.data.synthesize import fake_dialogues

    ds = list(fake_dialogues(6, seed=0))
    tagged = [t for d in ds for t in d["turns"] if "user_emotion" in t]
    untagged = [t for d in ds for t in d["turns"] if "user_emotion" not in t]
    assert tagged and untagged, "both labeled and unlabeled dialogues must exist"
    for t in tagged:
        assert t["user_emotion"] in EMOTION_CLASSES
        assert t["response_style"] in EMOTION_CLASSES
        assert t["lang"] in LANG_TAGS
    assert ds == list(__import__("omni.data.synthesize", fromlist=["fake_dialogues"]).fake_dialogues(6, seed=0))


def test_prepare_s2s_threads_emotion_tags(tmp_path, test_cfg, byte_tok) -> None:
    from omni.audio.codec import FakeCodec
    from omni.data.dataset import ShardDataset
    from omni.data.prepare import prepare_s2s
    from omni.data.synthesize import SineTTS, fake_dialogues

    prepare_s2s(
        tmp_path / "shards",
        dialogues=list(fake_dialogues(6, seed=0)),
        tts=SineTTS(),
        codec=FakeCodec(
            n_codebooks=test_cfg.model.n_codebooks,
            codec_vocab=test_cfg.model.audio_codec_vocab,
        ),
        tokenizer=byte_tok,
        cfg=test_cfg,
        max_samples=6,
        seed=0,
    )
    ds = ShardDataset(tmp_path / "shards")
    assert len(ds) > 0
    rows = [ds[i].grid[0] for i in range(len(ds))]
    has_tags = [r for r in rows if (r == EMO_PCV).any()]
    assert has_tags, "labeled dialogues must produce <emo_pcv> in the monologue"
    for r in has_tags:
        cols = (r == EMO_PCV).nonzero().flatten()
        for c in cols:
            pcv = int(r[c + 1])
            assert pcv in list(EMOTION_CLASSES.values())[:N_PERCEIVED_EMOTIONS] or (
                pcv in EMOTION_CLASSES.values()
            )
        # every tagged row carries a response style AND a language tag
        # (fake_dialogues rotates languages since the 2026-07 multilingual pass)
        row_langs = {int(v) for v in r.tolist() if int(v) in LANG_TAGS.values()}
        assert (r == EMO_RSP).any() and row_langs
    all_langs = {
        int(v) for r in has_tags for v in r.tolist() if int(v) in LANG_TAGS.values()
    }
    assert len(all_langs) >= 2, "labeled dialogues must span multiple languages"
    plain = [r for r in rows if not (r == EMO_PCV).any()]
    assert plain, "unlabeled dialogues stay tag-free (v1 behavior preserved)"


def test_generator_prefix_forcing(test_cfg, byte_tok, fake_codec) -> None:
    from omni.infer.generate import OmniGenerator
    from omni.model.omni import OmniModel

    torch.manual_seed(0)
    model = OmniModel(test_cfg.model)
    model.init_weights()
    model.eval()
    gen = OmniGenerator(model, test_cfg, device="cpu", tokenizer=byte_tok)
    prefix = turn_prefix(lang="en", response_style="calm")
    r = gen.tts("hi", fake_codec, prefix_ids=prefix, max_frames=8, seed=1)
    ids = list(r.text_ids)
    pos = ids.index(LANG_TAGS["en"])
    assert ids[pos : pos + 3] == [LANG_TAGS["en"], EMO_RSP, EMOTION_CLASSES["calm"]]
    wav = torch.sin(torch.linspace(0, 400, fake_codec.samples_per_frame * 6))
    r2 = gen.s2s(wav, fake_codec, prefix_ids=prefix, max_frames=8, seed=1)
    ids2 = list(r2.text_ids)
    assert LANG_TAGS["en"] in ids2 and EMOTION_CLASSES["calm"] in ids2


def test_emotion_label_normalization() -> None:
    from omni.data.prepare import _emotion_label

    assert _emotion_label({"emotion": "Anger"}, "emotion") == "angry"
    assert _emotion_label({"emotion": "hap"}, "emotion") == "happy"
    assert _emotion_label({"emotion": "confused"}, "emotion") is None
    assert _emotion_label({"emotion": "angry"}, None) is None
    assert _emotion_label({}, "emotion") is None
