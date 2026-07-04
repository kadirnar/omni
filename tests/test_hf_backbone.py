"""Contract tests for the v6 pretrained-backbone path (DESIGN_V6).

Everything is CPU-only and offline: the "pretrained" backbone is a tiny
randomly-initialized ``LlamaForCausalLM`` built in-process from a config
(never downloaded), with vocab_size=256 so ``text_vocab_size = 64 + 256 = 320``
matches the ByteTokenizer and the existing fake data pipeline exactly.
"""

from __future__ import annotations

import json

import pytest
import torch

from omni.config import PRESETS, load_config
from omni.model import HFOmniModel, OmniModel, build_model, load_model
from omni.text.tokenizer import HFTextTokenizer

BB_VOCAB = 256


def tiny_backbone(seed: int = 7):
    """Tiny random Llama causal LM, constructed locally (zero network)."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    return LlamaForCausalLM(
        LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=BB_VOCAB,
            max_position_embeddings=512,
            tie_word_embeddings=False,
        )
    )


def hf_cfg(*extra: str):
    """Tiny OmniConfig on the backbone path (2 codebooks, stagger by default)."""
    return load_config(
        "tiny",
        [
            "model.n_codebooks=2",
            "model.text_vocab_size=320",
            "model.backbone_id=test/tiny-llama",  # never resolved: backbone injected
            "model.backbone_dtype=fp32",
            "data.max_sample_frames=120",
            "data.batch_size=2",
            "data.num_workers=0",
            *extra,
        ],
    )


DEPTH_OVERRIDES = (
    "model.audio_delay_mode=flat",
    "model.use_depth=true",
    "model.depth_d_model=32",
    "model.depth_n_layers=1",
    "model.depth_n_heads=2",
)


@pytest.fixture(params=["stagger", "depth"])
def cfg_and_model(request):
    """(cfg, HFOmniModel) in both audio output modes; cfg.model is the model's
    derived cfg, mirroring the checkpoint-loading convention."""
    cfg = hf_cfg(*(DEPTH_OVERRIDES if request.param == "depth" else ()))
    torch.manual_seed(0)
    model = build_model(cfg.model, backbone=tiny_backbone())
    cfg.model = model.cfg
    return cfg, model


def _batch(cfg, B: int = 2, T: int = 12, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    S = cfg.model.n_streams
    grid = torch.randint(0, cfg.model.text_vocab_size, (B, S, T), generator=g)
    grid[:, 1:] = torch.randint(0, cfg.model.audio_vocab_size, (B, S - 1, T), generator=g)
    channel = torch.randint(0, 2, (B, T), generator=g)
    mask = torch.rand(B, S, T, generator=g) > 0.5
    return grid, channel, mask


# ---------------------------------------------------------------------- config
def test_backbone_presets_build():
    for name in ("qwen3-1.7b", "qwen3-8b", "llama32-3b", "gemma3-4b"):
        cfg = PRESETS[name]()
        assert cfg.model.backbone_id
        assert cfg.model.use_depth and cfg.model.audio_delay_mode == "flat"
        assert cfg.model.freeze_backbone


def test_lora_requires_backbone():
    with pytest.raises(AssertionError, match="lora_rank"):
        load_config("tiny", ["model.lora_rank=4"])


# ------------------------------------------------------------------- tokenizer
class _FakeHFTok:
    """Duck-typed stand-in for a transformers tokenizer: whitespace words
    hashed into [0, BB_VOCAB)."""

    model_id = "test/tiny-llama"

    def __len__(self) -> int:
        return BB_VOCAB

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False, "omni must never request specials"
        return [sum(map(ord, w)) % BB_VOCAB for w in text.split()]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        return " ".join(f"tok{i}" for i in ids)


def test_hf_tokenizer_offset_and_specials():
    tok = HFTextTokenizer(_FakeHFTok())
    assert tok.vocab_size == 64 + BB_VOCAB
    ids = tok.encode("hello omni world")
    assert len(ids) == 3 and all(i >= 64 for i in ids)
    # round-trip through the fake tok's id-space
    assert tok.decode(ids) == " ".join(f"tok{i - 64}" for i in ids)
    # specials are skipped by default, rendered on request, never sent to hf
    from omni.streams import BOS, END_OF_TURN

    mixed = [BOS, ids[0], END_OF_TURN, ids[1]]
    assert tok.decode(mixed) == f"tok{ids[0] - 64} tok{ids[1] - 64}"
    rendered = tok.decode(mixed, skip_specials=False)
    assert rendered.startswith("<bos>") and "<end_of_turn>" in rendered
    # out-of-range ids are always dropped
    assert tok.decode([tok.vocab_size + 5]) == ""


def test_hf_tokenizer_via_build(monkeypatch):
    from omni.text import tokenizer as tok_mod

    monkeypatch.setattr(
        tok_mod.HFTextTokenizer,
        "from_pretrained",
        classmethod(lambda cls, mid: cls(_FakeHFTok(), model_id=mid)),
    )
    t = tok_mod.build_tokenizer("hf:test/tiny-llama")
    assert isinstance(t, tok_mod.HFTextTokenizer)
    assert t.model_id == "test/tiny-llama"


# ----------------------------------------------------------------------- model
def test_factory_dispatch():
    plain = load_config("tiny", ["model.n_codebooks=2", "model.text_vocab_size=320"])
    assert isinstance(build_model(plain.model), OmniModel)
    m = build_model(hf_cfg().model, backbone=tiny_backbone())
    assert isinstance(m, HFOmniModel)


def test_derived_cfg_matches_backbone(cfg_and_model):
    cfg, m = cfg_and_model
    assert m.cfg.d_model == 64
    assert m.cfg.text_vocab_size == 64 + BB_VOCAB
    assert m.cfg.n_layers == 2 and m.cfg.n_kv_heads == 2
    assert m.cfg.backbone_id == "test/tiny-llama"


def test_forward_shapes_and_loss(cfg_and_model):
    cfg, m = cfg_and_model
    grid, channel, mask = _batch(cfg)
    out = m(grid, channel)
    B, S, T = grid.shape
    assert out.text_logits.shape == (B, T, cfg.model.text_vocab_size)
    assert out.audio_logits.shape == (B, cfg.model.n_codebooks, T, cfg.model.audio_vocab_size)
    assert out.text_logits.dtype == torch.float32
    total, metrics = m.loss(out, grid, mask)
    assert total.requires_grad and "loss/text" in metrics
    total.backward()


def test_backbone_frozen_adapters_trainable(cfg_and_model):
    cfg, m = cfg_and_model
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.special_emb.parameters())
    grid, channel, mask = _batch(cfg)
    out = m(grid, channel)
    total, _ = m.loss(out, grid, mask)
    total.backward()
    assert all(p.grad is None for p in m.backbone.parameters())
    assert m.special_emb.weight.grad is not None
    assert m.audio_embs[0].weight.grad is not None
    assert m.channel_emb.weight.grad is not None


def test_special_vs_backbone_embedding_routing(cfg_and_model):
    """Text ids < 64 must hit special_emb; ids >= 64 the backbone table."""
    cfg, m = cfg_and_model
    S = cfg.model.n_streams
    apad = cfg.model.audio_codec_vocab  # AUDIO_PAD: embeddings still summed,
    grid = torch.full((1, S, 2), apad, dtype=torch.long)  # constant across cols
    grid[0, 0, 0] = 5  # special id
    grid[0, 0, 1] = 64 + 5  # backbone id 5
    channel = torch.zeros(1, 2, dtype=torch.long)
    h = m.embed(grid, channel)
    sp = m.special_emb.weight[5]
    bb = m.backbone.get_input_embeddings().weight[5]
    base = h[0, 0] - sp  # audio-pad + channel contribution, equal in both cols
    assert torch.allclose(h[0, 0], base + sp, atol=1e-6)
    assert torch.allclose(h[0, 1], base + bb, atol=1e-5)


def test_channel_emb_zero_init(cfg_and_model):
    _, m = cfg_and_model
    assert torch.equal(m.channel_emb.weight, torch.zeros_like(m.channel_emb.weight))


def test_prefill_step_matches_forward(cfg_and_model):
    """KV-cached prefill + one step must reproduce the full-sequence logits."""
    cfg, m = cfg_and_model
    m.eval()
    grid, channel, _ = _batch(cfg, B=1, T=10)
    with torch.no_grad():
        full = m(grid, channel)
        cache = m.new_cache(1, "cpu", torch.float32)
        if cfg.model.use_depth:
            tl_p, h_p = m.prefill_hidden(grid[:, :, :9], channel[:, :9], cache)
            tl_s, h_s = m.step_hidden(grid[0, :, 9][None], channel[0, 9][None], cache)
            assert torch.allclose(tl_s, full.text_logits[:, -1], atol=1e-4)
            assert h_s.shape == (1, cfg.model.d_model)
        else:
            tl_p, al_p = m.prefill(grid[:, :, :9], channel[:, :9], cache)
            tl_s, al_s = m.step(grid[0, :, 9][None], channel[0, 9][None], cache)
            assert torch.allclose(tl_s, full.text_logits[:, -1], atol=1e-4)
            assert torch.allclose(al_s, full.audio_logits[:, :, -1], atol=1e-4)
        assert cache.pos == 10
        assert torch.allclose(tl_p, full.text_logits[:, 8], atol=1e-4)


def test_wrong_decode_api_raises(cfg_and_model):
    cfg, m = cfg_and_model
    if not cfg.model.use_depth:
        pytest.skip("only depth models forbid prefill/step")
    grid, channel, _ = _batch(cfg, B=1, T=4)
    with pytest.raises(RuntimeError, match="depth transformer"):
        m.prefill(grid, channel, m.new_cache(1, "cpu", torch.float32))


def test_param_counts(cfg_and_model):
    _, m = cfg_and_model
    counts = m.param_counts()
    assert 0 < counts["trainable"] < counts["total"]
    assert counts["non_embedding"] < counts["total"]


def test_optim_param_groups_frozen_vs_unfrozen():
    cfg = hf_cfg()
    m = build_model(cfg.model, backbone=tiny_backbone())
    groups = m.optim_param_groups(cfg.train)
    # frozen backbone: only new-module params, all at train.lr, split into a
    # decay group and a no-decay group (embeddings + 1-D params)
    assert all(g["lr"] == cfg.train.lr for g in groups)
    frozen_ids = {id(p) for p in m.backbone.parameters()}
    assert all(id(p) not in frozen_ids for g in groups for p in g["params"])
    no_decay = [g for g in groups if g.get("weight_decay") == 0.0]
    assert no_decay, "embeddings/norm gains must sit in a weight_decay=0 group"
    emb_ids = {id(e.weight) for e in m.audio_embs} | {id(m.special_emb.weight)}
    grouped_no_decay = {id(p) for g in no_decay for p in g["params"]}
    assert emb_ids <= grouped_no_decay

    cfg2 = hf_cfg("model.freeze_backbone=false")
    m2 = build_model(cfg2.model, backbone=tiny_backbone())
    groups2 = m2.optim_param_groups(cfg2.train)
    lrs = {g["lr"] for g in groups2}
    assert lrs == {cfg2.train.lr, cfg2.train.backbone_lr}
    n_all = sum(1 for _ in m2.parameters())
    assert sum(len(g["params"]) for g in groups2) == n_all


# ------------------------------------------------------------------ generation
def test_generator_end_to_end_and_determinism(cfg_and_model):
    from omni.audio.codec import FakeCodec
    from omni.infer.generate import OmniGenerator
    from omni.text.tokenizer import ByteTokenizer

    cfg, m = cfg_and_model
    codec = FakeCodec(n_codebooks=cfg.model.n_codebooks)
    gen = OmniGenerator(m, cfg, device="cpu", tokenizer=ByteTokenizer())

    r = gen.tts("hi omni", codec, seed=0, max_frames=10)
    assert r.frames > 0 and r.text == "hi omni"
    wav = torch.sin(torch.linspace(0, 200, codec.samples_per_frame * 6))
    r2 = gen.s2s(wav, codec, seed=0, max_frames=8)
    assert r2.frames > 0
    r3 = gen.asr(wav, codec, seed=0, max_frames=8)
    assert isinstance(r3.text_ids, list)

    again = gen.tts("hi omni", codec, seed=0, max_frames=10)
    assert again.text_ids == r.text_ids
    assert torch.equal(again.audio_codes, r.audio_codes)


def test_generator_voice_reference(cfg_and_model):
    from omni.audio.codec import FakeCodec
    from omni.infer.generate import OmniGenerator
    from omni.text.tokenizer import ByteTokenizer

    cfg, m = cfg_and_model
    codec = FakeCodec(n_codebooks=cfg.model.n_codebooks)
    gen = OmniGenerator(m, cfg, device="cpu", tokenizer=ByteTokenizer())
    gen.set_voice(torch.randn(codec.sample_rate).clamp(-1, 1), codec)
    r = gen.tts("hi", codec, seed=0, max_frames=8)
    assert r.frames > 0


def test_duplex_generator_smoke():
    from omni.audio.codec import FakeCodec
    from omni.infer.duplex import DuplexGenerator
    from omni.text.tokenizer import ByteTokenizer

    cfg = hf_cfg("model.duplex=true")
    torch.manual_seed(0)
    m = build_model(cfg.model, backbone=tiny_backbone())
    cfg.model = m.cfg
    codec = FakeCodec(n_codebooks=cfg.model.n_codebooks)
    dgen = DuplexGenerator(m, cfg, device="cpu", tokenizer=ByteTokenizer(), seed=0)
    wav = torch.sin(torch.linspace(0, 100, codec.samples_per_frame * 4))
    text, wav_out = dgen.run_file(wav, codec)
    assert wav_out.ndim == 1 and wav_out.numel() > 0


# ------------------------------------------------------------------ save/load
def test_save_load_roundtrip(tmp_path, cfg_and_model):
    cfg, m = cfg_and_model
    m.save_pretrained(tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "adapters.safetensors").exists()
    assert (tmp_path / "ckpt" / "config.yaml").exists()
    # adapters land bit-exact on a DIFFERENT random backbone instance...
    m2 = load_model(tmp_path / "ckpt", backbone=tiny_backbone(seed=99))
    assert isinstance(m2, HFOmniModel)
    assert torch.equal(m2.special_emb.weight, m.special_emb.weight)
    assert torch.equal(m2.audio_embs[0].weight, m.audio_embs[0].weight)
    # ...and no backbone weights leak into the adapter file
    from safetensors.torch import load_file

    keys = load_file(str(tmp_path / "ckpt" / "adapters.safetensors")).keys()
    assert not any(k.startswith("backbone.") for k in keys)
    assert not any(k.startswith("depth.audio_embs.") for k in keys)


def test_full_generation_parity_after_reload(tmp_path, cfg_and_model):
    from omni.audio.codec import FakeCodec
    from omni.infer.generate import OmniGenerator
    from omni.text.tokenizer import ByteTokenizer

    cfg, m = cfg_and_model
    codec = FakeCodec(n_codebooks=cfg.model.n_codebooks)
    r = OmniGenerator(m, cfg, tokenizer=ByteTokenizer()).tts("hey", codec, seed=1, max_frames=8)
    m.save_pretrained(tmp_path / "ckpt")
    m2 = load_model(tmp_path / "ckpt", backbone=m.backbone)  # same backbone weights
    r2 = OmniGenerator(m2, cfg, tokenizer=ByteTokenizer()).tts("hey", codec, seed=1, max_frames=8)
    assert r2.text_ids == r.text_ids
    assert torch.equal(r2.audio_codes, r.audio_codes)


# -------------------------------------------------------------------- training
def test_trainer_freeze_contract_and_export(tmp_path):
    from omni.audio.codec import FakeCodec
    from omni.data.dataset import build_dataloader
    from omni.data.prepare import prepare_s2s
    from omni.data.synthesize import SineTTS, fake_dialogues
    from omni.text.tokenizer import ByteTokenizer
    from omni.train.loop import Trainer

    cfg = hf_cfg(
        "train.max_steps=2", "train.warmup_steps=1", "train.save_every=0",
        "train.log_every=0", "train.eval_every=0", "train.precision=fp32",
        f"train.ckpt_dir={tmp_path / 'ckpt'}",
    )
    shards = tmp_path / "shards"
    prepare_s2s(
        shards, dialogues=list(fake_dialogues(6, seed=0)), tts=SineTTS(),
        codec=FakeCodec(n_codebooks=2), tokenizer=ByteTokenizer(), cfg=cfg,
        max_samples=6,
    )
    torch.manual_seed(0)
    model = build_model(cfg.model, backbone=tiny_backbone())
    cfg.model = model.cfg
    bb_w = model.backbone.get_input_embeddings().weight.clone()
    sp_w = model.special_emb.weight.clone()

    trainer = Trainer(cfg, model, build_dataloader(cfg, [str(shards)]))
    metrics = trainer.fit()
    assert metrics["step"] == 2.0
    assert torch.equal(bb_w, model.backbone.get_input_embeddings().weight)
    assert not torch.equal(sp_w, model.special_emb.weight)

    trainer.export_model(tmp_path / "export")
    m2 = load_model(tmp_path / "export", backbone=tiny_backbone(seed=1))
    assert torch.equal(m2.special_emb.weight, model.special_emb.weight)


# ------------------------------------------------------------------- data meta
def test_shard_meta_records_tokenizer_id(tmp_path):
    from omni.audio.codec import FakeCodec
    from omni.data.prepare import prepare_s2s
    from omni.data.synthesize import SineTTS, fake_dialogues

    cfg = hf_cfg()
    tok = HFTextTokenizer(_FakeHFTok(), model_id="test/tiny-llama")
    prepare_s2s(
        tmp_path / "shards", dialogues=list(fake_dialogues(3, seed=0)),
        tts=SineTTS(), codec=FakeCodec(n_codebooks=2), tokenizer=tok, cfg=cfg,
        max_samples=3,
    )
    meta = json.loads((tmp_path / "shards" / "meta.json").read_text())
    assert meta["tokenizer_id"] == "test/tiny-llama"


def test_shard_meta_omits_tokenizer_id_for_byte(tmp_path):
    from omni.audio.codec import FakeCodec
    from omni.data.prepare import prepare_s2s
    from omni.data.synthesize import SineTTS, fake_dialogues
    from omni.text.tokenizer import ByteTokenizer

    cfg = hf_cfg()
    prepare_s2s(
        tmp_path / "shards", dialogues=list(fake_dialogues(3, seed=0)),
        tts=SineTTS(), codec=FakeCodec(n_codebooks=2), tokenizer=ByteTokenizer(),
        cfg=cfg, max_samples=3,
    )
    meta = json.loads((tmp_path / "shards" / "meta.json").read_text())
    assert "tokenizer_id" not in meta


# ------------------------------------------------------------------------ lora
def test_lora_injection():
    pytest.importorskip("peft", reason="LoRA needs the optional 'peft' package")
    cfg = hf_cfg("model.lora_rank=2")
    m = build_model(cfg.model, backbone=tiny_backbone())
    trainable_bb = [n for n, p in m.backbone.named_parameters() if p.requires_grad]
    assert trainable_bb and all("lora_" in n for n in trainable_bb)
    groups = m.optim_param_groups(cfg.train)
    bb_groups = [g for g in groups if g["lr"] == cfg.train.backbone_lr]
    assert bb_groups, "trainable LoRA params must get a backbone_lr group"
    n_lora = len(trainable_bb)
    assert sum(len(g["params"]) for g in bb_groups) == n_lora
