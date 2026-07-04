"""Configuration for the omni speech-to-speech model.

All configs are plain dataclasses, loadable from YAML with dotted-path overrides.
Presets: tiny (CPU smoke test), small (1 GPU), base (8 GPU). See docs/DESIGN.md §3.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Llama-style decoder over 1 text stream + n_codebooks audio streams."""

    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    n_kv_heads: int = 2
    d_ff: int = 768
    max_frames: int = 512  # context length in 12.5 Hz frame steps (delayed grid)
    rope_theta: float = 500_000.0
    text_vocab_size: int = 32_768  # includes the 64 reserved special ids (0..63)
    n_codebooks: int = 8
    audio_codec_vocab: int = 2048  # real codec codes per codebook (specials appended)
    dropout: float = 0.0
    norm_eps: float = 1e-5
    init_std: float = 0.02
    tie_text_head: bool = True
    grad_checkpoint: bool = False
    # --- v2 extensions (defaults preserve v1 behavior) ---
    # "stagger": codebook k delayed k+1 frames (MusicGen-style, parallel heads).
    # "flat": all codebooks delayed 1 frame; requires the depth transformer, which
    # predicts the codebooks of one frame sequentially (CSM/Moshi-style).
    # "lead": semantic codebook delayed 1, acoustics 2 (Moshi's ablated winner —
    # acoustics trail the semantic stream by one frame); requires the depth
    # transformer like "flat".
    audio_delay_mode: str = "stagger"
    use_depth: bool = False
    depth_d_model: int = 256
    depth_n_layers: int = 2
    depth_n_heads: int = 4
    # Fraction of frame positions whose depth-transformer loss is computed each
    # step (CSM "compute amortization": 1/16 is reported lossless). 1.0 trains
    # on every frame; only affects training-mode forward, never eval/decode.
    depth_loss_ratio: float = 1.0
    # Full duplex: streams = 1 text + n_codebooks assistant audio (predicted)
    # + n_codebooks user audio (input-only), time-synchronized, no turn-taking.
    duplex: bool = False
    # --- v6 extensions (docs/DESIGN_V6_PRETRAINED_BACKBONE.md) ---
    # backbone_id: HF causal-LM id (e.g. "Qwen/Qwen3-1.7B") -> HFOmniModel with a
    # pretrained temporal backbone; None keeps the from-scratch OmniModel. When
    # set, d_model/n_layers/n_heads/n_kv_heads/d_ff/rope_theta/tie_text_head are
    # taken from the HF config and text_vocab_size is derived as
    # 64 + backbone embedding rows (both overwritten on the model's cfg).
    backbone_id: str | None = None
    backbone_dtype: str = "bf16"  # "bf16" | "fp32" backbone parameter dtype
    freeze_backbone: bool = True  # stage-1 modality alignment; False = full finetune
    lora_rank: int = 0  # >0: LoRA on the backbone (optional dep `peft`); 0 = off
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    @property
    def audio_vocab_size(self) -> int:
        """Codec codes + AUDIO_PAD, AUDIO_BOS, AUDIO_EOS."""
        return self.audio_codec_vocab + 3

    @property
    def n_streams(self) -> int:
        return 1 + (2 * self.n_codebooks if self.duplex else self.n_codebooks)

    @property
    def max_delay(self) -> int:
        """Largest per-stream delay (stagger: n_codebooks; flat: 1; lead: 2)."""
        if self.audio_delay_mode == "stagger":
            return self.n_codebooks
        if self.audio_delay_mode == "lead":
            return 1 if self.n_codebooks == 1 else 2
        return 1

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.audio_delay_mode in ("stagger", "flat", "lead"), self.audio_delay_mode
        # Locked together: flat/lead delays make several codebooks of one frame
        # land on the same step, which is only sound with sequential intra-frame
        # prediction (the depth transformer).
        assert self.use_depth == (self.audio_delay_mode in ("flat", "lead")), (
            "use_depth=True requires audio_delay_mode='flat' or 'lead' and vice versa"
        )
        if self.use_depth:
            assert self.depth_d_model % self.depth_n_heads == 0
        assert 0.0 < self.depth_loss_ratio <= 1.0, (
            "depth_loss_ratio must be in (0, 1]"
        )
        assert self.backbone_dtype in ("bf16", "fp32"), self.backbone_dtype
        assert self.lora_rank >= 0, "lora_rank must be >= 0"
        if self.lora_rank > 0:
            assert self.backbone_id is not None, (
                "lora_rank > 0 requires a pretrained backbone (model.backbone_id)"
            )


@dataclass
class DataConfig:
    shard_dir: str = "data/shards"
    batch_size: int = 8
    num_workers: int = 4
    # Longest *undelayed* sample; delayed length must fit max_frames.
    max_sample_frames: int = 504
    bucket_by_length: bool = True
    seed: int = 0


@dataclass
class TrainConfig:
    lr: float = 3e-4
    # LR for UNFROZEN pretrained-backbone parameters (incl. LoRA); new modules
    # (audio embs, special emb/head, channel emb, depth) always use `lr`.
    # Ignored while model.freeze_backbone is True. DESIGN_V6 §5.
    backbone_lr: float = 2e-5
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200
    max_steps: int = 1000
    schedule: str = "cosine"  # "cosine" | "wsd" | "constant"
    min_lr_ratio: float = 0.1
    accum_steps: int = 1
    # "auto": single process -> none; multi-GPU -> ddp below fsdp_threshold_params else fsdp2.
    strategy: str = "auto"  # "auto" | "none" | "ddp" | "fsdp2"
    fsdp_threshold_params: int = 300_000_000
    precision: str = "bf16"  # "bf16" | "fp32"
    compile: bool = False
    ckpt_dir: str = "checkpoints/run"
    save_every: int = 500
    save_keep: int = 3  # keep the newest K step_* checkpoint dirs (0 = keep all)
    log_every: int = 10
    eval_every: int = 250
    eval_steps: int = 20
    resume: bool = True  # resume from latest checkpoint in ckpt_dir if present
    # Weights & Biases (rank0 only; lazy import — needs `pip install 'omni[wandb]'`).
    # The run id is persisted at <ckpt_dir>/wandb_run_id.txt, so a resumed
    # training continues the SAME wandb run.
    wandb: bool = False
    wandb_project: str = "omni"
    wandb_run_name: str | None = None  # None -> wandb picks a name
    wandb_entity: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"
    seed: int = 0
    # Per-head loss weights (risk #12 in DESIGN.md): text head and audio heads.
    text_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    semantic_loss_weight: float = 1.0  # extra multiplier on codebook 0


@dataclass
class SamplingConfig:
    """Moshi-derived defaults; tune per model size (tiny/small may prefer near-greedy)."""

    text_temperature: float = 0.7
    text_top_k: int = 25
    audio_temperature: float = 0.8
    audio_top_k: int = 250
    max_frames: int = 375  # 30 s at 12.5 Hz


@dataclass
class OmniConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    preset: str = "tiny"
    tokenizer_path: str = "data/tokenizer/omni_bpe.json"
    codec: str = "mimi"  # "mimi" | "fake"
    codec_model_id: str = "kyutai/mimi"


def _preset_tiny() -> OmniConfig:
    cfg = OmniConfig(preset="tiny")
    return cfg


def _preset_small() -> OmniConfig:
    """1-GPU ablation preset; mirrors the quality path (32 cb, flat+depth) at ~0.3B."""
    cfg = OmniConfig(preset="small")
    cfg.model = ModelConfig(
        d_model=1024, n_layers=16, n_heads=16, n_kv_heads=4, d_ff=2816,
        max_frames=1024, n_codebooks=32,
        audio_delay_mode="flat", use_depth=True, depth_loss_ratio=0.0625,
        depth_d_model=512, depth_n_layers=2, depth_n_heads=8,
        text_vocab_size=48_000,  # multilingual 48k BPE (DESIGN_V4 §3)
    )
    cfg.data.max_sample_frames = 1016
    cfg.train.strategy = "auto"
    return cfg


def _preset_base() -> OmniConfig:
    """Legacy 8-codebook stagger preset (DESIGN.md v1)."""
    cfg = OmniConfig(preset="base")
    cfg.model = ModelConfig(
        d_model=1536, n_layers=26, n_heads=12, n_kv_heads=4, d_ff=4096,
        max_frames=2048, grad_checkpoint=True,
    )
    cfg.data.max_sample_frames = 2040
    return cfg


def _preset_quality() -> OmniConfig:
    """Primary 8-GPU preset per docs/DESIGN_V3_AUDIO.md: Mimi at 32 codebooks
    (4.4 kbps; the v1 8-codebook fidelity cap was the main quality bound) with
    the CSM-style depth transformer, ~1.0B params total."""
    cfg = OmniConfig(preset="quality")
    cfg.model = ModelConfig(
        d_model=1536, n_layers=26, n_heads=12, n_kv_heads=4, d_ff=4096,
        max_frames=2048, grad_checkpoint=True, n_codebooks=32,
        audio_delay_mode="flat", use_depth=True, depth_loss_ratio=0.0625,
        depth_d_model=1024, depth_n_layers=4, depth_n_heads=8,
        text_vocab_size=48_000,  # multilingual 48k BPE (DESIGN_V4 §3)
    )
    cfg.data.max_sample_frames = 2040
    cfg.train.semantic_loss_weight = 100.0  # Moshi setting (see DESIGN_V3 §4)
    return cfg


def _preset_hf(preset: str, backbone_id: str) -> OmniConfig:
    """Shared shape of the v6 pretrained-backbone presets (DESIGN_V6 §4):
    Qwen3/Llama/Gemma temporal backbone + the V3 audio side (Mimi at 32
    codebooks, flat delays, depth transformer). Backbone-derived fields
    (d_model, layer/head counts, text_vocab_size) keep their placeholder
    defaults here and are overwritten from the HF config at model build."""
    cfg = OmniConfig(preset=preset)
    cfg.model = ModelConfig(
        max_frames=2048, grad_checkpoint=True, n_codebooks=32,
        audio_delay_mode="flat", use_depth=True, depth_loss_ratio=0.0625,
        depth_d_model=1024, depth_n_layers=4, depth_n_heads=8,
        backbone_id=backbone_id,
    )
    cfg.data.max_sample_frames = 2040
    cfg.train.semantic_loss_weight = 100.0  # Moshi setting (DESIGN_V3 §4)
    return cfg


PRESETS = {
    "tiny": _preset_tiny,
    "small": _preset_small,
    "base": _preset_base,
    "quality": _preset_quality,
    # v6 pretrained-backbone presets; the model downloads on first build.
    "qwen3-1.7b": lambda: _preset_hf("qwen3-1.7b", "Qwen/Qwen3-1.7B-Base"),
    "qwen3-8b": lambda: _preset_hf("qwen3-8b", "Qwen/Qwen3-8B-Base"),
    "llama32-3b": lambda: _preset_hf("llama32-3b", "meta-llama/Llama-3.2-3B"),
    "gemma3-4b": lambda: _preset_hf("gemma3-4b", "google/gemma-3-4b-pt"),
}


def _to_dict(cfg: OmniConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def _coerce(cur: Any, v: Any) -> Any:
    """Coerce a YAML-parsed value to the type of the existing field.

    PyYAML reads '1e-3' as a string (YAML floats need '1.0e-3'), so numeric
    fields coerce explicitly. Malformed values raise instead of silently
    degrading (a typo'd boolean must not disable a feature).
    """
    if isinstance(cur, tuple) and isinstance(v, (list, tuple)):
        if not cur or (cur and isinstance(cur[0], str)):
            return tuple(v)  # variable-length tuple (e.g. wandb_tags)
        if len(v) != len(cur):
            raise ValueError(
                f"expected {len(cur)} elements for this tuple field, got {len(v)}: {v!r}"
            )
        return tuple(_coerce(c, x) for c, x in zip(cur, v))
    if isinstance(cur, bool):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off"):
                return False
            raise ValueError(f"invalid boolean value {v!r} (use true/false)")
        return bool(v)
    if isinstance(cur, float) and isinstance(v, (int, str)):
        return float(v)
    if isinstance(cur, int) and isinstance(v, (float, str)):
        iv = int(float(v))
        assert iv == float(v), f"non-integer value {v!r} for int field"
        return iv
    return v


def _apply_dict(obj: Any, d: dict[str, Any], path: str = "") -> None:
    for k, v in d.items():
        if not hasattr(obj, k):
            raise KeyError(f"unknown config key: {path}{k}")
        cur = getattr(obj, k)
        if dataclasses.is_dataclass(cur) and isinstance(v, dict):
            _apply_dict(cur, v, path=f"{path}{k}.")
        else:
            setattr(obj, k, _coerce(cur, v))


def _apply_override(cfg: OmniConfig, dotted: str) -> None:
    """Apply one 'a.b.c=value' override; value parsed as YAML."""
    key, _, raw = dotted.partition("=")
    if not _ or key == "":
        raise ValueError(f"override must look like a.b=c, got: {dotted!r}")
    value = yaml.safe_load(raw)
    obj: Any = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    leaf = parts[-1]
    if not hasattr(obj, leaf):
        raise KeyError(f"unknown config key: {key}")
    setattr(obj, leaf, _coerce(getattr(obj, leaf), value))


def load_config(
    preset_or_path: str = "tiny",
    overrides: list[str] | None = None,
) -> OmniConfig:
    """Build a config from a preset name or a YAML file, then apply overrides.

    YAML files may contain a top-level `preset:` key to start from a preset.
    """
    if preset_or_path in PRESETS:
        cfg = PRESETS[preset_or_path]()
    else:
        path = Path(preset_or_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{preset_or_path!r} is neither a preset {list(PRESETS)} nor a YAML file"
            )
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        cfg = PRESETS[d.pop("preset", "tiny")]()
        _apply_dict(cfg, d)
    for ov in overrides or []:
        _apply_override(cfg, ov)
    # YAML dicts and overrides mutate the constructed dataclasses with setattr,
    # which bypasses __post_init__; rebuild ModelConfig so its invariants
    # (flat<->use_depth lock, head divisibility, ...) are re-checked and fail
    # with their intended messages.
    cfg.model = ModelConfig(**dataclasses.asdict(cfg.model))
    # Keep dependent fields coherent.
    assert cfg.data.max_sample_frames + cfg.model.max_delay <= cfg.model.max_frames, (
        "data.max_sample_frames + model.max_delay must fit model.max_frames"
    )
    return cfg


def save_config(cfg: OmniConfig, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(_to_dict(cfg), f, sort_keys=False)
