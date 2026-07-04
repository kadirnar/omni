"""Model package: transformer layers, the from-scratch multistream OmniModel,
and the v6 pretrained-backbone HFOmniModel, plus the factory/loader that
dispatches between them on ``cfg.backbone_id``."""

import dataclasses
from pathlib import Path

import torch
import yaml

from ..config import ModelConfig
from .hf_omni import HFCache, HFOmniModel
from .layers import (
    MLP,
    Attention,
    Block,
    DepthTransformer,
    KVCache,
    LayerKVCache,
    RMSNorm,
    apply_rope,
    precompute_rope,
)
from .omni import ModelOutput, OmniModel, multistream_loss


def build_model(cfg: ModelConfig, backbone=None) -> "OmniModel | HFOmniModel":
    """cfg.backbone_id set (or a backbone injected) -> HFOmniModel; else the
    from-scratch OmniModel. Downloads happen only on the HF path with no
    injected backbone (an explicit user choice, like ``--codec mimi``)."""
    if cfg.backbone_id is not None or backbone is not None:
        return HFOmniModel(cfg, backbone=backbone)
    return OmniModel(cfg)


def load_model(
    save_dir: "str | Path",
    backbone=None,
    map_location: "str | torch.device" = "cpu",
) -> "OmniModel | HFOmniModel":
    """Load a checkpoint dir written by either model class's save_pretrained
    (or Trainer.export_model), dispatching on config.yaml's backbone_id."""
    p = Path(save_dir)
    with open(p / "config.yaml") as f:
        d = yaml.safe_load(f) or {}
    if d.get("backbone_id") or backbone is not None:
        return HFOmniModel.from_pretrained(p, backbone=backbone, map_location=map_location)
    return OmniModel.from_pretrained(p, map_location=map_location)


#: ModelConfig fields that determine parameter shapes / optimizer-group
#: structure. Checkpoint resume and warm starts compare these and fail with a
#: clear message instead of a raw state_dict shape error.
STRUCTURAL_KEYS = (
    "d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff",
    "text_vocab_size", "n_codebooks", "audio_codec_vocab",
    "audio_delay_mode", "use_depth", "depth_d_model", "depth_n_layers",
    "depth_n_heads", "duplex", "backbone_id", "tie_text_head",
)


def structural_diff(saved_model_cfg: dict, cfg: ModelConfig) -> list[str]:
    """Human-readable list of structural mismatches between a saved model
    config (dict form) and the live ModelConfig; empty when compatible."""
    current = dataclasses.asdict(cfg)
    return [
        f"{k}: saved={saved_model_cfg[k]!r} vs current={current[k]!r}"
        for k in STRUCTURAL_KEYS
        if k in saved_model_cfg and saved_model_cfg[k] != current[k]
    ]


def load_weights(model: "OmniModel | HFOmniModel", save_dir: "str | Path") -> None:
    """Load exported weights into an ALREADY-BUILT model, in place.

    This is the stage-transition warm start (DESIGN_V6 §5): build the model
    for the NEW stage config (e.g. ``freeze_backbone=false`` or a fresh
    ``lora_rank``), then pull the previous stage's exported weights into it —
    optimizer and schedule start fresh, unlike checkpoint resume.

    From-scratch exports (``model.safetensors``) load strictly. Backbone
    exports (``adapters.safetensors``) load the adapter set; backbone weights
    stay as loaded from the hub, and LoRA keys that exist only in the new
    stage (or only in the export) are tolerated. Structural config mismatches
    raise before any tensor is touched.
    """
    p = Path(save_dir)
    with open(p / "config.yaml") as f:
        saved = yaml.safe_load(f) or {}
    diffs = structural_diff(saved, model.cfg)
    if diffs:
        raise ValueError(
            f"--init-from {p}: exported model is structurally incompatible: "
            + "; ".join(diffs)
        )

    from safetensors.torch import load_file, load_model as st_load_model

    if (p / "adapters.safetensors").exists():
        sd = load_file(str(p / "adapters.safetensors"))
        target_dtype = next(model.backbone.parameters()).dtype
        sd = {k: v.to(target_dtype) for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        stale_lora = [k for k in unexpected if "lora_" in k]
        if len(stale_lora) != len(unexpected):
            raise ValueError(
                f"--init-from {p}: unexpected non-LoRA keys "
                f"{[k for k in unexpected if 'lora_' not in k][:5]}"
            )
        bad = [
            k for k in missing
            if type(model)._is_adapter_key(k) and "lora_" not in k
        ]
        if bad:
            raise ValueError(f"--init-from {p}: export is missing adapter keys {bad[:5]}")
    else:
        st_load_model(model, str(p / "model.safetensors"))


__all__ = [
    "MLP",
    "Attention",
    "Block",
    "DepthTransformer",
    "HFCache",
    "HFOmniModel",
    "KVCache",
    "LayerKVCache",
    "RMSNorm",
    "apply_rope",
    "precompute_rope",
    "ModelOutput",
    "OmniModel",
    "STRUCTURAL_KEYS",
    "build_model",
    "load_model",
    "load_weights",
    "multistream_loss",
    "structural_diff",
]
