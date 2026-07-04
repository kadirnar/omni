# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Speech-to-speech LLM: one decoder-only transformer over parallel token streams (1 text stream + N Mimi codec audio streams) on a 12.5 Hz frame grid. Two backbone paths share every other module: the from-scratch `OmniModel` (tiny/CPU tests, ablations) and the production `HFOmniModel` (v6) wrapping a pretrained HF causal LM (Qwen3/Llama/Gemma, presets `qwen3-1.7b|qwen3-8b|llama32-3b|gemma3-4b`) selected by `model.backbone_id`. This Mac is a CPU/MPS-only dev box (no CUDA); everything must run offline with `FakeCodec` (HF-path tests inject a tiny in-process random Llama, never downloading) — real training happens on an 8-GPU node.

## Commands

```bash
# setup
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python -e ".[dev]"

# tests (all CPU, offline, no downloads)
.venv/bin/pytest                          # full suite
.venv/bin/pytest tests/test_model.py      # one file
.venv/bin/pytest tests/test_model.py -k name_substring   # one test

# offline end-to-end smoke (fake data -> train -> generate)
.venv/bin/python scripts/prepare_data.py fake --n 256 --out data/shards/fake \
    --preset tiny model.n_codebooks=2 model.text_vocab_size=320
.venv/bin/python scripts/train.py --preset tiny --data data/shards/fake \
    model.n_codebooks=2 model.text_vocab_size=320 data.num_workers=0 \
    train.max_steps=200 --export checkpoints/tiny
.venv/bin/python scripts/chat.py --task tts --text "hello" --out hello.wav \
    --ckpt checkpoints/tiny --codec fake --tokenizer byte

# streaming test console (needs ".[serve]")
.venv/bin/python scripts/serve.py --preset tiny    # -> http://127.0.0.1:7860
```

Config system: `--preset tiny|small|quality|base` plus `section.key=value` overrides on any script (e.g. `model.n_codebooks=2 train.max_steps=20`). `--data DIR:WEIGHT` mixes shard dirs for training.

## Frozen contracts

`src/omni/config.py`, `src/omni/streams.py`, `src/omni/grids.py`, and `pyproject.toml` are FROZEN interface contracts — read them first, do not edit them (surface concerns instead). `docs/INTERFACES.md` is the binding API contract between modules; changes elsewhere must conform to it.

## Architecture

The core idea: every task (ASR, TTS, speech continuation, text LM, spoken dialogue s2s, full-duplex) is the **same token grid** with different columns loss-masked. A grid is `[S, T]` where `S = 1 + n_codebooks` streams: row 0 is text (with special/control tokens like `<s2s>`, `<lang_en>`, `<emo_pcv>`), rows 1..N are Mimi codebook tokens, one column = one 80 ms frame. A per-position `loss_mask` and per-column `channel` (user/assistant) complete a sample.

**Delay invariant:** grids on disk and in `Sample` are UNDELAYED; the model and generator consume DELAYED grids (`streams.apply_delay`, MusicGen-style per-codebook delay). The delay is applied at batch time so changing it never invalidates prepared data. Logits at position `p` predict grid position `p+1`; `loss_mask[s, p+1]` gates that target.

Data flow: `data/prepare.py` (dialogues/HF datasets + TTS + codec → binary shards) → `data/dataset.py` (shards → delayed batches) → `model/` (`build_model(cfg.model)` dispatches: `omni.py` from-scratch or `hf_omni.py` pretrained backbone; both sum per-stream embeddings and share the depth transformer / parallel heads) → `train/loop.py` (`Trainer`, auto DDP/FSDP2 by size, resumable checkpoints, optional wandb) → `infer/generate.py` (`OmniGenerator`: frame-by-frame decode via `model.new_cache()`, voice cloning via reference-audio prefill) → `serve/` (FastAPI streaming console).

**v6 backbone path** (`docs/DESIGN_V6_PRETRAINED_BACKBONE.md`): `HFOmniModel` drives the HF LM through `inputs_embeds`; text ids offset by +64 (`HFTextTokenizer`) so specials 0..63 keep their frozen layout (fresh 64-row `special_emb`/`special_head` concatenated onto the pretrained `lm_head` logits). `freeze_backbone` (default) trains only the new modules; `train.backbone_lr` applies when unfrozen; `model.lora_rank>0` needs `omni[lora]` (peft). Exports write `adapters.safetensors` (backbone referenced by id, not copied); `load_model(dir)` dispatches on config.yaml. Structural `ModelConfig` fields are derived from the HF config at build — always adopt them via `cfg.model = model.cfg`.

Module map (each file's API pinned in `docs/INTERFACES.md`):
- `audio/codec.py` — `AudioCodec` ABC, `MimiCodec` (real, downloads), `FakeCodec` (deterministic offline stand-in), `build_codec`, wav IO
- `text/tokenizer.py` — `TextTokenizer` (48k BPE), `ByteTokenizer` (zero-artifact, vocab 320, used in tests), `HFTextTokenizer` (backbone tokenizer +64, `"hf:<model_id>"` in CLIs); ids 0..63 are reserved specials, real text starts at 64
- `streams.py` / `grids.py` — special tokens, delay pattern, grid builders, `turn_prefix`
- `model/hf_omni.py` — `HFOmniModel` + `HFCache` (v6); `model/__init__.py` — `build_model`/`load_model` factory
- `infer/duplex.py` — full-duplex (mic and model on the same clock)
- `optim/perf.py` — quantization/compile paths (torchao optional; compiles HF decoder layers on the backbone path)

## Coding rules (from docs/INTERFACES.md)

- No network at import time, ever; nothing downloads unless the user explicitly invoked that path (`--codec mimi`, HF dataset prep, real TTS backends).
- Optional deps (`kokoro`, `torchao`, `wandb`, TTS backends) are imported lazily inside the function that needs them with a clear `ImportError`. No new dependencies.
- Determinism: explicit `seed` args; local `torch.Generator` for sampling, `random.Random(seed)` for python shuffles.
- No einops; plain torch. Keep CPU/MPS-safe (guard bf16/fused/CUDA paths behind device checks).
- Type hints everywhere; docstrings state tensor shapes like `[B, S, T]`.

## Docs

`docs/DESIGN.md` (architecture + stage A→D training recipe; dataset ids there are unverified), `docs/DESIGN_V3_AUDIO.md` (32-codebook + depth transformer), `docs/DESIGN_V4_EMOTION_I18N.md` (emotion/multilingual control tokens), `docs/DESIGN_V5_VOICE.md` (voice cloning), `docs/research/` (raw research reports). Remaining GPU-dependent work is tracked in the implementation queues of V3/V4.
