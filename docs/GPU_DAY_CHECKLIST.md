# GPU-Day Checklist

Ordered smoke sequence for the first session on the 8-GPU node. Distilled from
the 2026-07-03 architecture review (docs/research/ARCHITECTURE_REVIEW_2026-07-03.md)
and the fix pass that followed. Everything below the preflight was verified
offline; the preflight pins what a CPU box cannot.

## 0. Preflight (before any bulk work; downloads Mimi + Qwen3)

```bash
RUN_SLOW=1 .venv/bin/pytest tests/test_gpu_preflight.py -v
```

Pins: Mimi frame count == FakeCodec's `ceil(T/1920)`; `num_quantizers` really
limits codebooks (8 and 32); decode shape; the serve rolling-window encoder
approximates batch codes better than stateless chunks; HFTextTokenizer +64
offset against the real Qwen3 tokenizer; `qwen3-1.7b` builds and steps.

Also run the full offline suite once on the node (env drift):
`.venv/bin/pytest -q`.

## 1. Distributed smoke (fake data, minutes)

```bash
.venv/bin/python scripts/prepare_data.py fake --n 512 --out data/shards/fake \
    --preset tiny model.n_codebooks=2 model.text_vocab_size=320
torchrun --standalone --nproc_per_node=2 scripts/train.py --preset tiny \
    --data data/shards/fake model.n_codebooks=2 model.text_vocab_size=320 \
    train.max_steps=50 train.ckpt_dir=checkpoints/smoke-ddp
# resume must continue from step 50:
torchrun --standalone --nproc_per_node=2 scripts/train.py --preset tiny \
    --data data/shards/fake model.n_codebooks=2 model.text_vocab_size=320 \
    train.max_steps=100 train.ckpt_dir=checkpoints/smoke-ddp
```

Then the same two runs with `--nproc_per_node=8` and, for the v6 path,
`train.strategy=fsdp2` explicitly once (the auto heuristic picks ddp for
frozen backbones; fsdp2 must also work — the wrapper shards
`backbone.get_decoder().layers`).

## 2. v6 backbone smoke (downloads Qwen3-1.7B)

```bash
# data prep with the real tokenizer writes shard format v2 (uint32 text row)
.venv/bin/python scripts/prepare_data.py fake --n 256 --out data/shards/hf-fake \
    --preset qwen3-1.7b model.n_codebooks=8
.venv/bin/python scripts/train.py --preset qwen3-1.7b --data data/shards/hf-fake \
    model.n_codebooks=8 train.max_steps=20 train.ckpt_dir=checkpoints/smoke-hf \
    --export checkpoints/smoke-hf-export
# stage transition: warm-start an unfrozen run from the frozen export
.venv/bin/python scripts/train.py --preset qwen3-1.7b --data data/shards/hf-fake \
    model.n_codebooks=8 model.freeze_backbone=false train.max_steps=5 \
    train.ckpt_dir=checkpoints/smoke-hf2 --init-from checkpoints/smoke-hf-export
```

Watch for: `[train]` per-head losses all finite; checkpoint dirs stay small
(frozen backbone is NOT saved); `save_keep` pruning old step dirs.
Llama-3.2 / Gemma-3 presets: repeat the build step only (`get_decoder()` /
embedding-scaling quirks were code-reviewed, never executed).

## 3. Latency gate (the reason the grid is 12.5 Hz)

```bash
.venv/bin/python scripts/benchmark.py --preset qwen3-1.7b --device cuda
.venv/bin/python scripts/serve.py --preset qwen3-1.7b --codec mimi --device cuda
```

Gate: decode step + depth rollout + Mimi encode/decode < 80 ms/frame
(DESIGN_V5: TTFA < 500 ms). Sampling is on-device on CUDA; if a step still
misses the budget, profile the depth rollout first (32 sequential codebook
steps/frame).

## 4. Gate 0 audio ceiling (before any big training)

Mimi-32 resynthesis ceiling per DESIGN_V3 §Gate-0, per language (v4). Then
stage A data prep with the REAL corpora — the mixture weights on
`--data DIR:WEIGHT` are sampling PROPORTIONS (resampled, shuffle-proof);
verify the printed dataset counts match the recipe before long runs.

## Known-unverified list (fixed in code, first exercised here)

- FSDP2 + frozen HF backbone end-to-end (wrap fixed to shard decoder layers;
  DCP save/load of the lean checkpoint under multi-rank).
- `train.backbone_lr` two-tier optimizer on a real unfrozen/LoRA run.
- Mimi streaming behavior inside the duplex console (rolling window is an
  approximation — compare against `test_streaming_encoder_window_close_to_batch`).
- bf16 autocast eval parity on CUDA (eval now runs under autocast).
