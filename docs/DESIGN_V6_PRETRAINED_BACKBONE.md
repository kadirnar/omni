# Design V6 — Pretrained LLM backbone (Qwen3 / Llama 3 / Gemma)

> Status: IMPLEMENTED 2026-07-03 (queue items 1–5; item 6 — GPU validation with
> a real backbone — is first-GPU-session work). Supersedes the from-scratch
> backbone of DESIGN.md §3 and the Stage A text-pretrain of §4 for the
> production path. The from-scratch `OmniModel` stays for tiny/CPU contract
> tests and ablations. Contract tests: `tests/test_hf_backbone.py` (34 tests,
> offline, tiny in-process random Llama).
>
> Goal: keep everything that makes this repo work — the frame grid, Mimi codec,
> delay machinery, emotion/i18n control tokens (V4), voice cloning (V5), depth
> transformer (V3) — but replace the randomly-initialized temporal transformer
> with a pretrained decoder-only LLM, so we never pay for text pretraining and
> inherit strong multilingual + reasoning ability from day one.

## 1. Why this works with the existing architecture

The temporal backbone in `model/omni.py` is already interface-shaped: it maps a
sum-of-streams embedding `[B, T, D]` to hidden states `[B, T, D]`, and everything
else (audio embeddings, depth transformer, loss, frame-synchronous generator)
hangs off that. Moshi, Qwen2.5-Omni, LLaMA-Omni2, and CSM all demonstrate the
same recipe: **a pretrained text LLM consumes/produces the text stream while new
audio embedding tables and audio heads are grafted on and aligned with speech
data**. Nothing about the 12.5 Hz grid, the undelayed-on-disk contract, or the
`loss_mask`/`channel` format changes.

What we delete from the plan: Stage A (5–15B tokens of text pretraining), the
48k custom BPE, and most of Stage B's text-LM replay. What we keep verbatim:
`streams.py`, `grids.py`, codec, data shards, generator frame loop, serve.

## 2. Backbone integration (`model/hf_omni.py`, new)

`HFOmniModel` — same public surface as `OmniModel` (`embed`, `forward`, `loss`,
`prefill/step[_hidden]`, `save_pretrained`/`from_pretrained`, `param_counts`),
so `Trainer`, `OmniGenerator`, chat/serve work against either class through the
existing method contract.

- **Backbone**: `transformers.AutoModelForCausalLM.from_pretrained(cfg.backbone_id)`,
  used with `inputs_embeds=` (never `input_ids=`) so we control the input sum.
  Qwen3, Llama 3.x, and Gemma 2/3 all support `inputs_embeds` + `past_key_values`
  + `output_hidden_states`. `d_model` is read from the HF config
  (`hidden_size`), not from ours.
- **Input embedding = sum of streams**, exactly as today:
  `backbone.get_input_embeddings()(text_ids_mapped)` + new
  `audio_embs[k]` (`n_codebooks` tables at backbone hidden size) + new
  `channel_emb` (2 × hidden). New tables init to **zero-mean, small std, and the
  channel/audio contribution starts near zero** so step 0 behavior ≈ the text LLM.
- **Text head**: the backbone's own `lm_head` (tied or not — backbone's choice).
- **Audio output**: unchanged V3 design — `DepthTransformer` with
  `in_proj: hidden_size -> depth_d_model` (flat delays), or parallel heads for
  stagger mode. The depth transformer is new-initialized either way.
- **KV cache**: HF `Cache` (`DynamicCache`/`StaticCache`) instead of our
  `KVCache`. `prefill`/`step` keep their signatures; the cache object type is
  opaque to the generator (it only calls `allocate`-like factory + passes it
  back). RoPE, GQA, norms are the backbone's own — `model/layers.py` is not
  used by the HF path.
- **Position/context**: one grid step = one backbone position. 2048 frames
  (164 s) is far inside every candidate's context window.

## 3. Text-stream id space: keep the frozen 0..63 contract

The frozen contract says text ids 0..63 are omni specials and "real text starts
at 64". We keep that **exactly**, with an offset scheme — no edits to
`streams.py`/`grids.py`, and shards stay valid across backbones of the same
tokenizer:

- `HFTextTokenizer` (in `text/tokenizer.py`, duck-typed like `TextTokenizer`):
  wraps the backbone's HF tokenizer. `encode(text)` returns
  `[hf_id + 64 for hf_id in hf_tokenizer.encode(text)]`;
  `vocab_size = 64 + hf_vocab_size`; `decode` strips ids < 64 (or renders their
  `<name>` strings). Never auto-adds the backbone's own BOS/EOS/chat template.
- **Embedding lookup** for text row id `i`:
  `i < 64` → new trainable `special_emb` table (64 × hidden, fresh init);
  `i >= 64` → frozen/finetuned backbone embedding of `i - 64`.
- **Text logits**: `cat([special_head(h), backbone.lm_head(h)], dim=-1)` —
  a new 64-row special head plus the pretrained head, so `<end_of_turn>`,
  `<text_pad>`, `<lang_*>`, `<emo_*>` etc. are ordinary logits at ids 0..63 and
  all V4/V5 control-token machinery (`turn_prefix`, monologue parsing, chips in
  the serve UI) works unmodified.

This avoids `resize_token_embeddings` entirely: no mutation of pretrained
weights, checkpoints of adapters stay separable, and the same shards serve any
backbone **with the same tokenizer family**. Shard `meta.json` gains a
`tokenizer_id` field; `dataset.py` asserts it matches the run's backbone
(shards are cheap to re-tokenize per family — audio codes dominate prep cost
and are tokenizer-independent).

## 4. Config (requires a sanctioned edit to the FROZEN `config.py`)

New `ModelConfig` fields (defaults preserve current behavior; `backbone_id=None`
means the from-scratch path, and every existing preset/test is untouched):

```python
backbone_id: str | None = None      # e.g. "Qwen/Qwen3-1.7B", "meta-llama/Llama-3.2-3B", "google/gemma-3-4b-pt"
backbone_dtype: str = "bf16"
freeze_backbone: bool = True        # stage 1; stage 2 sets False (or uses LoRA)
lora_rank: int = 0                  # 0 = no LoRA; >0 needs optional dep `peft`
lora_alpha: int = 16
lora_dropout: float = 0.0
```

When `backbone_id` is set: `d_model/n_layers/n_heads/...` are ignored (read from
the HF config), `text_vocab_size` is derived (`64 + hf_vocab`), and
`model.build_model(cfg)` (new factory in `model/__init__.py`) returns
`HFOmniModel`; otherwise `OmniModel`. New presets:

| preset | backbone | audio side | use |
|---|---|---|---|
| `qwen3-1.7b` | Qwen/Qwen3-1.7B | 32 cb, flat + depth (V3) | 1-GPU dev / small prod |
| `qwen3-8b` | Qwen/Qwen3-8B | 32 cb, flat + depth | 8-GPU production target |
| `llama32-3b` | meta-llama/Llama-3.2-3B | 32 cb, flat + depth | ablation |
| `gemma3-4b` | google/gemma-3-4b-pt | 32 cb, flat + depth | ablation |

**Recommended default: Qwen3.** Apache-2.0, strongest multilingual coverage of
the three (all 12 `LANG_TAGS` languages incl. Turkish), sizes from 0.6B–8B on
one node, GQA + generous context. Llama 3.2 is license-gated; Gemma is the
fallback if Qwen tokenizer/codec interactions disappoint. Base (`-pt`) or
low-RLHF checkpoints preferred over heavily-instructed ones — the monologue
format is our own and instruct tuning fights `<text_pad>` pacing.

## 5. Training recipe (replaces DESIGN.md §4 stages)

Fused AdamW, cosine/WSD, bf16 — the existing `Trainer` — plus **param groups**:
new modules (audio embs, special emb/head, channel emb, depth) at full LR
(~3e-4); backbone at 0 (frozen) or ~1e-5..2e-5 when unfrozen; LoRA params at
~1e-4 when used.

- **Stage 1 — modality alignment** (backbone FROZEN; train audio embs, special
  emb/head, channel emb, depth transformer): ASR 30% / TTS 40% / audio-LM 20% /
  s2s 10%. The LLM already knows text; this stage teaches the new tables to
  read/write Mimi space. ~2–5k h speech. Cheap: <10% of params get grads.
- **Stage 2 — full s2s + emotion + cross-lingual** (unfreeze backbone at low LR,
  or LoRA r=32..64 on attn+MLP for the 8B): V4 mixture — s2s dialogues with
  emotion tags 50% / SER-tagged ASR 15% / TTS incl. paralinguistics 20% /
  cross-lingual s2s + voice-consistency pairs (V5) 15%. Small text-only replay
  (2–5%) guards against catastrophic forgetting of the backbone.
- **Stage 3 — polish** (optional, unchanged from V3/V4 queues): duplex, DPO on
  emotion-appropriateness, CFG decode.

Distributed: unchanged `auto` strategy; 8B backbone + depth ≈ 8.5B → FSDP2 path
(threshold already at 300M). `freeze_backbone` + FSDP2 works (frozen params
still shard); LoRA needs `peft` as a **lazy optional dep** (`omni[lora]`).

## 6. Inference & latency

`OmniGenerator` frame loop unchanged (it only touches the model through
`prefill_hidden`/`step_hidden` + `depth.sample`). Budget: 80 ms/frame. A 1.7B
bf16 backbone is ~1–3 ms/step on an A100/H100 (well inside budget with the
depth transformer's 32 sequential codebook steps — see V3's depth-amortization
queue item, which still applies); 8B ~throughput-bound but fits with static
cache + `torch.compile` (`optim/perf.py` extends to the HF path via
`model.backbone.forward` compile). CPU/MPS dev uses the tiny random backbone
(below), not a real checkpoint.

## 7. Offline dev & tests (no downloads, per the frozen global rules)

`tests/` gain an `HFOmniModel` contract suite driven by a **locally constructed
tiny backbone**: `transformers.LlamaForCausalLM(LlamaConfig(hidden_size=64,
num_hidden_layers=2, vocab_size=512, ...))` built in-process — random weights,
zero network. A `FakeHFTokenizer` (whitespace/byte-level, duck-typed) pairs with
it. Every existing contract test (grid round-trip, delay math, generator index
math, trainer smoke) runs against both model classes via a fixture param.
`backbone_id` real downloads happen only on explicit user paths, same rule as
`--codec mimi`.

## 8. What changes / what doesn't

| unchanged | changed | new |
|---|---|---|
| `streams.py`, `grids.py`, delay invariant | `config.py` (+6 fields, new presets) — sanctioned freeze edit | `model/hf_omni.py` |
| codec, shard format (+`tokenizer_id` in meta) | `model/__init__.py` factory | `HFTextTokenizer` |
| `OmniGenerator` frame loop, chat, serve | `Trainer` param groups; export saves adapters + backbone ref | tiny-backbone test fixtures |
| V4 emotion/i18n tokens, V5 voice cloning | `prepare_data.py`/`train.py` `--backbone` flag | presets `qwen3-*`, `llama32-3b`, `gemma3-4b` |
| from-scratch `OmniModel` (tests/ablation) | DESIGN.md §3–4 superseded for prod | `omni[lora]` optional extra |

**Checkpoint format**: `save_pretrained` writes adapters (audio embs, special
emb/head, channel emb, depth, LoRA) + `config.yaml` with `backbone_id`; the
backbone itself is referenced, not copied, unless Stage 2 did full finetune
(then a merged safetensors export via `--export-full`).

## 9. Implementation queue

1. ~~`config.py` backbone fields + presets + factory dispatch~~ DONE (freeze edit; every prior test green).
2. ~~`HFTextTokenizer` + offset scheme + tiny-backbone/fake-tokenizer test fixtures~~ DONE.
3. ~~`HFOmniModel`: embed/forward/loss parity vs contract tests; HF cache prefill/step~~ DONE (exact prefill/step-vs-forward logits parity verified; generator/duplex/perf now allocate through `model.new_cache()`).
4. ~~Trainer param groups + freeze/LoRA wiring; save/load adapters~~ DONE (`omni[lora]` extra; adapter-only `adapters.safetensors` export; LoRA weights ride the adapter file).
5. ~~CLI wiring + shard `tokenizer_id`~~ DONE (backbone presets work in every script; `--tokenizer hf:<model_id>`; `tokenizer_id` recorded in `meta.json` as provenance).
6. GPU validation (first GPU session): `qwen3-1.7b` build + Stage 1 on LibriSpeech ASR/TTS; per-arch smoke of Llama 3.2 / Gemma 3 (`get_decoder()`/embedding-scaling quirks); then the V4 emotion/i18n mixture as Stage 2. Also pending: full-finetune consolidated export (`--export-full`), checkpoint dirs currently save the full model state (incl. frozen backbone) via `trainer_state.pt`/DCP — fine for correctness, wasteful at 8B (optimization queue item).

## Risks

- **Instruct-tuned backbones fight the monologue format** — prefer base
  checkpoints; if pacing breaks, fall back to word-aligned monologue (DESIGN.md
  risk list).
- **Tokenizer coupling of shards** — mitigated by `tokenizer_id` in meta and
  the fact that codec tokens (the expensive part) are reusable; text re-tokenize
  is a fast metadata pass.
- **Frozen-stage audio quality plateau** — if Stage 1 alignment stalls, unfreeze
  the top-k backbone layers early (Moshi does full finetune; we have the knob).
- **Gemma tied-embedding / soft-cap quirks** — handled inside `HFOmniModel`
  (always read `hidden_states[-1]`, never assume `lm_head` untied); covered by
  per-arch smoke tests on GPU day 1.
