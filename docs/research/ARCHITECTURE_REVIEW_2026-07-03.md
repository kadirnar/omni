# Architecture Review — 2026-07-03 (multi-agent)

**Method.** 42-agent review: 6 code auditors deep-read every subsystem (executing snippets to verify invariants), 8 researchers fetched ~50 primary sources (papers, GitHub, engineering blogs — list in appendix), 7 judges scored one design dimension each against that prior art, and each judge's top code-asserting findings were re-verified by independent adversarial agents reading the repo (**19 CONFIRMED, 2 PARTIAL-with-correction, 0 refuted**). In parallel, empirical validation on this machine: full offline test suite and an end-to-end smoke run (fake data → 200 train steps → export → TTS generation).

**Bottom line.** The architecture is the field-consensus design, executed with unusual discipline — but the production v6 path **cannot currently reach the 8-GPU node**: six verified defects fire on the first real data-prep or `torchrun` invocation. All are localized fixes, none is a redesign.

---

## Verdicts by dimension

| Dimension | Verdict |
|---|---|
| Core sequence design (12.5 Hz grid, text row 0, delay at collate) | **sound-with-caveats** |
| Multi-codebook prediction (heads + stagger / depth + flat) | **sound-with-caveats** |
| v6 pretrained-backbone adaptation (+64 offset, freeze, adapters) | **sound-with-caveats** |
| Task unification (one loss-masked grid for all tasks) | **sound-with-caveats** |
| Data pipeline & codec strategy | **questionable** (two P0 defects) |
| Training recipe & scaling | **questionable** (three P0 defects) |
| Inference & product path | **sound-with-caveats** |

## Empirical validation (this machine, offline)

- Test suite: **172 passed, 1 skipped, 0 failed**.
- Smoke train (tiny preset, 2 codebooks, CPU): loss 4.67 → 2.52 over 200 steps with all streams learning (text 3.4→1.6, audio_0 5.6→3.9, audio_1 4.9→2.1); checkpoint save, export, reload, and TTS generation all worked; the model echoed the prompt text in its monologue.
- Sharp edge found by running it: `train.resume=true` (default) + default shared `ckpt_dir=checkpoints/run` made the trainer load a stale checkpoint from an earlier run with a different config — `maybe_resume` (`train/loop.py:429`) does **no config-compatibility check** and dies with a raw `load_state_dict` size-mismatch error.

---

## P0 — verified defects that block the 8-GPU run

All six were asserted by a judge and independently CONFIRMED by an adversarial verifier quoting the code.

1. **uint16 shard format rejects every v6 backbone tokenizer.** `ShardWriter` raises for `text_vocab_size > 65536` (`data/prepare.py:87-88`, `_GRID_DTYPE = <u2`), but `HFTextTokenizer` vocab is 64 + backbone size: Qwen3 ~152k, Llama-3.2 ~128k, Gemma-3 ~262k. `--tokenizer hf:ID` prep — the only path that produces v6 training data — cannot run for any supported backbone. Needs a shard-format v2 (e.g. uint32 or split-dtype text row). *The format is pinned in INTERFACES.md (binding contract) — surfacing, not editing.*
2. **`strategy=auto` crashes every production backbone on multi-GPU.** `pick_strategy` counts **total** params (frozen backbone included) against `fsdp_threshold_params=300M`, so every v6 model routes to FSDP2; the wrapper then iterates `model.blocks` (`train/distributed.py:167`), which `HFOmniModel` doesn't have (layers live in `backbone.get_decoder().layers` — `optim/perf.py` handles this, `distributed.py` doesn't). First `torchrun` dies with `AttributeError` unless `train.strategy=ddp` is passed explicitly. Fix: shard the HF decoder layers, and/or count only trainable params for the threshold.
3. **Dataset mixture weights are a silent no-op.** `MixDataset` weights only assign interleave *ordering* keys — every sample appears exactly once — and `BucketBatchSampler` then reshuffles all indices (`data/dataset.py:186-257`), erasing even that order. `--data DIR:WEIGHT` yields size-proportional mixing regardless of weights, so the stage A→D ratios (and the ~0.5% text-replay ratio the forgetting literature says is decisive) silently don't happen. Weights must resample/repeat, not reorder.
4. **No stage-transition path (freeze → unfreeze cannot warm-start).** `scripts/train.py:72` always calls `build_model` (fresh weights); `load_model` is unreachable from the training CLI; and resuming with `freeze_backbone` flipped loads a DCP optimizer state whose param-group structure no longer matches (`hf_omni.py:380-382` emits 1 vs 2 groups). DESIGN_V6 §5 Stage 1→2 has no implemented mechanism. Add `--init-from DIR` (load exported adapters, fresh optimizer/schedule).
5. **Reserved control ids are injectable via transcripts.** The trained BPE maps literal `"<user>"`/`"<eos>"` in text to control ids < 64 (verified: `encode('say <user> hi')` → id 4). Only the textlm path filters them (`prepare.py:590`); `prepare_asr_tts` (`prepare.py:793`) and `prepare_s2s` (`prepare.py:893,919`) pass raw ids into loss-target positions. Real scraped transcripts will train the model to emit turn/task control tokens mid-content. One shared filter at every encode site fixes it. (Applies to the trained-BPE path; ByteTokenizer/HFTextTokenizer can't emit ids < 64.)
6. **Vocab-size trap for padded-embedding backbones.** `hf_omni.py:95` derives `text_vocab_size` from `emb.num_embeddings`, but prep's `_check_tokenizer_cfg` (`prepare.py:414-425`) tells the user to set the *tokenizer* size (64 + len(tok)). For Qwen-style padded embeddings these differ, so following the printed hint produces shards that `_check_meta` rejects at train time. Derive both from one source.

## P1 — verified high-value risks (fix or ablate before/at GPU day)

- **No test pins generator delay math to `streams.apply_delay`.** The delay pattern is hand-re-derived once in `_delayed_prompt`/`_input_column` (`infer/generate.py`, shared by `stream()` and `duplex.py`), with equivalence claimed only in docstrings; no test asserts it (grep-verified). An off-by-one would pass the whole suite and surface as degraded audio after training. *Single highest-leverage test to add.*
- **TTS train/infer text-alignment mismatch.** With `--align`, prep emits word-aligned monologues (`prepare.py:803-812`), but `prompt_forced_text` (`grids.py:398-400`) always forces packed one-token-per-frame text at inference — out-of-distribution if most data is aligned. Moshi aligns both sides (Whisper timestamps at train *and* inference). Ablate packed-vs-aligned, or align the forced text.
- **Depth transformer trains on every frame.** Both forward paths run one depth call over all B·T positions, materializing `[B*T, 32, 2051]` fp32 logits (~0.5 GB/sample at T=2048) and adding roughly backbone-scale FLOPs. CSM trains its depth decoder on a random **1/16** of frames with no measurable loss difference — the single biggest memory/compute win available.
- **Flat-mode decode does 32 sequential CPU-synced depth steps per 80 ms frame.** `depth_sample_fn` moves logits to CPU per codebook (`generate.py:402-408`); `DepthTransformer.sample` re-runs the growing prefix per codebook with no KV cache (`layers.py:388-410`). That's 32 GPU→CPU round-trips + O(K²) attention per frame against the 80 ms budget. Batch sampling on-device.
- **Full-sequence fp32 text logits over the backbone vocab.** `_text_logits` (`hf_omni.py:227-231`) upcasts `[B,T,64+152k]` to fp32 (~1.2 GB/sample at T=2048) with no masked-position selection or chunked CE.
- **Present-head loss renormalization makes gradient balance batch-dependent.** `multistream_loss` divides by the weight-sum of heads *present* in the micro-batch; with `semantic_loss_weight=100` text CE carries weight 1/133 in mixed batches but 1.0 in text-only batches (~100× swing with task composition). Also 33 host syncs per micro-batch from per-head `bool(m.any())` + mask indexing. Normalize by a constant weight sum; vectorize the masked CE.
- **Duplex serving breaks the same-clock contract.** `serve/app.py:304` encodes each 80 ms chunk with a **stateless** Mimi (training encoded whole utterances; Mimi's receptive field spans seconds) — live user codes differ from training distribution. Moshi uses Mimi's streaming encoder state. Also: encode runs synchronously on the event loop, no pacing/drop policy when a tick exceeds 80 ms, `ChunkDecoder` re-decodes a 16-frame window per tick (~17× work), and the console is hardcoded to `device="cpu"` (no `--device` flag), so the DESIGN_V5 TTFA < 500 ms gate can't even be measured pre-GPU.
- **Resampling is unfiltered linear interpolation, in two divergent copies.** `prepare.py:185` (endpoint-inclusive linspace) vs `audio/codec.py:241` (rate-exact arange) — measured maxdiff ~0.1 on a sine; a 15 kHz tone aliases to 9 kHz on 48k→24k. HF corpus prep usually goes through `datasets.Audio`/torchcodec (FFmpeg, filtered), but `load_wav` (voice refs, serve) and TTS-output paths use the linear one. Use one shared polyphase resampler (torchaudio/soxr).
- **Checkpointing at scale.** Multi-rank DCP saves the full model including the never-changing frozen ~8B backbone every `save_every` (~16-32 GB/step dir), no retention pruning, mid-epoch resume replays the epoch from its start, and `maybe_resume` has no config-compat check (bit us empirically, see above). Distributed save/resume is never tested before first GPU contact — add a 2-process gloo CI test.
- **Frozen-backbone-only duplex/s2s has no precedent.** Freeze-Omni validates frozen backbones only for turn-based QA with a state-machine duplex; Moshi full-finetunes 7B for always-on duplex; GLM-4-Voice/Step-Audio-2/Kimi-Audio/Qwen-Omni all train the backbone. Budget a LoRA/unfreeze stage with ~0.5% text-data replay (Interspeech-2025 forgetting study: LoRA-only collapsed text QA 70%→14%; replay recovered it to ~66%).

## Improvements suggested by prior art (cheap, high-confidence)

- **Third delay mode: semantic lead.** `streams.delays()` offers only `stagger` (k→k+1) and `flat` (all 1); flat pins Mimi's *semantic* codebook 0 to the same delay as acoustics. Moshi's ablation on this exact grid — semantic delay 0, acoustics uniformly delayed 1-2, *with* the depth transformer — improved ppl 42.2→36.8 and stability. Because delay is applied at collate, shards are unaffected; but `streams.py`/`config.py` are frozen contracts (`use_depth == (mode=="flat")` is asserted), so this needs a contract revision — surfaced here.
- **Per-task text-audio delay as a knob.** Kyutai's Delayed Streams Modeling distinguishes tasks by *delay* (TTS: audio ~2 s behind text; ASR: text 0.5-2.5 s behind audio; pretraining randomizes ±0.6 s), not loss masks alone. Omni fixes text delay 0 everywhere, making TTS harder than the DSM formulation (near-zero text lookahead). The collate-time architecture already permits this.
- **CSM-style depth-loss amortization (1/16 frames)** — see P1; direct precedent, reported lossless.
- **Pin the FakeCodec↔Mimi frame-count contract.** FakeCodec pins `T=ceil(samples/1920)`; Mimi's count for non-multiple inputs is whatever transformers emits — unverified (MimiCodec has zero tests, even gated). A one-frame mismatch shifts every alignment when swapping fake→mimi. Run a small paid-download validation (frame counts, HFTextTokenizer roundtrip on the real SentencePiece) before bulk prep.
- **Voice cloning:** per CSM's 90 s-context result, consider raising the 125-frame (10 s) reference cap; implement the DESIGN_V5 cached-prefix optimization (`[<bos>+voice segment]` KV prefix is already layout-stable across tasks/turns).

## What is genuinely strong (verified)

- **The grid contract holds end to end, by execution not by reading:** apply_delay/undelay round-trip exactly (stagger/flat × duplex), masks land at `dl[s]+t`, logits-at-p-predict-p+1 gating is correct, special ids 0..51 unique under the 64 reserve, preset arithmetic (`max_sample_frames + max_delay ≤ max_frames`) holds for all eight presets.
- **Structural train/infer parity:** inference prompts are built by the *same* `grids.py` builders as training samples; forced text matches segment layout token-for-token; duplex user rows are doubly loss-excluded (mask **and** head iteration). Cleaner than Mini-Omni2's 181k-vocab surgery; equivalent to Kyutai's DSM formalization.
- **Production presets sit on the validated point:** 32 Mimi codebooks with flat delay-1 + a CSM-sized depth transformer (4 layers, d=1024) — exactly CSM's shipped configuration (and Moshi's RQ-Transformer shape at 8), avoiding Sesame's documented time-to-first-audio objection to stagger at large K. `semantic_loss_weight=100` replicates Moshi. Stagger survives correctly as the small-K/test path (MusicGen validated K≤8).
- **The +64 special-token offset design** (fresh 64-row `special_emb`/`special_head` concatenated before the intact pretrained `lm_head`) is cleaner than the vocab-surgery alternatives in Mini-Omni2-class repos; adapter-only export correctly handles tied heads, shared depth embeddings, and LoRA keys with bit-exact reload tests.
- **Undelayed-shards + collate-time delay** is better engineering than Moshi/MusicGen practice — delay ablations are free and never invalidate data.
- **Offline test discipline** (FakeCodec/SineTTS, in-process random Llama for HF-path tests, 172 green tests) is unusually honest; depth forward/sample parity is verified numerically.

## Where omni sits vs the field (July 2026)

| System | Sequence design | Codec | Backbone handling | Duplex |
|---|---|---|---|---|
| **omni** | parallel streams, text row 0, delay @ collate | Mimi 12.5 Hz, 8→32 cb | frozen HF LM + new modules (LoRA opt.) | same-clock 2nd stream group |
| Moshi (Kyutai) | parallel streams (RQ-Transformer), inner monologue | Mimi 12.5 Hz, 8 cb | 7B trained end-to-end | same-clock dual stream |
| Sesame CSM | backbone + depth decoder, 32 cb | Mimi 12.5 Hz, 32 cb | Llama trained; depth on 1/16 frames | — |
| GLM-4-Voice | interleaved text/speech single stream | 12.5 Hz single semantic cb + flow vocoder | full training | turn-based |
| Qwen3-Omni / Qwen3.5-Omni | Thinker-Talker (two models) | multi-codebook | full training | streaming, turn-based |
| Kimi-Audio | parallel text+audio heads, one LM | 12.5 Hz semantic + acoustic | full training | turn-based |
| Freeze-Omni | adapters on frozen LLM | — | **frozen** (the omni precedent) | 3-state interrupt machine |
| Mini-Omni/2 | text-0 + delayed audio layers (same layout as omni) | SNAC 24 kHz | mostly frozen, adapters | — |

Consensus points omni matches: 12.5 Hz frame grid (Moshi/CSM/GLM-4-Voice/Kimi-Audio all converged), text-as-row-0 inner monologue, depth-transformer factorization at high codebook counts, reference-audio-prefix voice cloning (VALL-E/CSM lineage), control-token prefixes on the text row (Whisper/CosyVoice3 conventions). The one place omni is ahead of everyone: collate-time delay over undelayed shards. The one place it's betting without precedent: always-on duplex on a *frozen* backbone.

## Minor issues (auditor findings, unverified-by-second-agent but file:line-cited)

- `config.py:265` `_coerce` zips tuples against current value — `train.betas=[0.9,0.95,0.99]` silently truncates to 2; `config.py:268` unrecognized bool strings coerce to False (`model.duplex=frue` → disabled, no error). *(frozen file — surfaced)*
- `grids.py:136` empty-audio `assistant_speech_segment` places AUDIO_EOS on the `<assistant>` column as a loss target; no guard. `streams.py:138` `turn_prefix` drops `intensity` when `response_style` is None. *(frozen files — surfaced)*
- `hf_omni.py:231` logits span embedding rows > tokenizer vocab (padded rows samplable, then silently dropped by decode; no logit mask). `hf_omni.py:92-104` doesn't adopt `rope_theta`/`head_dim` from the HF config (exported config.yaml records wrong values; harmless today).
- `hf_omni.py:380` weight decay applies to RMSNorm weights and all embedding tables (standard practice excludes them).
- `loop.py:291` eval runs without autocast (fp32 val vs bf16 train numerics, ~2× slower, per-rank only); `loop.py:94` wsd schedule ignores warmup overlap.
- `dataset.py:315` `_check_meta` never compares tokenizer identity across mixed shard dirs (two different 48k BPEs mix silently).
- `tokenizer.py:150` paralinguistic/emotion tags map to special ids only on the trained BPE; on HFTextTokenizer, `"<laugh>"` byte-encodes to ordinary ids ≥64 with no guard.
- `serve/app.py:137,265` busy-lock TOCTOU (second connection queues instead of getting the documented busy error); `app.py:301` `torch.frombuffer` on odd-length client frames kills the session uncaught.
- `prepare.py:261` turn truncation can strand a dangling `<emo_pcv>`/`<emo_rsp>` marker without its class token; `prepare.py:723` voice-pool dict grows unboundedly (~10 GB at 100k speakers).
- `test_codec.py:87` wav roundtrip asserted at atol=2e-3 where bit-exactness is achievable.

## Suggested order of work

1. Shard format v2 (uint32 text row) + control-id filter + vocab-size unification — unblocks all v6 data prep. *(contract revision)*
2. FSDP2 wrap over HF decoder layers (or trainable-param threshold) + `--init-from` stage transition + 2-process gloo CI test — unblocks the training recipe.
3. Resampling-based mixture weights — unblocks the stage A→D curriculum and text replay.
4. The delay-parity property test (generator vs `apply_delay`) — cheapest insurance in the repo.
5. Depth-loss 1/16 amortization + on-device sampling + constant-sum loss normalization — memory/latency/stability at scale.
6. Streaming Mimi encode + pacing in duplex serve + `--device` flag — makes the latency gate measurable.
7. Ablations enabled by the architecture: semantic-lead delay mode *(contract revision)*, per-task text-audio delay, packed-vs-aligned TTS.

## Sources consulted (deduplicated)

Papers: Moshi (arXiv:2410.00037), Delayed Streams Modeling (2509.08753), Hibiki (2502.03382), MusicGen (2306.05284), RQ-Transformer (2203.01941), AudioLM (2209.03143), VALL-E (2301.02111), VALL-E 2 (2406.05370), UniAudio (2310.00704), GLM-4-Voice (2412.02612), Qwen2.5-Omni (2503.20215), Qwen3-Omni (2509.17765), Qwen3.5-Omni (2604.15804), Kimi-Audio (2504.18425), Step-Audio 2 (2507.16632), Freeze-Omni (2411.00774), LLaMA-Omni (2409.06666), LLaMA-Omni2 (2505.02625), Mini-Omni (2408.16725), Mini-Omni2 (2410.11190), SLAM-Omni (2412.15649), forgetting-mitigation study (2505.17496, Interspeech 2025), dGSLM (2203.16502), LSLM (2408.02622), SyncLLM (2409.15594), duplex survey (2509.14515), BayLing-Duplex (2606.14528), XY-Tokenizer (2506.23325), X-Codec-2.0 update (2601.20185), CosyVoice 2 (2412.10117), CosyVoice 3 (2505.17589).
Code/blogs/cards: kyutai-labs/moshi, kyutai-labs/delayed-streams-modeling, kyutai.org blog (MoshiRAG 2026-04), kyutai/mimi model card, Sesame "Crossing the uncanny valley of conversational voice" + SesameAILabs/csm + sesame/csm-1b + HF Transformers CSM docs, THUDM/GLM-4-Voice, QwenLM/Qwen3-Omni, MoonshotAI/Kimi-Audio, stepfun-ai/Step-Audio2, hubertsiuzdak/snac, canopyai/Orpheus-TTS, Zyphra/Zonos conditioning docs, huggingface/parler-tts, ElevenLabs v3 audio-tags post, si.inc/posts/hertz-dev.

*Generated by a 42-agent review workflow (map → research → judge → adversarial verify), 2026-07-03. Empirical runs: pytest (172 pass/1 skip), 200-step smoke train + TTS generation on CPU.*

---

# ADDENDUM 2026-07-03: fix pass completed

All six P0 blockers, every P1 risk, and the minor issues above were fixed the
same day (suite: 172 → 246 tests, all green; contract deltas recorded in
INTERFACES.md "2026-07-03 review-fix amendments"). Map:

| Finding | Fix |
|---|---|
| P0-1 uint16 shard format | Shard format v2: uint32 text row, auto-selected; v1 read compat |
| P0-2 FSDP2 `model.blocks` crash + threshold | Wrap shards HF decoder layers; auto strategy counts TRAINABLE params |
| P0-3 mixture weights no-op | MixDataset resamples (weights = proportions); unweighted = natural; mixed forms rejected |
| P0-4 no stage transition | `--init-from` + `omni.model.load_weights` (fresh optimizer, structural check) |
| P0-5 control-id injection | Reserved ids < 64 filtered at every transcript encode site (incl. aligned monologues) |
| P0-6 vocab trap | `_check_meta` accepts smaller meta vocab for backbone configs; tokenizer_id cross-dir check |
| Delay-parity test gap | tests/test_delay_parity.py pins `_delayed_prompt`/`_input_column` to `apply_delay` (all modes × duplex); `generate` now consumes `stream` (one loop) |
| Depth loss on every frame | `depth_loss_ratio` (CSM 1/16 in depth presets), `ModelOutput.audio_positions` |
| 32 CPU syncs/frame decode | On-device sampling on CUDA (generate/stream/duplex) |
| Loss renorm batch-dependence + 33 syncs | Constant-sum denominator, ignore_index CE, one host sync, internal zero anchor |
| Weight decay on norms/embeddings | Decay-split param groups on both model classes |
| Duplex serve stateless encode | `StreamingEncoder` rolling-context window; encode off the event loop; `behind_ms` pacing metric |
| CPU-locked console | `create_app(device=)`, `scripts/serve.py --device` (chat already had it) |
| Aliasing linear resamplers (×3) | One Kaiser-sinc `resample()` in audio/codec.py, used by prep/serve/load_wav |
| Resume config crash (empirical) | Structural compat check with actionable error; checkpoint retention `save_keep`; lean checkpoints skip the frozen backbone; eval under autocast; wsd warmup guard |
| Moshi semantic-lead delay | New `"lead"` mode (contract amendment), trains e2e offline |
| TTS aligned/packed OOD | Prep alternates aligned/packed TTS rows; forced-text overflow raises |
| Mimi/tokenizer contracts unverifiable offline | tests/test_gpu_preflight.py (RUN_SLOW gated) + docs/GPU_DAY_CHECKLIST.md |
| Misc (turn_prefix intensity, empty segments, bool/tuple coercion, busy TOCTOU, frombuffer, voice-pool memory, truncation markers, FakeCodec/Mimi decode guards) | All fixed with tests |

Deliberately NOT changed: `checkpoints/run` default (documented + resume now
fails loudly), ChunkDecoder windowed re-decode (inherent to crossfade design),
per-task text-audio delay (recipe-level knob, DSM-style — future ablation).
