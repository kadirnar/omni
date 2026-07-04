# Omni-S2S v3: Audio-Quality Redesign

> Produced 2026-07-02 from a 4-agent paper/systems survey (raw reports: docs/research/quality-*.md) after the Kokoro-class TTS plan was rejected. Supersedes DESIGN.md §2 (codebook count), §4 (data recipe), and §6 sampling defaults where they conflict.

# Omni-S2S Audio-Quality Redesign (v3 brief)

Goal: replace the rejected Kokoro-grade output with high, natural voice quality while keeping the code-complete frame-grid LM, the frozen kyutai/mimi codec, full/half duplex, and the everything-synthesizable data property.

## 1. Audio output path

**PRIMARY: Mimi @ 32 codebooks + the existing CSM-style depth transformer (`audio_delay_mode="flat"`, `use_depth=True`), decoded by Mimi's own decoder.**

Where the reports disagreed — output-architectures preferred a semantic-token LM + pretrained flow-matching renderer; codecs-vocoders preferred Mimi-32 + depth — we choose **Mimi-32** because: (a) it is nearly free given our implemented depth path and undelayed-shard format; (b) it preserves 80 ms frame-synchronous streaming and full duplex (a renderer adds 0.5–1 s first-audio latency and breaks the duplex loop); (c) it is de-risked at exactly our scale by sesame/csm-1b (Apache-2.0, 1B+100M) and kyutai/tts-1.6b-en_fr (CC-BY-4.0, 1B+600M), both of which generate all 32 Mimi codebooks and sound natural; (d) no license entanglement — the released kyutai/mimi checkpoint has `num_quantizers=32` by default, and our `MimiCodec` already accepts `n_codebooks` up to 32. The megahours of acoustic naturalness that the renderer path would supply are instead **distilled through the training data** (Section 3: renderer-grade TTS synthesizes the assistant speech).

Exact config (new `quality` preset):
- Codec: `kyutai/mimi`, `encode(num_quantizers=32)` → 4.4 kbps. Reconstruction ceiling moves from PESQ 2.07 / "muffled, crackly" (Kyutai's own words for 8 cb) to **PESQ 3.18 / ESTOI 0.910**.
- Backbone: keep validated base shapes — d1536 / L26 / H12 / KV4 / ff4096 / ctx 2048 frames.
- Depth transformer scaled up: `depth_d_model=1024, depth_n_layers=4, depth_n_heads=8` (from 256/2/4), 32 positions, shared audio embeddings as today.
- Training compute: CSM **1/16-frame amortization** — full 32-position depth pass on a random 1/16 of frames; cb0 logits on every frame (position-0-only depth call or a CSM-style backbone cb0 head). Without this, 32-position teacher forcing on all B·T positions quadruples step cost; with it, cost stays near the 8-cb run.
- **Do NOT use stagger mode at 32 cb**: the MusicGen delay would add 31 frames = 2.48 s intrinsic latency. Stagger stays only for tiny/CI back-compat.

Quantified (quality preset, half-duplex):
- Params added vs 8-cb base (0.76B): +75M backbone audio-embedding tables (24 extra × 2051 × 1536), +~120M depth transformer (in_proj 1.6M, 4 blocks ≈ 50M, 32 heads × 2051 × 1024 ≈ 67M) → **≈ 0.95–1.0B total** (mirrors CSM-1B). Duplex: keep user-side rows at 8 codebooks (input-only; semantic content lives in low codebooks) to avoid another +75M.
- Decode latency at 12.5 Hz: backbone step ~15–25 ms (1B, compiled, A100-class) + 32 sequential depth steps ≈ 10–15 ms + Mimi-32 decode ≈ 5 ms → **~35–45 ms per 80 ms frame: real-time with ~2x headroom**. First audio after turn end drops from ~640 ms (stagger-8) to ~80–240 ms (flat delay = 1 frame).
- Expected quality vs current: removes the Mimi-8 fidelity cap entirely (the single largest verified quality lever available to us); with the new data + DPO, target DistillMOS ≥ 4.0 and a clear blind-test win over Kokoro. Honest ceiling: CSM trained on ~1M h; at our ~30k h expect "clean, expressive, contextually-prosodic assistant voice," not human-indistinguishable.
- Integration effort: ~3–5 days (preset + depth amortization + shard re-encode + sampling/CFG changes + tests), plus one GPU pass to re-encode shards.

**FALLBACK (only if Gate C/D fails on acoustics): Mimi-cb0 semantic stream + self-trained flow-matching renderer.** Backbone emits only cb0 (optionally cb0–3); a ~150–200M chunk-aware causal flow-matching model (CosyVoice2 Apache-2.0 training code / Chatterbox S3Gen MIT as scaffold) conditions on cb0 tokens + a fixed persona reference and predicts mel; `nvidia/bigvgan_v2_24khz_100band_256x` (MIT, 122M) vocodes. Validated pattern: Kimi-Audio (12.5 Hz semantic → FM+BigVGAN) and Voxtral TTS (FM beats a 36-step depth transformer). Costs: +0.5–1 s first-audio latency, weeks of renderer training on 8 GPUs, speaker identity moves into reference embeddings. It keeps shards/codec unchanged (cb0 is already row 1 of every shard). We rejected off-the-shelf renderers (glm-4-voice-decoder etc.) because none conditions on Mimi tokens and switching tokenizers invalidates the entire stack.

## 2. Model scale

Naturalness evidence: Moshi 7B/7M h = flat voice; CSM 1B+100M/1M h = near-human without context; Orpheus 3B/100k h = expressive TTS; LFM2-Audio 1.5B on Mimi-8 = "fine, not human" (a direct warning about our current setup). Conclusion: **~1B-class backbone + depth decoder is the honest minimum for natural output, and data matters more than params beyond that**. Revised preset table:

| preset | role | backbone | n_cb / mode | depth | total |
|---|---|---|---|---|---|
| tiny | CPU CI | d256/L6 (unchanged) | 8, **flat+depth default** | d256/L2 | ~22M |
| small | 1-GPU ablations | d1024/L16 | **32, flat+depth** | d512/L2 | ~0.3B |
| base | legacy 8-cb | d1536/L26 | 8, stagger | — | 0.76B |
| **quality (new)** | 8-GPU primary | d1536/L26/ctx2048 | **32, flat+depth** | d1024/L4/H8 | **~1.0B** |

Tiny and small default to the depth path so CI and ablations exercise what we ship. Honest minimum for "consistently natural": ≥1B params and **≥100k h speech** (CSM datapoint; Kyutai TTS used 2.5M h). Our committed plan is ~30–35k h (Section 3), which should land "clearly natural, far above Kokoro" with DPO and context conditioning; the documented path to 100k+ h is scaling the Emilia-YODAS ingest (114k h, CC-BY-4.0) — no architecture change required.

## 3. Data recipe v2 (no Kokoro)

All licenses commercially clean. The everything-synthesizable property is preserved: every stage can be regenerated from text + the new TTS backends.

**Bulk assistant/dialogue synthesis (~2,500 h):**
- `microsoft/VibeVoice-1.5B` (MIT) — dialogue-native, multi-speaker, long-form stable; MOS-family leader among open models. RTF 0.2–0.5 → 2,500 h ≈ **500–1,250 H100-hours** (3–7 days on the 8-GPU node).
- `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` (Apache-2.0, RTF ~0.3) — single-speaker turns, instruct-based emotion control, and zero-shot voice cloning from Emilia-YODAS speaker prompts (this is the user-side voice diversity engine).
- `Soul-AILab/SoulX-Podcast-1.7B` (Apache-2.0) — tag-driven nonverbals (<|laughter|>, <|sigh|>, <|breathing|>) to populate our paralinguistic-token training data.
- Text sources unchanged (soda, ultrachat, oasst2), speakability filter unchanged.

**Fixed-persona polish stage (~300–500 h):** VibeVoice-Large 7B MIT mirror (`aoi-ot/VibeVoice-Large`; Microsoft pulled the official 7B) with ONE curated voice prompt, optionally passed through `stepfun-ai/Step-Audio-EditX` (Apache-2.0) for emotion/style editing — this is Moshi's single-actor-many-styles trick executed synthetically. ~400–800 GPU-hours. Avoid: Voxtral TTS and F5-TTS (CC-BY-NC), Higgs Audio v2 (license dispute, org pivot to non-commercial), Index-TTS2 (non-commercial), Dia (unstable identity), MegaTTS3 (cloning gated by ByteDance queue).

**Real expressive corpora (~25–30k h Stage B mix):** existing ASR sets (librispeech 960 h, mls_eng 8–12k h, peoples_speech clean 3–5k h, libritts_r 585 h) + **`amphion/Emilia-Dataset` YODAS split (CC-BY-4.0, take a filtered conversational-English 10–20k h subset)** for spontaneous prosody + `kyutai/DailyTalkContiguous` (CC-BY-SA-4.0, stereo — direct duplex training data). Expresso (CC-BY-NC) reserved for evaluation only. CANDOR/Fisher: turn-taking/backchannel timing statistics only, never audio.

**User-side audio diversity:** thousands of CosyVoice3-cloned voices from Emilia prompts + real ASR corpora on the user stream + 0.9–1.1x speed perturb + RIR/noise augmentation (openSLR RIRs, MUSAN) applied only to user rows.

**Mandatory QC loop on all synthetic audio:** whisper-large-v3 WER filter (reject >5%), speaker-similarity gate vs the prompt voice, and an AudioSet event classifier (e.g. MIT/ast-finetuned-audioset) to catch VibeVoice's documented spontaneous background-music insertions. Optionally restore/clean real corpora with Open-Miipher-2 or resemble-enhance before Mimi encoding (higher-leverage than post-enhancement). Total synthesis budget ≈ 1,200–2,000 GPU-hours.

## 4. Naturalness training additions

Adopt NOW (cheap, verified):
1. **Moshi loss settings**: `semantic_loss_weight=100` (field already exists), flat 1-frame acoustic delay (comes with depth mode), 30% text-token masking in pretraining. ~Zero cost; Moshi-documented quality gains.
2. **CSM 1/16 depth amortization** — required to make 32-cb training affordable; "no perceivable difference" per Sesame.
3. **CFG-ready conditioning dropout**: drop text/context conditioning with p=0.1 during training; decode with logit interpolation γ≈2–2.5. Largest single verified inference-time gain (Koel-TTS: CER 2.56→0.69, SSIM +0.05) for 2x decode FLOPs (batchable; still real-time given our headroom).
4. **Paralinguistic tags** — Orpheus-style `<laugh> <sigh> <chuckle> <gasp> <cough>` mapped into our 64 reserved special ids; training data supplied by SoulX/VibeVoice tag-rich synthesis. Near-zero architecture cost.
5. **Decode hygiene**: VALL-E-2 repetition-aware sampling; split sampling params (cb0 temp ~0.7/top-k 50 vs acoustic 0.8/250).
6. **Audio-context conditioning** — CSM's actual naturalness breakthrough: ensure Stage C grids carry full multi-turn audio+text history (our format already supports it; make multi-turn the default, not the exception).

Adopt LATER (post-SFT milestones):
7. **DPO-on-speech (MPO/Koel recipe)** after Stage C: sample 4–8 candidates per prompt over ~100 h of dialogue text; Pareto-rank by whisper-large-v3 WER + ECAPA/WavLM speaker-sim + log-F0 RMSE + DistillMOS; DPO+CE, lr 1e-6. Never rank by ground-truth-preferred (Koel: degenerates) and never MOS-predictor-alone (RRPO: reward hacking via clicks/plosives). Days on 8 GPUs; expected CER −20–50%, ~2:1 ABX naturalness wins.
8. **Stage D single-voice polish** (300–500 h fixed persona, Section 3) after DPO.
9. **Post-enhancement** (resemble-enhance, AP-BWE 24→48 kHz): A/B only, ship only if it wins; modest and artifact-prone on already-clean 24 kHz output.

## 5. Evaluation gates

Battery (new `src/omni/eval/`): WER via `openai/whisper-large-v3`; MOS proxy via **Distill-MOS** (pip `distillmos`, arXiv 2502.05356); speaker-sim via `microsoft/wavlm-base-plus-sv` (or speechbrain/spkrec-ecapa-voxceleb); F0 statistics (RMSE, variance) via torchaudio/pyworld; **TTSDS2** (pip `ttsds`) as the release-gate metric — the only objective metric with Spearman >0.5 vs humans in every domain; UTMOS explicitly demoted (r≈0.15 on modern speech). Fixed 100-utterance dev battery, logged every `eval_every`.

- **Gate 0 (codec swap sanity, week 1):** Mimi-32 resynthesis of the dev set: DistillMOS ≥ 3.8, WER delta vs ground truth ≤ +0.5 abs. This number is the ceiling all later gates normalize against.
- **Gate B (post Stage B):** TTS-mode WER ≤ 8%; DistillMOS ≥ 3.2; per-codebook teacher-forced NLL trending below the 8-cb baseline at matched steps.
- **Gate C (post Stage C S2S):** WER ≤ 5%; DistillMOS ≥ 90% of Gate-0 ceiling (≈3.5+); intra-conversation speaker-sim ≥ 0.75 (voice stability); turn-gap histogram within human range (from CANDOR stats).
- **Gate D (post DPO + persona polish, release gate):** DistillMOS ≥ 4.0; TTSDS2 within 5 points of Gate-0 ceiling; WER ≤ 3% and **not regressed vs pre-DPO** (anti-reward-hacking check plus artifact spot-listen); human CMOS vs a Kokoro-rendered baseline of the same dialogues: **must win ≥ +0.5** — this operationalizes the user's rejection of Kokoro quality.

## 6. Code impact

See the ordered `code_impact` list. Smallest set delivering the jump: quality preset + depth amortization + 32-cb shard re-encode + new TTS backends + CFG/sampling + eval module. The fallback renderer, DPO trainer, and asymmetric duplex embeddings are staged behind gates.

# Decisions

## Primary audio output path
CHOICE: Mimi @ 32 codebooks (kyutai/mimi, num_quantizers=32, 4.4 kbps) generated by the existing CSM-style depth transformer (flat delay mode), decoded by Mimi's own decoder; new 'quality' preset with depth_d_model=1024, depth_n_layers=4, depth_n_heads=8 and CSM 1/16-frame training amortization
RATIONALE: Nearly free given our implemented depth path and undelayed shards; preserves 80 ms streaming and full duplex; de-risked at our exact scale by sesame/csm-1b (Apache-2.0) and kyutai/tts-1.6b-en_fr (CC-BY-4.0); lifts the reconstruction ceiling from PESQ 2.07 ('muffled, crackly' per Kyutai) to PESQ 3.18/ESTOI 0.910 with zero new pretrained components or licenses
ALT: Semantic-token LM + pretrained flow-matching renderer (output-architectures report's primary): higher quality-per-training-hour but adds 0.5-1 s latency, breaks duplex streaming, and requires either abandoning the Mimi token space (GLM-4-Voice decoder) or weeks training our own renderer — kept as the gated fallback; Qwen3-Omni MTP + causal ConvNet code2wav rejected (requires training a codec/renderer at Qwen scale)

## Fallback audio path
CHOICE: Backbone emits only Mimi cb0 (12.5 Hz semantic row already in every shard); train a ~150-200M chunk-aware flow-matching mel renderer (CosyVoice2 Apache-2.0 / Chatterbox S3Gen MIT code as scaffold) + nvidia/bigvgan_v2_24khz_100band_256x (MIT) vocoder, conditioned on a fixed persona reference
RATIONALE: Kimi-Audio proves 12.5 Hz semantic-only -> FM+BigVGAN rendering works; Voxtral TTS shows FM beats a deep depth transformer; keeps shards/codec/tokenizer unchanged since cb0 is grid row 1; only triggered if Gate C/D fails on acoustics
ALT: Off-the-shelf renderers (zai-org/glm-4-voice-decoder, Kimi detokenizer, Step-Audio-2-mini) rejected: none conditions on Mimi tokens, so adopting one means retokenizing the entire data/model stack

## Delay mode at 32 codebooks
CHOICE: Flat delay + depth transformer only; MusicGen stagger deprecated for anything above 8 codebooks (kept for tiny/base back-compat)
RATIONALE: A 32-codebook stagger adds 31 frames = 2.48 s intrinsic latency at 12.5 Hz; depth mode costs 32 small sequential steps (~10-15 ms) inside the 80 ms frame budget and cuts first-audio latency from ~640 ms to ~80-240 ms
ALT: Keeping stagger with 16 cb (1.28 s latency) rejected; parallel heads at 32 cb without stagger rejected (intra-frame conditional independence)

## Model scale / preset table
CHOICE: Add 'quality' preset ~1.0B total (base d1536/L26 backbone + 32 cb embeddings + d1024/L4 depth ~= 0.95-1.0B); switch tiny and small defaults to flat+depth (small gets 32 cb) so CI/ablations exercise the shipped path; keep base as 8-cb legacy
RATIONALE: CSM-1B (1B+100M) is the smallest system with demonstrated near-human naturalness; LFM2-Audio-1.5B on Mimi-8 sounding 'fine, not human' is direct evidence against our current 8-cb setup; reusing validated base shapes minimizes risk
ALT: A larger d2048/L20 (~1.6B) 'quality-xl' deferred — at our data scale (~30k h) more params won't buy naturalness; 3B-class (Orpheus/Higgs) out of single-node from-scratch budget

## Honest minimum for natural output
CHOICE: State plainly: >=1B params AND >=100k h speech for consistently natural acoustic generation from the LM itself; our committed ~30-35k h plan targets 'clearly natural, far above Kokoro' via renderer-grade synthetic data distillation, context conditioning, and DPO — not human-indistinguishability
RATIONALE: Data ladder across all reports: Moshi 7M h = flat, CSM 1M h = near-human, nobody demonstrates human-like acoustics under ~1M h without delegating to a pretrained renderer; overpromising here is the main expectation risk
ALT: Claiming CSM parity at 30k h rejected as dishonest; scaling to 100k+ h via Emilia-YODAS/YODAS2 documented as the no-architecture-change growth path

## Bulk synthesis TTS (Kokoro replacement)
CHOICE: microsoft/VibeVoice-1.5B (MIT) for multi-turn two-speaker dialogues (~2,500 h at RTF 0.2-0.5 ~= 500-1,250 H100-h) + FunAudioLLM/Fun-CosyVoice3-0.5B-2512 (Apache-2.0) for single turns, instruct emotion, and Emilia-cloned user-voice diversity + Soul-AILab/SoulX-Podcast-1.7B (Apache-2.0) for paralinguistic-tag data
RATIONALE: All MIT/Apache with dialogue-native long-form stability and top open subjective quality (VibeVoice MOS 3.76 > ElevenLabs v3 3.40); keeps everything-synthesizable; CosyVoice3 zero-shot cloning gives thousands of user-side voices
ALT: Higgs Audio v2 rejected (license dispute, org pivoted successors to non-commercial); Voxtral TTS and F5-TTS rejected (CC-BY-NC); Index-TTS2 rejected (non-commercial); Dia rejected (unstable voice identity); Chatterbox kept as optional spice (MIT, but watermark + hallucination rate)

## Fixed-persona polish TTS
CHOICE: VibeVoice-Large 7B MIT mirror (aoi-ot/VibeVoice-Large) with one curated voice prompt for 300-500 h, plus stepfun-ai/Step-Audio-EditX (Apache-2.0) emotion/style editing passes (~400-800 GPU-h)
RATIONALE: Executes Moshi's single-actor-many-styles final-finetune trick synthetically; EditX is the only open tool for iterative paralinguistic editing of existing audio
ALT: Hiring a voice actor (Moshi's actual method) deferred on cost; official microsoft 7B unavailable (pulled), hence the MIT mirror with its supply-chain risk noted

## Real expressive corpora
CHOICE: amphion/Emilia-Dataset YODAS split (CC-BY-4.0, filtered conversational-EN 10-20k h) + kyutai/DailyTalkContiguous (CC-BY-SA-4.0, stereo duplex) added to existing ASR sets; Expresso (CC-BY-NC) demoted to evaluation-only; CANDOR/Fisher used solely for turn-timing statistics
RATIONALE: Emilia-YODAS is the only large commercially-licensed spontaneous-speech corpus; DailyTalkContiguous matches our dual-stream duplex format natively; keeps the license posture clean
ALT: Original Emilia 101k h (CC-BY-NC) rejected for training; Switchboard/Fisher audio rejected (8 kHz, fees); YODAS2 500k h noted as the later 100k+ h scaling reservoir

## Naturalness techniques: adopt now
CHOICE: (1) semantic_loss_weight=100 + 30% text masking (Moshi), (2) CSM 1/16 depth amortization, (3) CFG conditioning-dropout p=0.1 in training with gamma~2-2.5 decode, (4) Orpheus-style paralinguistic tags in the 64 reserved special ids, (5) repetition-aware sampling + split cb0/acoustic temperatures, (6) multi-turn audio-context grids by default
RATIONALE: Each is verified with published effect sizes (CFG: CER 2.56->0.69; amortization: no perceivable loss; context conditioning: CSM's stated breakthrough) at near-zero or amortized cost, and none requires new pretrained components
ALT: Style captions (Parler/CapSpeech) deferred — tags cover the assistant persona use-case at far lower data cost

## Naturalness techniques: adopt later
CHOICE: Post-Stage-C DPO (MPO/Koel recipe: 4-8 self-samples per prompt over ~100 h, Pareto-ranked by whisper WER + WavLM/ECAPA sim + log-F0 RMSE + DistillMOS, DPO+CE lr 1e-6), then Stage-D single-voice polish; post-enhancement (resemble-enhance / AP-BWE) only if it wins A/B
RATIONALE: DPO is the biggest proven post-SFT win at exactly our scale (Koel 380M-1.1B; MPO on one A6000) but needs a converged SFT model first; multi-metric rewards + KL anchor guard against RRPO-documented reward hacking (mouth-click artifacts fooling MOS predictors)
ALT: Ground-truth-as-preferred DPO (SpeechAlign) rejected (Koel: model degeneration); UTMOS-only rewards rejected (r~0.15 with humans, hackable); GRPO noted as a follow-on once DPO plateaus

## Evaluation gates
CHOICE: Battery = openai/whisper-large-v3 WER + Distill-MOS (pip distillmos) + microsoft/wavlm-base-plus-sv speaker-sim + F0 stats as CI regression metrics; TTSDS2 (pip ttsds) + small human CMOS as release gates; four gates: Gate 0 codec-ceiling calibration, Gate B (WER<=8, DMOS>=3.2), Gate C (WER<=5, DMOS>=90% of ceiling, intra-conv sim>=0.75), Gate D release (DMOS>=4.0, TTSDS2 within 5 pts of ceiling, WER<=3 non-regressed, human CMOS vs Kokoro baseline >= +0.5)
RATIONALE: TTSDS2 is the only objective metric with Spearman>0.5 vs humans in every domain; UTMOS-class predictors collapse (r~0.15) on modern conversational speech, so they gate nothing; the explicit beat-Kokoro CMOS gate operationalizes the user's rejection
ALT: PESQ/ViSQOL rejected as gates (adversarial codec decoders score poorly while sounding good, per Moshi paper); UTMOS kept only as a trend line

## Duplex parameter cost at 32 cb
CHOICE: Keep user-side (input-only) audio rows at 8 codebooks while assistant output uses 32 (shards store 32 rows; the model embeds a prefix per group)
RATIONALE: User audio is never a loss target; speech understanding lives in the low/semantic codebooks, so 32-cb user embeddings add ~75M params for no benefit
ALT: Symmetric 32/32 kept as a config fallback if the asymmetric grid change proves invasive to the frozen streams.py/grids.py contracts

# Implementation queue (ordered)

1. src/omni/config.py: add 'quality' preset (base backbone d1536/L26/ctx2048, n_codebooks=32, audio_delay_mode='flat', use_depth=True, depth_d_model=1024, depth_n_layers=4, depth_n_heads=8, semantic_loss_weight=100); new ModelConfig fields depth_frame_subsample (default 16, 1=off) and cond_dropout_p (default 0.0, quality=0.1); new TrainConfig field text_mask_p (default 0.0, pretrain=0.3); SamplingConfig fields cfg_gamma, cb0_temperature/cb0_top_k, repetition_window/threshold; flip tiny default to flat+depth and small to 32cb+depth (d512/L2)
2. src/omni/model/omni.py + model/layers.py: CSM 1/16 compute amortization in the depth training path — full 32-position teacher-forced depth pass only on a random depth_frame_subsample subset of B*T positions, cb0 logits on every frame (position-0-only depth call or CSM-style backbone cb0 head); loss() masks depth codebooks 1..31 to the subsampled positions. Without this the 32-cb quality preset is not trainable at 8-cb cost
3. scripts/prepare_data.py + src/omni/data/prepare.py + audio/codec.py: always encode shards with num_quantizers=32 (RVQ prefix property keeps rows 0..7 identical to today's shards, so tooling is unchanged); shard grids become [T, 1+32] (uint16 still fits, codes<2048); model/dataset reads the first 1+n_codebooks rows per config; one GPU re-encode pass over existing corpora; FakeCodec path already supports 32
4. src/omni/data/synthesize.py: add VibeVoiceTTS (microsoft/VibeVoice-1.5B + aoi-ot/VibeVoice-Large), CosyVoice3TTS (FunAudioLLM/Fun-CosyVoice3-0.5B-2512, with zero-shot voice-prompt cloning arg), SoulXPodcastTTS (Soul-AILab/SoulX-Podcast-1.7B) backends behind the existing TTSBackend abc; demote KokoroTTS to deprecated; keep SineTTS for offline CI; extend build_tts registry
5. scripts/synthesize_data.py: mandatory QC pipeline for synthetic audio — whisper-large-v3 WER filter (reject >5%), speaker-sim gate vs prompt voice, AudioSet event classifier to catch VibeVoice background-music insertions; add --persona mode (fixed voice prompt) for the Stage-D polish set and paralinguistic-tag passthrough into the text stream
6. src/omni/text/tokenizer.py + streams.py: map Orpheus-style paralinguistic tags (<laugh>, <sigh>, <chuckle>, <gasp>, <cough>, ...) into the 64 reserved special ids; ensure they flow through grid building and are loss-active on the text stream
7. src/omni/train/loop.py + data collate: CFG conditioning dropout (with p=cond_dropout_p replace text rows with <text_pad>/drop context so unconditional decoding is trained) and text_mask_p masking during pretraining stages
8. src/omni/infer/generate.py + infer/duplex.py + chat.py: CFG dual-batch decode (conditional + unconditional rows, logit interpolation gamma=cfg_gamma), VALL-E-2 repetition-aware sampling on audio streams, split cb0 vs acoustic sampling params; verify 32-step depth sampling stays under the 80 ms frame budget in scripts/benchmark.py
9. NEW src/omni/eval/ + scripts/evaluate.py: fixed 100-utterance battery computing whisper-large-v3 WER, Distill-MOS, wavlm-base-plus-sv speaker-sim, F0 RMSE/variance, plus TTSDS2 wrapper for release gates; Gate-0 calibration mode (Mimi-32 resynthesis ceiling); wire a cheap 16-prompt subset into Trainer eval_every logging
10. NEW src/omni/train/dpo.py (post-Stage-C milestone): candidate sampling (4-8 per prompt), multi-metric Pareto ranking (WER + speaker-sim + log-F0 RMSE + DistillMOS), DPO+CE loss at lr 1e-6 with KL/CE anchor; reuses eval/ scorers
11. Optional (behind Gate C): asymmetric duplex embeddings — user_audio_embs stay at 8 codebooks while assistant group uses 32 (saves ~75M params); touches streams.py/grids.py frozen contracts, so schedule with a full test pass and INTERFACES.md version bump
12. Fallback track (only if Gate C/D fails): new renderer/ module training a ~150-200M flow-matching mel model conditioned on Mimi cb0 (CosyVoice2/Chatterbox S3Gen scaffold) + BigVGAN-v2 vocoder, and an infer path that bypasses Mimi decode; shards already contain cb0, so no data change
# Risks

- Reports disagreed on the primary path: output-architectures recommended semantic tokens + pretrained flow-matching renderer as best quality-per-training-hour at <=1B; we chose Mimi-32 + depth for integration cost, duplex streaming, and license cleanliness. The residual risk is exactly the LFM2-Audio datapoint (Mimi-RVQ output at ~1.5B sounds 'fine, not human'); mitigations are renderer-grade synthetic data distillation and DPO, with the cb0+renderer fallback pre-specified behind Gate C/D
- Data-scale honesty: CSM needed ~1M h for near-human naturalness; our plan is ~30-35k h. Expect a large jump over Kokoro-flat output, not human parity — if stakeholder expectation is 'Sesame-level', the only cures are 100k+ h (Emilia-YODAS/YODAS2 scaling) or the renderer fallback
- VibeVoice supply chain: Microsoft pulled the official 7B after misuse reports; the plan depends on an MIT mirror (aoi-ot/VibeVoice-Large) for the polish stage and on 1.5B for bulk. Mirror could vanish or be tainted — snapshot weights immediately; CosyVoice3 is the fully-safe Apache substitute at some quality cost. Also VibeVoice occasionally inserts background music — the QC classifier gate is mandatory, not optional
- MOS-predictor unreliability: UTMOS-class metrics correlate ~0.15 with humans on modern conversational speech and DPO rewards are hackable (RRPO documented click/plosive artifacts that fool reward models). All gates therefore rest on TTSDS2 + small human CMOS; if those are skipped for speed, the objective battery can green-light a bad-sounding model
- 32-cb cost growth: +~0.2B params (embeddings + depth heads dominate), 4x audio-token storage in shards, one full re-encode GPU pass, and 32 sequential depth steps per frame at inference — benchmark on the target GPU early; if the frame budget is violated at batch>1 serving, drop to 16 codebooks (PESQ 2.67, still a big step over 2.07) as the intermediate setting
- Depth-path code maturity: the v2 depth transformer and duplex code never received an independent multi-agent review (implementation agents died mid-run); this redesign makes depth the primary path and touches frozen contract files (config.py, streams.py, grids.py) — run the deferred review of model/omni.py depth path and infer/duplex.py before committing quality-preset compute
- License hygiene on data: Emilia must be the YODAS split only (CC-BY-4.0; the original 101k h is NC), Expresso is NC (eval only), Higgs Audio v2's Apache status is disputed — the recipe deliberately excludes it; keep a per-source license manifest in the shard index to make audits trivial
- CFG at inference doubles decode FLOPs and interacts untested with full-duplex dual-stream decoding; if the duplex loop cannot absorb 2x, restrict CFG to half-duplex assistant turns (where it matters most) and ship duplex without it
- Fixed-persona polish overfit: 300-500 h of one synthetic voice can collapse prosodic diversity learned earlier (Moshi's flat voice is partly this); keep Stage D short, mix 10-20% multi-voice replay, and gate on TTSDS2 not regressing vs post-DPO checkpoint
## Deviations applied when landing this doc

- The synthesis table flips the tiny preset to flat+depth by default; we keep tiny on stagger so the v1 contract tests keep exercising both modes (test_depth.py covers depth on tiny-derived configs). small and the new quality preset carry the depth default.
- config knobs for not-yet-implemented mechanisms (depth_frame_subsample, cond_dropout_p, text_mask_p, cfg_gamma) land together with their implementations (queue items 2, 7, 8), not before.
