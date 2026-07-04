# Omni-S2S v5: Voice Cloning (reference-pinned cross-lingual identity)

> Produced 2026-07-02 from a 3-agent paper survey (raw reports: docs/research/voice-conditioning.md, omni-refresh-2026.md, crosslingual-voice-data.md). Amends DESIGN.md §1/§4/§6, DESIGN_V3 §3-5, DESIGN_V4 §2-4. Frozen-core touches: streams ids 49..51 + one additive grids builder (voice segment).

# Omni-S2S v5: Voice-Cloning Amendment (reference-pinned speaker identity)

> Merges 3 research reports (voice-conditioning, omni-refresh, crosslingual-voice-data). Amends DESIGN.md §1/§4/§6, DESIGN_V3 §3-5, DESIGN_V4 §2-4. Frozen-core touches: `streams.py` (3 additive names at ids 49..51) and — first since the freeze — `grids.py` (one additive builder method + a default-None kwarg). `config.py`/`model/` are NOT touched: the mechanism adds **zero parameters and zero new modules**.

## 1. Mechanism: transcript-free Mimi-token prompt prefix (PersonaPlex-style)

A `<voice>`-delimited segment of raw Mimi codes from the reference speaker is prefilled into the grid directly after `<bos>`, before the task tag; the model learns "speak in the voice heard here." All three reports converge on this over a speaker-encoder embedding or hybrid:

- **Proven at exactly our stack**: PersonaPlex (NVIDIA, 2602.06053) is Moshi-class frame-grid over Mimi with a voice-prompt segment first (prefix-cacheable), user channel filled during the prompt, loss masked over the whole prompt — SSIM 0.57 vs Moshi 0.10, stable across full-duplex multi-turn. Chroma 1.0 replicates at 1B (TTFT 147 ms). Step-Audio 2, CosyVoice 2, MaskGCT, Seed-TTS all ship prefix ICL.
- **Negative evidence against embeddings**: CosyVoice 2 *removed* the speaker embedding from its LM (it leaks language/paralanguage, "harms prosody naturalness and cross-lingual capability"); Kyutai's own prefix-cloning scores 74.9% — competitive with their withheld cross-attention encoder; Vevo-style flow-stage injection is unavailable to us (Mimi is frozen, no renderer stage).
- **Real-time by construction**: the reference is a one-time KV prefill; per-frame decode cost is untouched; the depth transformer never runs during prefill (reference tokens are known).

Exact layout, masking, and pairing are in the `grid_layout` section (binding). Summary: markers `<voice>`=49, `<voice_end>`=50 (`<accent_keep>`=51 named-but-untrained; 52..63 stay free); segment = `<voice>` + R reference frames (transcript-free: TEXT_PAD under the audio) + `<voice_end>`, channel ASSISTANT, **loss masked on every row over the whole segment**; placed between `<bos>` and the task tag so one `[<bos>+voice]` KV prefix serves every task, turn, and session for a given voice. Reference length: **10 s = 125 frames at inference (127 grid cols with markers); training samples 3–20 s** (VALL-E 2: SIM 0.447→0.558 from 3 s→10 s; Mega-TTS 2 flattens past 10 s).

## 2. Identity x style x language (disentanglement recipe)

Voice identity (acoustic prefix) and our v4 control tags (`<lang_XX>`, `<emo_pcv>/<emo_rsp>` + 12 classes) are **orthogonal by construction only if training decorrelates them** — ControlSpeech documents prompt-prosody vs style-tag conflicts; IndexTTS2 proves the factorization trains cleanly in AR codec LMs. Recipe:

1. **Same voice x all styles**: each seed voice appears with all 12 emotion tags, all intensities, and all 5 languages; references for a voice are themselves drawn across emotional states.
2. **Conflict pairs (20–30% of voice-conditioned data)**: reference prosody contradicts `<emo_rsp>` (angry reference, `<calm>` tag) — tag must win, prefix pins only timbre.
3. **Cross-lingual pairs (50%)**: reference language ≠ target language, all 20 directed pairs of {en,fr,de,zh,tr} — this is what actually teaches cross-lingual identity (X-Voice stage-2 recipe; PersonaPlex's open gap is English-only).
4. **Unconditioned path retained (≈50% of speech-gen grids carry no segment)**: default-persona behavior (Stage D) survives; folds into the v3 `cond_dropout_p` CFG training, and CFG decode over voice-conditioning can strengthen identity (Koel-TTS SSIM +0.05) as a free option.

**Accent policy: keep identity, target-language phonetics.** The active `<lang_XX>` tag plus native-accented synthetic targets (our generators are native in each target language) push native accent; VALL-E X quantifies the tradeoff (LID: accent 2.98→4.10 for −0.04 ASV) — we take native accent as default. `<accent_keep>` (id 51) is reserved for the opposite mode but **not trained in v1** (accented-L2 training data is hard to synthesize; X-Voice shows textual LID alone can leave residual L1 accent — if the accent probe fails, the fix is more cross-lingual pairs, not architecture).

Ordering is unchanged: the per-turn monologue prefix `<lang_XX> <emo_pcv> P <emo_rsp> R` stays exactly as v4; the voice segment is global (once per grid), not per-turn.

## 3. Data recipe (synthetic-first, real-calibrated)

**Seed voices**: 500 commercially-clean speakers v1 (LibriSpeech/MLS/Emilia-YODAS prompts, ≥10 s clean reference each); scale to 2–5k voices for the release run via the existing Emilia-prompt cloning path (PersonaPlex used 26k voice samples — 500 validates the mechanism, not the ceiling).

**Generators (all already in `synthesize.py` except dots.tts)**: Fun-CosyVoice3-0.5B-2512 (Apache-2.0) for en/fr/de/zh incl. `<|endofprompt|>` instruct emotion — its zero-shot SIM (0.718–0.780 WavLM-large-ft) is at human level, so it is a valid teacher; **Chatterbox Multilingual V3 (MIT) for ALL Turkish targets** (only open model covering all 5 languages) and as second-source diversity; dots.tts (Apache-2.0, best open multilingual SIM 83.9) evaluated as third source; VibeVoice en/zh long-form only. Per speaker: 5 langs × 40–60 utts × ~9 s ≈ 2.5 h → **~1,000 h kept voice-conditioned data** (synthesize 1,500–2,000 h raw for 30–50% QC rejection) ≈ **200–500 GPU-h synthesis + 50–100 GPU-h QC ≈ 2–3 days on the 8-GPU node, <$1.5k**.

**Real same-speaker multilingual data exists and we use it two ways**: TidyVoice (ICASSP 2026, CC-BY-4.0) curates ~4,500 genuine Common Voice polyglots (81 langs, ~91% verified same-speaker) — (a) a small real fine-tune slice (tens of hours) against synthetic-clone circularity, (b) **Tidy-X cross-language same-speaker trials calibrate our per-language-pair SIM thresholds**. FLEURS/VoxPopuli/CSS10 are NOT same-speaker (interpreters/different speakers) — eval prompts only.

**QC per clip**: whisper-large-v3 WER <10–15% (per-language), LID = target language, DNSMOS ≥ 3, dual speaker-sim gate SIM(gen, ref) ≥ 0.60 same-lang / ≥ 0.55 cross-lingual (WavLM-large-ft primary, ERes2Net secondary), plus the v3 AudioSet event filter.

**Stage mixture**: Stage B — ~10% of TTS positions become voice-conditioned TTS (teaches the mechanism early, cheap); Stage C — ~50% of s2s/tts grids voice-conditioned with the pairing scheme above, ~10% of ASR grids carry an (ignored) voice segment for invariance; Stage D unchanged (default persona trains on the unconditioned path).

## 4. Real-time check (numbers to verify in benchmark)

At quality scale (~1.0B backbone, 12.5 Hz):
- **One-time prefill**: 127 positions ≈ 2·1e9·127 ≈ 0.25 TFLOP ≈ **2–5 ms compute on A100-class, <20 ms wall-clock**; the depth transformer is skipped during prefill (reference codes known — embedding sums only). Gate: prefill(127) < 40 ms measured.
- **Per-frame decode unchanged**: 35–45 ms/frame budget intact; KV grows by 127 of 2048 positions (6.2% of context ≈ 10 s of dialogue capacity; ≈ 52 KB/pos → **6.8 MB per session**). Gate: decode step time with a 127-position prefix within 1 ms of without.
- **Session KV reuse**: RoPE positions are absolute and the segment sits at fixed positions 0..R+1, so the `[<bos>+voice]` KV block is exactly reusable across turns (cache simply persists and appends) and across sessions (hash-keyed by reference audio, kyutai/vLLM-style). v1 correctness path: re-prefill per call (<20 ms — honestly fine); v1.1: cached prefix copy for concurrent serving.
- **TTFA**: flat delay keeps first audio ~100–250 ms after end-of-user-speech; cold voice adds <20 ms once per session, warm ≈ 0. **End-to-end TTFA target < 500 ms retained** (OpenAI Realtime-competitive). Duplex tick unchanged ≤ 45 ms.

## 5. Eval gates (Gate-V, after voice-SFT, before DPO)

**Scale discipline**: all gate numbers are on the **WavLM-large fine-tuned SV scale (Seed-TTS-eval "SIM-o", BytedanceSpeech/seed-tts-eval)**, human same-speaker ≈ 0.73–0.76. PersonaPlex's 0.57 is WavLM-TDNN — never cross-compare. ERes2Net (CV3-Eval protocol) is the mandatory second verifier; both eval-only (no license exposure in the product).

- **Per-language 5×5 matrix** (ref-lang × tgt-lang, ≥20 held-out voices × ≥10 utterances/cell): pass = every cell mean SIM ≥ 0.60 (same-lang diagonal) / ≥ 0.55 (cross-lingual), hard floor 0.50; thresholds calibrated per pair on Tidy-X with language-dependent score normalization (cross-lingual verifier drift is documented). Also per cell: WER (whisper-large-v3) and LID correctness not regressed vs unconditioned decode.
- **Identity-stability-across-emotion**: per voice, {12 emotion tags + neutral} × fixed text; emotion-induced SIM drop vs neutral ≤ 0.05; plus an emotion-realization check (SER judge reads the TAG's emotion, not the reference's) on conflict prompts.
- **Long-session drift** (2604.06327 failure mode): ≥2 min duplex sessions, sliding 10 s windows; min window SIM ≥ session mean − 0.05, no monotonic decline.
- **Teacher-ceiling tracking**: SIM(model, ref) within 0.05 of SIM(teacher-TTS, ref) on matched prompts — measures captured headroom.
- **Accent probe**: CommonAccent-style classifier / ABX; accent transfer is the dominant known failure.
- Existing v3 gates unchanged; DPO reuses its WavLM sim term ranked against the REFERENCE for voice-conditioned prompts. Expected landing zone at our scale: 0.60–0.70 in-language, 0.05–0.10 cross-lingual penalty.

**Watermarking/consent (binding)**: AudioSeal (MIT, weights included) on all product output; reference audio requires explicit consent (treat cloned voices as biometric data); note kyutai's finding that Mimi encode/decode strips existing watermarks — inbound watermark detection cannot gate references, so consent is procedural, not technical. We deliberately ship what OpenAI restricts to presets ("prevent impersonation") — a product-policy decision to record, not bury.

## 6. Rollout

1. streams/grids additive edits + tests (CI, FakeCodec, tiny refs) — 1–2 days.
2. Generator/duplex/chat `--voice` + benchmark — 2–3 days.
3. Pair synthesis pipeline + QC — pipeline days, ~2–3 GPU-days.
4. Voice-SFT on small preset → Gate-V small → quality run → Gate-V full.
Disagreements resolved and residual uncertainty recorded under risks.
# Grid layout (binding)

MARKER TOKENS (streams.py; reserved ids 49..63 -> spend 3, ids 52..63 stay free):
  49 <voice>        voice-reference segment start (text stream)
  50 <voice_end>    voice-reference segment end (text stream)
  51 <accent_keep>  named + reserved, UNTRAINED in v1 (keep-reference-accent switch; VALL-E X LID tradeoff)

VOICE SEGMENT (UNDELAYED grid; R = reference frames; inference default R=125 (10 s), training R ~ U[37..250] (3..20 s), cap 375):
  col:            0       1         2 .. R              R+1
  text row 0:     <bos>   <voice>   TEXT_PAD ...        <voice_end>
  audio rows 1..n_q: APAD  r_0       r_1 .. r_{R-1}      APAD          (raw Mimi codes of the reference, ASSISTANT group)
  channel:        ASST    ASST      ASST ...            ASST
  loss_mask:      FALSE on ALL rows over cols 0..R+1 (PersonaPlex whole-prompt masking; <bos> was already unmasked in asr/tts/s2s)
  Transcript-free: no reference transcript anywhere (kills the parroting channel; X-Voice/PersonaPlex). Reference = random chunk of a DIFFERENT utterance by the target speaker (never the target itself, never continuation-only) — kills content/duration leakage.
  Position: BETWEEN <bos> and the task tag, so the [<bos>+segment] KV prefix (absolute RoPE positions 0..R+1) is identical across tasks/turns/sessions for one voice -> one cached prefill serves everything.

PER TASK ([V] = cols 1..R+1 above; layouts after [V] are byte-identical to today):
  tts    : <bos> [V] <tts> [assistant speech seg] <eos>
  s2s    : <bos> [V] <s2s> ([user seg, ch=USER] <end_of_turn> [assistant speech seg]) x N <eos>   — ONE segment pins all N turns
  asr    : default NO segment (text-only output); ~10% of TRAINING asr grids get <bos> [V] <asr> ... as an invariance slice so a session cache prefix is never OOD
  textlm/alm/ser: never carry a segment
  duplex (S = 1+2*n_q): col 0 = <bos>; cols 1..R+1 = [V] with reference codes on ASSISTANT rows 1..n_q and USER rows n_q+1..2n_q = AUDIO_PAD throughout the segment (in-distribution "not speaking" filler — replaces PersonaPlex's 440 Hz sine; our duplex data already uses APAD for non-speech); conversation frame f moves from col f+1 to col f+R+2; <eos> last col. loss_mask: text + assistant rows FALSE over cols 0..R+1 (assistant-row loss resumes ON from col R+2 — "emitting APAD while listening" is still learned); user rows FALSE always. Channel CHANNEL_ASSISTANT throughout (unchanged).
  inference prompts: build_tts_prompt / build_s2s_prompt / build_asr_prompt gain the same optional segment; prompt_forced_text is UNCHANGED (<assistant> + tags/text as today; v4 turn_prefix tags stay per-turn, after <assistant>).

DELAY / DEPTH / SHARD INTERACTION: segment audio is ordinary undelayed assistant-row content; apply_delay shifts it k+1 (stagger) or 1 (flat) like any audio — no new delay logic, no shard-format change (version 1 uint16 grids unchanged; segment is just columns with loss_mask False). Loss-masked reference frames contribute nothing to the depth transformer's 1/16 amortized positions or semantic_loss_weight. Budget: prepare must subtract R+2 cols from max_sample_frames fitting (_fit_s2s_turns budget) and generator/duplex capacity checks add R+2 to the delayed-length bound.

TRAINING PAIRING SCHEME (prepare): (a) reference speaker == target speaker, reference utterance != target utterance, random 3-20 s chunk; (b) P(ref_lang != tgt_lang) = 0.5 uniformly over the 20 directed pairs of {en,fr,de,zh,tr}; (c) 20-30% emotion-conflict pairs: reference prosody contradicts the <emo_rsp> tag and the target audio follows the TAG; (d) ~50% of Stage-C speech-generation grids carry a segment, ~10% of Stage-B TTS, ~10% of asr (invariance); unconditioned remainder keeps the default-persona path alive and doubles as the CFG-unconditional branch (cond_dropout_p).
# Decisions

## Conditioning mechanism
CHOICE: Codec-token prompt prefix: transcript-free Mimi-code segment prefilled after <bos> (PersonaPlex-style in-context conditioning); no speaker encoder, no new modules, zero new parameters
RATIONALE: Proven at exactly our architecture (PersonaPlex: Moshi-class frame-grid over Mimi, SSIM 0.57 vs Moshi 0.10 across full-duplex; Chroma 1.0 at 1B); kyutai's prefix cloning (74.9%) is competitive with their dedicated embedding encoder; one-time prefill keeps per-frame decode untouched
ALT: Speaker-encoder embedding (rejected: CosyVoice 2 removed it from the LM because the vector leaks language/paralanguage and harms cross-lingual capability; also adds a trained module + injection point); hybrid Chatterbox/Qwen3-TTS dual path (rejected: second mechanism to train/eval with no evidenced need at our scale)

## Marker tokens and segment position
CHOICE: <voice>=49, <voice_end>=50, <accent_keep>=51 (named, untrained); segment sits BETWEEN <bos> and the task tag; ids 52..63 remain free
RATIONALE: Placing the segment before the task tag makes the [<bos>+voice] KV prefix identical across tts/s2s/duplex and all turns, so one cached prefill (keyed by reference hash) serves the whole session — PersonaPlex places the voice prompt first for exactly this reason; 3 ids spent, 12 kept in reserve
ALT: Segment after the task tag (rejected: cache becomes per-task); audio-stream delimiter tokens like PersonaPlex's custom audio delimiters (rejected: text-stream markers suffice, AUDIO_BOS/EOS stay reserved for their v1 semantics)

## Reference length
CHOICE: 10 s = 125 frames (127 grid cols) at inference; training samples 3-20 s uniformly, cap 30 s
RATIONALE: VALL-E 2: SIM rises monotonically 3s(0.447)->10s(0.558); Mega-TTS 2 flattens past 10 s (0.905@10s vs 0.922@60s); variable training length buys robustness; 127 positions = 6.2% of ctx 2048
ALT: 3 s minimum-viable (kept as supported lower bound, expect ~-0.1 SIM); 30-60 s (rejected as default: diminishing returns, eats dialogue context)

## Loss masking and anti-leakage
CHOICE: Loss masked on ALL rows over the entire segment; transcript-free (TEXT_PAD under reference audio); reference = random 3-20 s chunk of a DIFFERENT same-speaker utterance
RATIONALE: PersonaPlex masks the whole system prompt; VoiceStar shows prompt-region loss exclusion prevents copying; disjoint random chunks kill content/duration leakage; no transcript means no parroting channel and no reference-ASR dependency in the data pipeline
ALT: Continuation-style prompting (higher SIM in benchmarks but trains the wrong disjoint-reference behavior); prompt-transcript conditioning a la VALL-E (rejected: requires transcripts for every reference and reopens leakage)

## Cross-lingual + style disentanglement training pairs
CHOICE: 50% cross-lingual pairs (all 20 directed pairs of en/fr/de/zh/tr), 20-30% emotion-conflict pairs (tag contradicts reference prosody, tag wins), same voice appears in all languages and all 12 emotion tags; ~50% of Stage-C speech-gen grids conditioned, unconditioned rest doubles as the CFG branch
RATIONALE: Cross-lingual same-voice pairs are what actually teach cross-lingual identity (X-Voice stage-2; PersonaPlex left this gap English-only); IndexTTS2 proves timbre/emotion factorization trains in AR codec LMs but ControlSpeech shows it does NOT emerge without decorrelated pairs
ALT: Same-language-only pairing with hope of transfer (rejected: IWSLT 2026 shows persistent cross-lingual identity/intelligibility tension); 100% conditioned data (rejected: kills the default-persona path and CFG-unconditional branch)

## Accent policy
CHOICE: Default = target-language native phonetics driven by the existing <lang_XX> tag plus native-accented synthetic targets; <accent_keep> id reserved but untrained in v1
RATIONALE: VALL-E X: target LID buys native accent (2.98->4.10) for only -0.04 ASV; our synthetic targets are native by construction, which is stronger supervision than textual LID alone (X-Voice's residual-accent caveat)
ALT: X-Voice dual-level LID injection (FiLM + time-level) — rejected for v1 as an architecture change; training <accent_keep> now (rejected: accented-L2 training data is not synthesizable with our stack today)

## Data recipe
CHOICE: Synthetic-first: 500 seed voices (scale 2-5k for release) x 5 languages x 40-60 utts; Fun-CosyVoice3 for en/fr/de/zh (+instruct emotion), Chatterbox Multilingual (MIT) for ALL Turkish, dots.tts evaluated as third source; ~1,000 h kept of 1,500-2,000 h raw; TidyVoice Common-Voice polyglots as small real slice + Tidy-X threshold calibration
RATIONALE: Real same-speaker multilingual corpora are rare (Amazon: 'rarely practical to procure'); CosyVoice3 zero-shot SIM is at human level so it is a valid teacher; total cost 250-600 GPU-h (<$1.5k, 2-3 days on the 8-GPU node) — cheap enough to just do
ALT: Real-only (rejected: tens of hours max after filtering); VibeVoice beyond en/zh (rejected: unstable per maintainers); F5/Fish/Llasa (rejected: NC licenses)

## QC and eval speaker-sim gates
CHOICE: Dual verifiers: WavLM-large-ft SV (Seed-TTS-eval SIM-o scale) primary + ERes2Net (CV3-Eval) secondary; gates SIM >= 0.60 same-language, >= 0.55 cross-lingual, floor 0.50; per-language-pair thresholds calibrated on TidyVoice Tidy-X genuine cross-language trials; emotion-induced SIM drop <= 0.05; long-session drift <= 0.05
RATIONALE: SIM numbers are verifier-dependent (PersonaPlex 0.57 is WavLM-TDNN scale, NOT comparable); WavLM-large-ft human baseline 0.73-0.76 anchors the scale; cross-lingual verifier drift is documented, hence Tidy-X calibration + language-dependent normalization
ALT: Single-verifier gating (rejected: English-biased VoxCeleb-trained models shift scores across languages); adopting PersonaPlex's 0.55-0.65 numbers directly (rejected: wrong scale)

## Session caching / real-time
CHOICE: v1: re-prefill the voice segment per call (<20 ms wall at 1B, depth transformer skipped during prefill); v1.1: hash-keyed KV-prefix reuse across turns and sessions (exact because RoPE positions are absolute and the segment is position-0-anchored); benchmark gates: prefill(127) < 40 ms, decode-step delta < 1 ms, TTFA < 500 ms
RATIONALE: 0.25 TFLOP one-time cost is negligible (2-5 ms compute); 6.8 MB KV per session is trivial; correctness first, caching as serving optimization (prefix caching cuts TTFT ~800->300 ms in production serving)
ALT: Mandatory precomputed voice library kyutai-style (kept as a deployment mode, not a requirement); recompute per turn (subsumed: within a session the cache simply persists)

## Watermarking / consent
CHOICE: AudioSeal (MIT) on all product output; explicit consent required for reference audio (biometric-data posture); record that Mimi encode/decode strips inbound watermarks so consent is procedural, not technical
RATIONALE: Kyutai withheld their voice encoder for exactly this reason; 2026 practice treats cloned voices as biometric; AudioSeal is the standard MIT-licensed marker (flagging tool, not adversarially robust)
ALT: Preset-voices-only like OpenAI (rejected: contradicts the user requirement of arbitrary reference pinning — recorded as a deliberate product-policy divergence); no watermark (rejected)

# Implementation queue (ordered)

1. 1. /Users/kadirnar/projects/omni/src/omni/streams.py (FROZEN — additive edit, needs INTERFACES.md version bump + full test pass): add VOICE=49, VOICE_END=50, ACCENT_KEEP=51 to SPECIAL_TOKENS ('<voice>', '<voice_end>', '<accent_keep>'); RESERVED_SPECIAL_FORMAT fillers now cover only 52..63; ids 0..48 untouched.
2. 2. /Users/kadirnar/projects/omni/tests/test_emotion_i18n.py line 41 (the ONLY existing-test edit): assertion `names[49] == '<reserved_49>'` must become `names[49] == '<voice>'` (and 52 as the first reserved filler); flag in the PR as the frozen-contract ripple.
3. 3. /Users/kadirnar/projects/omni/src/omni/text/tokenizer.py: no code change (specials auto-read from SPECIAL_TOKENS), but the 48k BPE artifact must be retrained/patched so the literals '<voice>'/'<voice_end>'/'<accent_keep>' map to ids 49/50/51 as AddedTokens; refresh the stale ids-0..10 docstring; ByteTokenizer/CI unaffected.
4. 4. /Users/kadirnar/projects/omni/src/omni/grids.py (FROZEN — first post-freeze edit, additive, version bump): new `_GridBuilder.voice_segment(ref_codes)` (col '<voice>'+r_0, TEXT_PAD cols, '<voice_end>' col with APAD audio; channel CHANNEL_ASSISTANT; tmask/amask False throughout) + optional `voice_codes: torch.Tensor|None = None` kwarg on build_tts/build_s2s/build_asr/build_duplex and build_tts_prompt/build_s2s_prompt/build_asr_prompt, inserted between <bos> and task tag (duplex: between <bos> col and frame cols, assistant rows only, user rows APAD, loss resumes at col R+2). Default None = bit-identical grids; all 62+ existing tests pass unmodified.
5. 5. /Users/kadirnar/projects/omni/src/omni/data/prepare.py: voice pairing — new VoicePairBank (speaker -> {lang -> [ref wav/codes]}) + sampler enforcing ref-utterance != target, 3-20 s random chunks, P(cross-lingual)=0.5, 20-30% emotion-conflict pairs; thread voice_codes into build_tts/build_s2s (prepare_asr_tts gains voice_bank+voice_p, prepare_s2s likewise, prepare_duplex prepends the segment to both tracks' timeline); subtract R+2 from the _fit_s2s_turns/max_sample_frames budget when a segment is present. Shard format v1 unchanged.
6. 6. /Users/kadirnar/projects/omni/src/omni/data/synthesize.py: cross-lingual pair generation — per-call `lang` routing on ChatterboxTTS (currently fixed at __init__) and CosyVoice3TTS (en/fr/de/zh) with Chatterbox owning all tr; optional DotsTTS backend (Apache-2.0) behind TTSBackend; a pair-manifest emitter {speaker_id, ref_path, ref_lang, tgt_lang, emotion} consumed by prepare's VoicePairBank.
7. 7. /Users/kadirnar/projects/omni/scripts/synthesize_data.py: QC gate additions — dual speaker-sim gate SIM(gen, ref) via WavLM-large-ft SV (>=0.60 same-lang / >=0.55 cross-lingual, floor 0.50) + ERes2Net second opinion, LID-equals-target check, per-language whisper-large-v3 WER thresholds; keep the v3 DNSMOS/audio-event filters; write per-clip QC scores into the manifest.
8. 8. /Users/kadirnar/projects/omni/src/omni/infer/generate.py: `voice_wav` (or pre-encoded `voice_codes`) kwarg on OmniGenerator.tts/s2s/asr threading into the voice-segment prompt builders — the generate() frame loop needs ZERO changes (segment rides the prompt prefill); add `set_voice(wav, codec)`/session cache: hash-keyed storage of encoded reference codes (v1) and a prefilled [<bos>+voice] KVCache prefix reused across generate calls (v1.1; exact because RoPE positions are absolute and the segment is position-0-anchored).
9. 9. /Users/kadirnar/projects/omni/src/omni/infer/duplex.py: DuplexGenerator gains voice_codes at __init__/reset(): reset prefills the delayed [<bos> <voice> ref <voice_end>] block (R+2 positions, ONE prefill call) instead of the single <bos> column; tick/undelayed-buffer index math offsets by R+2 (frame f at col f+R+2); run_file capacity check becomes t_user + D + R + 2 <= max_frames. CAUTION: this touches the duplex index math that never received its deferred independent review — schedule that review with this change.
10. 10. /Users/kadirnar/projects/omni/src/omni/infer/chat.py (+ scripts/chat.py wrapper): `--voice REF.wav` flag for tts/s2s/duplex (load_wav -> codec.encode -> voice_codes); warn-and-ignore on asr; document that --voice composes with --emotion/--lang (tags keep style authority).
11. 11. /Users/kadirnar/projects/omni/src/omni/optim/perf.py + scripts/benchmark.py: `benchmark_decode(..., voice_frames: int = 0)` and `--voice-frames 127` — report voice-prefill wall ms separately and decode rtf with the prefilled cache; gates: prefill(127) < 40 ms at quality on the target GPU, per-step decode delta < 1 ms vs no prefix, duplex tick <= 45 ms.
12. 12. /Users/kadirnar/projects/omni/src/omni/eval/ (extends the v3/v4 eval queue): cross-lingual speaker-sim 5x5 matrix scorer (WavLM-large-ft + ERes2Net, Tidy-X-calibrated per-pair thresholds), identity-across-emotion battery (SIM drop <= 0.05 + SER-judge tag-vs-reference check), long-session drift probe (sliding 10 s windows), teacher-ceiling comparison; wire into Gate-V.
13. 13. NEW /Users/kadirnar/projects/omni/tests/test_voice.py: voice_segment layout/mask invariants (whole segment loss-False on every row, marker placement, channel ASSISTANT, apply_delay roundtrip with segment, duplex variant user rows APAD + assistant loss resuming at R+2); pairing sampler disjointness + cross-lingual share; OmniGenerator voice_wav smoke on FakeCodec (seeded determinism, prompt-region samples discarded); DuplexGenerator voice-prefix reset + run_file length invariant; chat --voice CLI smoke; conftest may add a voice_cfg fixture — nothing existing changes except the one line in item 2.
# Risks

- Frozen-contract exposure: this amendment makes the FIRST post-freeze edit to grids.py (plus additive streams.py names) and changes one existing test line (test_emotion_i18n.py:41 asserts names[49]=='<reserved_49>'); mishandling the version bump or the additive-default discipline (voice_codes=None must reproduce today's grids bit-for-bit) breaks every downstream module — land items 1-4 as one reviewed PR with the full suite green.
- Cross-lingual identity at our scale is unproven: PersonaPlex is English-only, IWSLT 2026 documents a persistent intelligibility-vs-identity tradeoff, and even Gemini 3.5's card admits voice drift — expect a 0.05-0.15 cross-lingual SIM penalty; if a language-pair cell fails Gate-V the remedy is more directed synthetic pairs for that pair (and longer references), NOT an architecture change.
- Turkish is single-sourced and watermarked: Chatterbox is the only open generator covering tr, and every output carries the Perth neural watermark — Turkish voice-clone training data is 100% watermarked audio, so watermark artifacts could be learned or depress the tr row/column of the SIM matrix; monitor tr cells separately, keep Chatterbox share bounded outside tr, and evaluate dots.tts as a tr second source when its card confirms coverage.
- Synthetic-clone circularity: training on TTS-cloned pairs caps identity at the teacher's cross-lingual SIM (CosyVoice3 ~0.72-0.78 same-lang, 66-67 cross-lingual on its own scale); the TidyVoice real-polyglot slice mitigates but is tiny (tens of hours) — track the teacher-ceiling eval and treat 'within 0.05 of teacher' as success, not human parity.
- Content/prosody leakage despite masking: transcript-free + disjoint-chunk + whole-segment masking kills parroting, but the model may still over-imitate reference prosody/tempo (fighting <emo_rsp>) — the 20-30% conflict-pair share is mandatory, and the identity-across-emotion gate (SIM drop <= 0.05 AND tag-realization check) is the tripwire; report disagreement recorded: voice-conditioning report suggested 3-30 s training refs, crosslingual-data report built 8-10 s utterances — resolved as 3-20 s train / 10 s infer; revisit upward before any mechanism change if identity is weak.
- SIM scale confusion is an easy silent failure: PersonaPlex 0.57 (WavLM-TDNN), VALL-E X 0.36 (WavLM-TDNN), CosyVoice3 66.9 (ERes2Net percent), our gates 0.60/0.55 (WavLM-large-ft) are FOUR different scales — every logged number must name its verifier; dual-verifier + Tidy-X calibration is the guard.
- Duplex voice prefill rides un-reviewed code: DuplexGenerator's index math gets an R+2 offset on top of a v2 implementation whose deferred independent review (DESIGN_V3 risk) never happened — run that review before spending quality-preset compute on voice-duplex training; the AUDIO_PAD-on-user-rows choice (vs PersonaPlex's 440 Hz sine) is believed in-distribution from prepare_duplex data but must be verified on real duplex checkpoints.
- KV-prefix reuse validity: exact reuse assumes absolute RoPE positions, a fixed R at inference, and no cache eviction/sliding-window changes — any future streaming-truncation or position-remapping work invalidates cached voice prefixes; the v1 re-prefill path (<20 ms) must remain the correctness fallback.
- Context tax and budget interactions: 127 columns per sample cut ~10 s of dialogue capacity at ctx 2048 (12% at small's 1024) and prepare must shrink turn budgets accordingly; ASR/textlm mixtures are untouched, but if voice-conditioned share is set too high, the unconditioned default-persona path (Stage D) and CFG-unconditional branch starve — hold the ~50% split and gate DistillMOS non-regression vs the pre-voice checkpoint.
- Misuse/consent is now a product surface: we deliberately ship zero-shot cloning that OpenAI restricts to presets; AudioSeal marking + explicit-consent capture for references + the documented fact that Mimi strips inbound watermarks (so no technical consent enforcement exists) must ship together with the feature, and the release decision should be recorded at Gate-V, not discovered at launch.