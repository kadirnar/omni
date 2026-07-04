# Single-node multi-GPU training of 100M–2B decoder-only LLMs (state: mid-2026)

**Torch versions.** Current stable is 2.12.1 (2026-06-17). Target torch>=2.8; FSDP1 (`FullyShardedDataParallel`) is deprecated since 2.11 — use FSDP2. Key 2.7–2.9 changes: `fully_shard` became public under `torch.distributed.fsdp` (stable since ~2.6/2.7; old `torch.distributed._composable.fsdp` path still works, torchtitan imports `FSDPModule` from it); torch 2.9 removed support for compiling *through* FSDP2 hooks without graph breaks (compile per-block before sharding, or use `fullgraph=False`); DCP gained HuggingFace safetensors I/O (blog 2025-06-06).

**Framework choice.** torchtitan is the reference from-scratch stack (ezyang, Aug 2025: "fork torchtitan"); OLMo-core "uses compile, FSDP2, selective AC following best practices from torchtitan"; litgpt pretrains with FSDP bf16-mixed + grad accumulation; nanotron is HF 3D-parallelism (overkill ≤2B). DeepSpeed is not used by any of these; HF Accelerate supports FSDP2 via `FullyShardedDataParallelPlugin(fsdp_version=2)` (since v1.6) but adds indirection. For ≤1B, plain DDP fits comfortably (1B fp32 master + Adam ≈16 GB/GPU) — keep it as fallback/CPU path; FSDP2 is the default for headroom and built-in bf16-param/fp32-reduce policy.

**FSDP2 facts (verified in 2.9/2.12 docs).** `from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy, CPUOffloadPolicy`. `fully_shard(module, *, mesh=None, reshard_after_forward=None, mp_policy=..., offload_policy=...)` mutates the module in place (no wrapper; FQNs unchanged; params become dim-0-sharded DTensors). `reshard_after_forward=None` → True for non-root, False for root; True frees params post-forward and re-all-gathers in backward; int reshards to a smaller world size. `MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None, output_dtype=None, cast_forward_inputs=True)`. Apply bottom-up: each transformer block, then root; build optimizer **after** sharding; call `model(x)`, not `model.forward(x)`. torchtitan groups tied embedding+lm_head into one FSDP unit and sets `reshard_after_forward=False` for final norm+lm_head (prefetched immediately anyway).

**Precision.** Standard: fp32 master weights, `MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)` (torchtitan defaults; it sets `cast_forward_inputs=False` and casts inputs itself). No GradScaler with bf16. Under DDP use `torch.autocast("cuda", torch.bfloat16)` with fp32 params/all-reduce. Pure-bf16 (no fp32 copy) is supported by torchtitan (RoPE/logits stay fp32) but at ≤1B prefer fp32 masters.

**Attention/packing.** `F.scaled_dot_product_attention(..., is_causal=True)` dispatches to FlashAttention-2/cuDNN on CUDA and falls back to math on CPU/MPS — portable, no flash-attn build; sufficient ≤8k ctx. For packed sequences with intra-document masking use FlexAttention `BlockMask` document masking (compiled FA-grade kernel) or `flash_attn_varlen_func` with `cu_seqlens`; many pretraining runs skip cross-doc masking entirely (concat+chunk, nanoGPT-style).

**AC + compile ordering.** Per-block `torch.utils.checkpoint` with `use_reentrant=False` (reentrant AC breaks/leaks with FSDP2, e.g. pytorch#169349) or `checkpoint_wrapper`. Order: AC → per-block `torch.compile` (regional compilation: one compile reused across blocks) → `fully_shard` — exactly torchtitan's `parallelize_llama`.

**Optimizer/LR.** `torch.optim.AdamW(lr, betas=(0.9,0.95), weight_decay=0.1, fused=True)` — fused works with DTensor; torchtitan's default implementation is "fused". LR: warmup (~1–5% of steps) + cosine→10% peak is still the safe default; WSD (warmup-stable-decay) matches/beats cosine at equal compute and decouples from horizon — torchtitan's scheduler is WSD-shaped (`warmup_steps=200` default; `decay_type` linear/sqrt/cosine).

**Grad accumulation.** Two correct options: (a) `model.set_requires_gradient_sync(is_last_microbatch)` (FSDPModule method) — skips reduce-scatter until last microbatch, holds unsharded grads (more memory); DDP analog `no_sync()`. (b) torchtitan reduce-scatters every microbatch and uses `set_gradient_divide_factor(1.0)` + token-count loss scaling. `torch.nn.utils.clip_grad_norm_` handles DTensor grads (global norm; `.full_tensor()` to log).

**Checkpointing.** `torch.distributed.checkpoint` (`dcp.save/load/async_save`) + `torch.distributed.checkpoint.state_dict.get_state_dict/set_state_dict` with `StateDictOptions(full_state_dict, cpu_offload, broadcast_from_rank0)`. Consolidated safetensors export: gather full state dict on rank0 and `safetensors.torch.save_file`, or DCP's `HuggingFaceStorageWriter/HuggingFaceStorageReader` (torch.distributed.checkpoint; writer takes `fqn_to_index_mapping`, newer versions add consolidation-to-single-file).

**Data.** Pre-tokenize to uint16/uint32 shards. Simplest/deterministic at this scale: flat .bin/.npy memmap + plain DataLoader (rank/worker sharded). Alternatives: HF datasets (arrow memmap, `split_dataset_by_node`), mosaicml-streaming MDS (`StreamingDataset`: built-in distributed sharding, deterministic shuffle, mid-epoch resumption — best if you outgrow memmap), webdataset (better for audio blobs). DataLoader: `num_workers=2–8, pin_memory=True, persistent_workers=True, drop_last=True`.

**Minimal-correct skeleton** (launch: `torchrun --standalone --nproc_per_node=8 train.py`; single-process `python train.py` for Mac smoke tests):

```python
import os, torch, torch.distributed as dist, torch.nn.functional as F
from contextlib import nullcontext
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.nn.parallel import DistributedDataParallel as DDP

ddp_env = "RANK" in os.environ
if ddp_env:
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available(): torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Transformer(cfg).to(device)                      # fp32 random init
use_fsdp2 = ddp_env and device == "cuda"
if use_fsdp2:
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for blk in model.blocks:                             # (optional) blk.compile() first
        fully_shard(blk, mp_policy=mp)                   # non-root: reshard_after_forward=True
    fully_shard(model, mp_policy=mp)                     # root: stays unsharded post-forward
elif ddp_env:
    model = DDP(model)                                   # device_ids set via set_device
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95),
                        weight_decay=0.1, fused=(device == "cuda"))  # after sharding
amp = torch.autocast(device, torch.bfloat16) if (device=="cuda" and not use_fsdp2) else nullcontext()
for step in range(total_steps):
    for i in range(accum):
        last = i == accum - 1
        if use_fsdp2: model.set_requires_gradient_sync(last)
        x, y = next(it); x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with (model.no_sync() if (isinstance(model, DDP) and not last) else nullcontext()), amp:
            loss = F.cross_entropy(model(x).flatten(0, 1).float(), y.flatten()) / accum
        loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # DTensor-aware
    opt.step(); opt.zero_grad(set_to_none=True); sched.step()
```

Checkpoint: `msd, osd = get_state_dict(model, opt); dcp.save({"model": msd, "optim": osd}, checkpoint_id=path)`; export `get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))` → rank0 `save_file`.

## KEY FACTS
- FSDP2 public API: from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy, OffloadPolicy, CPUOffloadPolicy; fully_shard(module, *, mesh=None, reshard_after_forward=None, shard_placement_fn=None, mp_policy=MixedPrecisionPolicy(), offload_policy=OffloadPolicy(), ignored_params=None) applies in place (no wrapper), params become dim-0 DTensors, FQNs unchanged [https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html]
- reshard_after_forward=None (default) resolves to True for non-root and False for root modules; True frees params after forward and re-all-gathers in backward; False keeps them unsharded; int reshards to a smaller world size [https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html]
- MixedPrecisionPolicy fields/defaults: param_dtype=None, reduce_dtype=None, output_dtype=None, cast_forward_inputs=True; FSDPModule methods include set_requires_gradient_sync(bool, recurse=True) for gradient accumulation, set_reshard_after_backward, set_is_last_backward, unshard/reshard, set_gradient_divide_factor [https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html]
- Must call fully_shard bottom-up (blocks then root), initialize the optimizer after sharding (DTensor params), and invoke model(input) not model.forward(input) unless register_fsdp_forward_method is used [https://docs.pytorch.org/docs/2.9/distributed.fsdp.fully_shard.html]
- FSDP1 (FullyShardedDataParallel) is deprecated as of PyTorch 2.11; FSDP2 uses per-parameter DTensor sharding instead of FlatParameter [https://huggingface.co/docs/accelerate/en/concept_guides/fsdp1_vs_fsdp2]
- PyTorch 2.9 breaking change: compiling through FSDP2 hooks without graph breaks is no longer supported — either run with fullgraph=False or apply torch.compile before applying FSDP; 2.9 also added torch._dynamo.error_on_graph_break() [https://pytorch.org/blog/pytorch-2-9/]
- Current stable PyTorch is 2.12.1 (released 2026-06-17); 2.12.0 on 2026-05-13, 2.11.0 on 2026-03-23 [https://github.com/pytorch/pytorch/wiki/PyTorch-Versions]
- torchtitan llama3 parallelize order: activation checkpointing → per-TransformerBlock torch.compile → fully_shard ('turn on per-TransformerBlock compile after AC wrapping and before FSDP'), with MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32, cast_forward_inputs=False) [https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize.py]
- torchtitan shards each transformer block as its own FSDP unit with reshard_after_forward=True (default policy, False when PP enabled), groups tied embeddings+lm_head into one FSDP unit, and sets reshard_after_forward=False for final norm+lm_head since FSDP would prefetch them immediately [https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/fsdp.py]
- torchtitan gradient accumulation does NOT use set_requires_gradient_sync: it runs backward (reduce-scatter) per microbatch, disables FSDP's automatic gradient division via set_gradient_divide_factor(1.0), and scales by global token count; grad clipping via a DTensor-aware clip_grad_norm_ [https://github.com/pytorch/torchtitan/blob/main/torchtitan/trainer.py]
- torchtitan optimizer default implementation is 'fused' (torch.optim.AdamW(fused=True), CUDA only), with alternatives foreach/for-loop and a fused_opt_states_bf16 mode keeping Adam moments in bf16 [https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/optimizer.py]
- torchtitan LR scheduler is warmup-stable-decay: linear warmup (default warmup_steps=200), optional stable phase, then decay with decay_type in {linear, sqrt, cosine} [https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/lr_scheduler.py]
- OLMo-core (AI2) 'effectively uses new PyTorch features like compile, FSDP2, and selective activation checkpointing following best practices from torchtitan', supports DDP/FSDP/HSDP, and uses hybrid sharding beyond ~256 GPUs [https://allenai.org/blog/olmo2-32B]
- litgpt's pretrain script defaults to FSDP with bfloat16 mixed precision and gradient accumulation; nanotron is HF's 3D-parallelism (TP/PP/DP) framework aimed at hundreds of GPUs [https://github.com/Lightning-AI/litgpt/blob/main/tutorials/pretrain_tinyllama.md]
- ezyang (Aug 2025) on torch.compile for training: use regional compilation (compile the transformer block, reused across blocks) to control compile time; recommended approach for a training stack is to fork torchtitan; compiled autograd requires the entire backward to be compileable [https://blog.ezyang.com/2025/08/state-of-torch-compile-august-2025/]
- With FSDP, non-reentrant activation checkpointing (use_reentrant=False) is strongly advised; reentrant checkpointing with FSDP2 has known memory-leak/correctness issues (e.g. pytorch/pytorch#169349) [https://github.com/pytorch/pytorch/issues/169349]
- DCP safetensors interop: torch.distributed.checkpoint.HuggingFaceStorageWriter(path, fqn_to_index_mapping)/HuggingFaceStorageReader(path) used with dcp.save/dcp.load over any fsspec backend; consolidated single-file output was planned at the June 2025 blog and torchtune (PR #2557) was first adopter; QuantizedHuggingFaceStorageReader also exists in 2.12 [https://pytorch.org/blog/huggingface-safetensors-support-in-pytorch-distributed-checkpointing/]
- torch.distributed.checkpoint 2.12 APIs: save/load/async_save(state_dict, checkpoint_id=..., storage_writer/reader=...); state_dict helpers get_model_state_dict/get_optimizer_state_dict/set_model_state_dict/set_optimizer_state_dict with StateDictOptions(full_state_dict, cpu_offload, broadcast_from_rank0) for consolidated export/load [https://docs.pytorch.org/docs/2.12/distributed.checkpoint.html]
- HF Accelerate supports FSDP2 via FullyShardedDataParallelPlugin(fsdp_version=2) (documented since accelerate v1.6.0), including FULL_STATE_DICT so save_pretrained works with FSDP2-wrapped models [https://huggingface.co/docs/accelerate/v1.6.0/en/concept_guides/fsdp1_vs_fsdp2]
- F.scaled_dot_product_attention dispatches among FlashAttention-2, memory-efficient, cuDNN, and math backends (FA2 integrated since torch 2.2, ~2x speedup); flash-attn package is only needed for maximum features (e.g. flash_attn_varlen_func with cu_seqlens); FlexAttention compiles arbitrary masks (BlockMask document masking for packed sequences) to FlashAttention-grade kernels [https://pytorch.org/blog/pytorch2-2/]
- WSD (warmup-stable-decay) empirically matches or outperforms cosine decay at equal compute and avoids committing to a fixed training horizon (decay branches from a constant-LR trunk); caveats: needs a decay trigger and stable-phase checkpoints understate final quality [https://www.emergentmind.com/topics/warmup-stable-decay-wsd-schedules]
- mosaicml-streaming StreamingDataset (MDS format) provides deterministic distributed sharding/shuffling and mid-epoch resumption for multi-node training; HF datasets uses memory-mapped Arrow cache (can be ~8x base dataset size); webdataset ships pre-shuffled tar shards [https://www.databricks.com/blog/mosaicml-streamingdataset]
- FlexAttention/document-masking guidance for packed pretraining sequences (intra-document causal masking to prevent cross-document attention) is covered in HF's sequence-packing writeup and the PyTorch FlexAttention docs [https://huggingface.co/blog/sirluk/llm-sequence-packing]

## RECOMMENDATION
Adopt the torchtitan pattern on torch>=2.8 (pin the current stable, 2.12.1): FSDP2 `fully_shard` per transformer block + root with `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)` and fp32 master weights; keep a DDP+autocast(bf16) fallback and a single-process CPU/MPS path (same code, sharding skipped) for Mac smoke tests. Launch with `torchrun --standalone --nproc_per_node=8`. Use `F.scaled_dot_product_attention(is_causal=True)` — no flash-attn dependency; adopt FlexAttention BlockMask only if intra-document masking of packed sequences proves necessary. Fused AdamW (0.9/0.95, wd 0.1), grad clip 1.0 via `torch.nn.utils.clip_grad_norm_`, gradient accumulation via `set_requires_gradient_sync(last_microbatch)` / DDP `no_sync()`. Per-block activation checkpointing (`use_reentrant=False`) applied before optional per-block regional `torch.compile`, both before `fully_shard`; ship compile off by default, enable as a flag. LR: warmup+cosine baseline, WSD (torchtitan-style) once you run open-ended token budgets. Checkpoints: DCP (`dcp.save` + `get_state_dict`) for resumable sharded state, plus rank0 consolidated safetensors export via `StateDictOptions(full_state_dict=True, cpu_offload=True)`. Data: pre-tokenize (speech-codec + text tokens) into flat memmap/parquet shards with rank×worker sharding; move to mosaicml-streaming only if dataset scale or resumption demands it. Avoid DeepSpeed and Accelerate — unnecessary at 100M–1B single-node.