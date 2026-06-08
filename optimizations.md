# Wan2.2 Inference Speedup Plan

## Context

The starting question was whether `torch.compile` on the **VAE** would make the pipeline faster. After exploring the codebase, the answer is **no, not meaningfully** — and this plan redirects effort to where the time actually goes.

**Why the VAE is the wrong target:**
- VAE `decode` runs **exactly once per generation**, at the very end of the loop ([text2video.py:368](wan/text2video.py#L368)). It is well under ~5% of wall-clock.
- The VAE is also hostile to `torch.compile`: mutable `feat_cache` lists, string markers (`'Rep'`), per-frame Python loops, and dynamic conv padding all force graph breaks ([vae2_1.py](wan/modules/vae2_1.py), [vae2_2.py](wan/modules/vae2_2.py)).

**Where the time actually goes:** the DiT denoising loop. Per video it runs `sample_steps` (40 for A14B, 50 for ti2v-5B) × **2 forward passes** (CFG cond + uncond, [text2video.py:346-349](wan/text2video.py#L346-L349)) of a 14B transformer = **80–100 heavy forwards**. That loop is the optimization surface.

**Target setting:** single GPU with `--offload_model True`; all five tasks (prefer shared code paths); **aggressive** quality tradeoffs acceptable. Note: under offload, the MoE expert swap in [`_prepare_model_for_timestep`](wan/text2video.py#L169-L201) happens **once** at the SNR boundary, not every step — the active expert stays resident on GPU within a phase, so compile/cache artifacts survive.

Every change is **flag-gated**; default behavior stays byte-identical when flags are off.

## Shared structure (applies to all 5 pipelines)

All pipelines share the same per-step pattern (two `model()` calls + CFG combine):
- [text2video.py:346-352](wan/text2video.py#L346-L352)
- [image2video.py:393-402](wan/image2video.py#L393-L402)
- [textimage2video.py:380-386](wan/textimage2video.py#L380-L386) and `:580-589`
- [speech2video.py:623-634](wan/speech2video.py#L623-L634)
- [animate.py:609-623](wan/animate.py#L609-L623)

`WanModel.forward` ([model.py:410-497](wan/modules/model.py#L410-L497)) already takes `x`/`context` as **List[Tensor]** and batches them via `torch.cat`/`torch.stack` — so it already supports batch > 1. The block loop is [model.py:489-490](wan/modules/model.py#L489-L490). For distributed runs, the same logic must be mirrored in `sp_dit_forward` ([sequence_parallel.py:64-144](wan/distributed/sequence_parallel.py#L64-L144)).

---

## Recommended approach (rollout order)

### Step 1 — Fewer sampling steps (zero code, validate quality floor first)
`--sample_steps` already exists ([generate.py:208](generate.py#L208)) and overrides `cfg.sample_steps`. No code change.
- Aggressive values (unipc handles low step counts well): A14B/s2v **20–25** (from 40), ti2v-5B **25–30** (from 50), animate **15** (from 20).
- Don't go below ~16 on A14B — the high-noise phase (boundary 0.875/0.900) needs ≥2–3 steps.
- **~1.6–2x**, orthogonal to everything. Run this first to find the acceptable quality floor before adding caching.

### Step 2 — Step caching (TeaCache-style) — the biggest lever
Skip the entire block stack on steps where the time-modulation embedding `e0` changes little vs. the last computed step, reusing the previous **residual** (`x_after_blocks − x_before_blocks`).
- **Implement in shared code**: inside `WanModel.forward` around the block loop ([model.py:488-491](wan/modules/model.py#L488-L491)), mirrored in [sequence_parallel.py:133-134](wan/distributed/sequence_parallel.py#L133-L134).
- **Signal**: rescaled relative-L1 of `e0` ([model.py:468](wan/modules/model.py#L468)) accumulated across steps; when `accum < threshold`, reuse cached residual; else recompute and reset.
- **State** lives on the `WanModel` instance (`self._cache_accum`, `self._cache_residual`, `self._cache_e0_prev`, `self._cache_threshold`), set to `None`/`0` in `__init__`. Add a `reset_cache()` helper called at the top of each `generate()` (e.g. [text2video.py:330](wan/text2video.py#L330)).
- **MoE-safe**: cache is per-model-instance, so `low_noise_model` and `high_noise_model` never share residuals (correct — never reuse across experts). Force a real compute on the first step after the boundary swap, and on the first/last 2 steps (protect structure + detail).
- **CFG**: cache cond and uncond separately. Cleanest if combined with Step 3 (batched → single residual). Without batching, key the residual dict by a `cfg_tag`.
- **Flag**: `--cache_threshold` (float, default `0.0` = off). Aggressive ≈ `0.1` (skips ~30–40%) to `0.15–0.2` (skips ~50–60%).
- **~1.5–2.5x**, composes multiplicatively.

### Step 3 — CFG batching (pairs naturally with caching)
Replace the two sequential `model()` calls with one call passing 2-element lists (`x=[latent,latent]`, `context=[context, context_null]`, `y=[y,y]` where present). The model concatenates into batch dim 2 ([model.py:454](wan/modules/model.py#L454)).
- Edit the per-step pattern in all 5 pipelines (locations above). Remove the interleaved `empty_cache()` calls in i2v ([image2video.py:396,400](wan/image2video.py#L396-L400)) on the batched path. For ti2v pass `t` as `torch.cat([timestep, timestep])`.
- **Flag**: `--cfg_batch` (default off). Halves kernel-launch/Python overhead; same FLOPs.
- **VRAM**: roughly doubles DiT activations. **Safe at 480p / ti2v-5B; risky at 720p A14B on 24GB** — gate by resolution / leave off at 720p.
- **~1.3–1.7x** (larger when launch-bound, smaller when compute-bound at 720p). Strong reason to pair with Step 2: batched forward yields a single residual cache entry (simpler).

### Step 4 — torch.compile the DiT blocks
Compile **each block** in `self.blocks` (not the whole `forward`) — the 32 identical blocks share one compiled artifact and avoid the graph-break-heavy pre/post-amble (List comprehensions, `view_as_complex/real`, flash_attn op).
- **Insert** in `_configure_model` ([text2video.py:125-167](wan/text2video.py#L125-L167) and parallels), **after** the SP monkey-patch block ([:150-154](wan/text2video.py#L150-L154)) and after `model.to(device)` ([:163-165](wan/text2video.py#L163-L165)):
  `for blk in model.blocks: blk.forward = torch.compile(blk.forward, mode="max-autotune-no-cudagraphs", dynamic=False)`
- **Contain graph breaks**: wrap `rope_apply` ([model.py:39-66](wan/modules/model.py#L39-L66)) and the flash-attn entry ([attention.py:24-130](wan/modules/attention.py#L24-L130)) with `torch.compiler.disable` so the surrounding GEMMs/norms/FFN compile cleanly.
- **Offload-safe**: use `max-autotune-no-cudagraphs` (NOT cudagraphs/reduce-overhead graph capture — pinned addresses break on CPU↔GPU swap). `dynamic=False` because seq_len is fixed per generate call ([text2video.py:257](wan/text2video.py#L257)). The compiled artifact survives `.to()` moves. Compile **after** SP patching so the patched `self_attn.forward` is captured (don't compile `model.forward` — it would clobber `sp_dit_forward`).
- **Flag**: `--compile` (default off). One-time warmup ~1–3 min — worth it for full runs, note in help text.
- **~1.15–1.35x**.

### Optional Step 5 — attention/quant backends (needs new deps + hardware gating)
Lower priority; only if more speed is needed.
- **SageAttention (int8 attn)**: new branch in [attention.py:flash_attention](wan/modules/attention.py#L24) above the FA3/FA2 branches, flag-gated. ~1.2–1.5x on attention. Works Ampere+; adds a dep.
- **fp8 DiT linears** (torchao/TE): quantize the q/k/v/o and FFN `nn.Linear` ([model.py:119-122,170-172,212-214](wan/modules/model.py#L119-L122)) in `_configure_model`. ~1.3–1.6x GEMMs + halves weight VRAM (lighter offload traffic). **Fast only on Ada (RTX 40xx)/Hopper; slower on Ampere** — gate behind `torch.cuda.get_device_capability() >= (8,9)`.

---

## Composition & expected gains

| Step | DiT speedup | New dep | VRAM impact |
|------|-------------|---------|-------------|
| 1 fewer steps | 1.6–2x | no | none |
| 2 step cache | 1.5–2.5x | no | none |
| 3 CFG batch | 1.3–1.7x | no | ~2x activations |
| 4 compile | 1.15–1.35x | no | none |
| 5 sage/fp8 | 1.2–1.6x | yes | fp8 halves weights |

Stacking steps 1–4 realistically gives **~4–8x** on the DiT loop.

**Best aggressive recipe:**
- **480p / ti2v-5B**: `--sample_steps 25 --cache_threshold 0.1 --cfg_batch --compile` → ~4–6x, safe on 24GB + offload.
- **720p A14B**: drop `--cfg_batch` (VRAM); `--sample_steps 25 --cache_threshold 0.1 --compile` → ~3–4x.
- Always protect first/last 2 steps and the boundary-crossing step from caching.

## Critical files
- [wan/modules/model.py](wan/modules/model.py) — `WanModel.forward` (caching + compile target), block loop 489-490, `__init__` cache state, `reset_cache()`.
- [wan/text2video.py](wan/text2video.py) — reference pipeline: per-step loop 346-352, `_configure_model` 125-167. **Mirror all edits** in image2video.py, textimage2video.py, speech2video.py, animate.py.
- [generate.py](generate.py) — new flags near 133-223; thread `cfg_batch`/`compile`/`cache_threshold` through pipeline construction + `generate()` calls (403-540).
- [wan/distributed/sequence_parallel.py](wan/distributed/sequence_parallel.py) — mirror caching in `sp_dit_forward` (64-144).
- [wan/modules/attention.py](wan/modules/attention.py) — `torch.compiler.disable` wrap; Step 5 backend insertion.

## Verification
1. **Correctness (flags off)**: run a short generation with all new flags off and confirm output is byte-identical to current `main` (same seed). E.g. `python generate.py --task ti2v-5B --size 704*1280 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --prompt "..." --sample_steps 50`.
2. **Per-step timing**: add a tqdm postfix or simple timer around the loop; compare steps/sec with each flag toggled on.
3. **Quality sweep**: fixed seed + prompt, sweep `--cache_threshold {0, 0.05, 0.1, 0.15}` and `--sample_steps {40,25,20}`; eyeball the videos to pick the aggressive floor.
4. **VRAM**: watch `nvidia-smi` peak with `--cfg_batch` at 480p vs 720p to confirm it fits under offload.
5. **Compile**: confirm `--compile` produces no hard graph-break errors (run with `TORCH_LOGS=recompiles` once) and that the warmup cost amortizes over a full run.
6. **Distributed (optional)**: if a multi-GPU box is available, run `torchrun ... --ulysses_size N --dit_fsdp` with `--cache_threshold 0.1` to confirm the `sp_dit_forward` caching mirror works.
7. Run `make format` before committing.
