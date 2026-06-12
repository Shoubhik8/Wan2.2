# torch.compile for the ti2v-5B DiT — Implementation Writeup

This documents the `torch.compile` optimization (Step 4 of [optimizations.md](optimizations.md))
as implemented for the **ti2v-5B** model. The goal was to compile the DiT transformer blocks for a
~1.15–1.35x speedup on the denoising loop **without modifying any existing source file**.

## TL;DR

- Two new files, both ending in `compiled_optimized`; nothing in the existing tree is edited.
- Each of the 30 DiT blocks is wrapped with `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)`.
- The two graph-break culprits (`rope_apply`, `attention`) are contained via runtime monkey-patch
  with `torch.compiler.disable` — no edits to `model.py`/`attention.py`.
- Compile is ON when you run `generate_compiled_optimized.py`; the stock `generate.py` is untouched
  and stays byte-identical.

## Why blocks, not the whole forward

The ti2v-5B DiT is 30 **identical** `WanAttentionBlock`s (`dim=3072`, `num_layers=30`,
[wan_ti2v_5B.py](wan/configs/wan_ti2v_5B.py)). `WanModel.forward`
([model.py:410-497](wan/modules/model.py#L410-L497)) has a graph-break-heavy pre/post-amble —
`List[Tensor]` comprehensions, `tensor`-from-Python-shape, `view_as_complex/real`, asserts — that
makes whole-model `fullgraph=True` non-viable. The probe
([tools/compile_graph_break_probe.py](tools/compile_graph_break_probe.py)) confirmed this:
whole-model `fullgraph=True` fails on an unbacked-symint guard, but a **single block compiles and runs
correctly** in non-fullgraph mode (max_abs_diff 5.3e-3 vs eager — bf16-level noise).

Compiling `blk.forward` per block means the 30 blocks share **one** compiled artifact (same code
object, fixed shapes) and we skip the messy amble entirely.

## The two graph breaks, and how they're contained

The probe localized exactly two in-block blockers:

| # | Location | Why it breaks |
|---|----------|---------------|
| 1 | `rope_apply` ([model.py:39-66](wan/modules/model.py#L39-L66)) | `grid_sizes.tolist()` → data-dependent `reshape` (unbacked symint `Eq(seq, u0*u1*u2)`); `view_as_complex/real` |
| 2 | `attention` ([attention.py:24](wan/modules/attention.py#L24)) | flash-attn varlen `cu_seqlens` dynamic slicing + opaque `flash_attn_varlen_func` |

Neither can be made fullgraph-clean, so the right tool is an **eager island**: wrap them in
`torch.compiler.disable` (the public alias of the `torch._dynamo.disable` the probe used). Dynamo then
compiles the surrounding GEMMs / norms / FFN and calls these two in eager mode.

**Key insight that keeps this non-invasive:** the blocks call both functions by **bare name**
(`attention(q=rope_apply(...), ...)` at [model.py:145-147](wan/modules/model.py#L145-L147),
[:175](wan/modules/model.py#L175)), and both names live in the `wan.modules.model` module namespace
(`from .attention import attention` at [model.py:9](wan/modules/model.py#L9); `rope_apply` is defined
in that file). Python resolves those names from the module globals **at call time**, so rebinding the
globals at runtime swaps in the disabled versions — no source edit:

```python
import wan.modules.model as _m
_m.rope_apply = torch.compiler.disable(_m.rope_apply)
_m.attention  = torch.compiler.disable(_m.attention)
```

## How the pipeline is swapped (also non-invasive)

`generate.py` builds the ti2v pipeline via **module attribute access** —
`wan.WanTI2V(...)` ([generate.py:430](generate.py#L430)) — not a bound import. So rebinding the
attribute reroutes ti2v through a subclass:

```python
wan.WanTI2V = WanTI2VCompiled
```

`WanTI2VCompiled(WanTI2V)` ([wan/textimage2video_compiled_optimized.py](wan/textimage2video_compiled_optimized.py))
overrides only `_configure_model`: it calls `super()._configure_model(...)` first (preserving the exact
`eval()`, optional sequence-parallel patch, and dtype/device behavior), then applies the disable
patches, compiles the blocks, and wraps timing. Compiling **after** `super()` matters so any
SP-patched `self_attn.forward` is the version captured ([optimizations.md:60](optimizations.md#L60)).

## Files

### `wan/textimage2video_compiled_optimized.py`
- `_apply_dynamo_disable_patches()` — idempotent (sentinel-guarded) global wrap of `rope_apply` + `attention`.
- `_compile_blocks(model, mode)` — `blk.forward = torch.compile(blk.forward, mode=mode, dynamic=False)` for each block.
- `_wrap_model_timing(model)` — wraps `WanModel.forward` with a `cuda.synchronize()`-bracketed timer,
  logging ms per call (2 calls/step: CFG cond + uncond). Forward #0 includes the autotune warmup.
- `WanTI2VCompiled(WanTI2V)` — the subclass; skips compile + warns under `--dit_fsdp`.

### `generate_compiled_optimized.py`
Thin wrapper: imports `generate`, rebinds `wan.WanTI2V`, then calls `generate._parse_args()` +
`generate.generate(args)`. Warns if a non-ti2v task is passed (compile only applies to ti2v-5B).

## Design choices

- **`mode="max-autotune-no-cudagraphs"`** — best kernel speedup and **offload-safe**. We must *not* use
  cudagraphs/`reduce-overhead`: cudagraph captures pinned device addresses that break across the
  `--offload_model` CPU↔GPU `.to()` swaps. Plain max-autotune fuses + autotunes without that capture.
- **`dynamic=False`** — `seq_len` is fixed for a whole `generate()` call, so latent shapes are constant
  across all 25–50 steps ⇒ one compile, no per-step recompiles. The autotune warmup (~1–3 min) is paid
  once on forward #0 and amortizes over the run.
- **Lazy compile + CPU placement** — ti2v defaults to `init_on_cpu=True` (model on CPU after config;
  moved to GPU at [textimage2video.py:365](wan/textimage2video.py#L365)). `torch.compile` is lazy, so
  wrapping while on CPU is fine — the artifact materializes on the first GPU forward and **survives
  `.to()` moves**, so the offload `.cpu()`/`.to(device)` cycle is safe.
- **Script-as-gate** — no `--compile` flag is needed: running the compiled entry point turns it on,
  running stock `generate.py` leaves it off. The global disable patches are only ever imported by the
  new script, so the stock pipeline is unaffected.

## Running it

```bash
cd Wan2.2
TORCH_LOGS=recompiles python generate_compiled_optimized.py \
    --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B \
    --offload_model True --convert_model_dtype \
    --sample_steps 25 --prompt "a cat playing piano"
```

## Verification checklist

1. **No hard graph break / single compile** — a 1–3 min stall on DiT forward #0, then fast steps; the
   `TORCH_LOGS=recompiles` output shows it compiles once (not per step).
2. **Speedup** — compare the steady-state per-forward ms (from the timing log) and total wall-clock
   against the same command on stock `generate.py`. Target ~1.15–1.35x on the DiT loop.
3. **Correctness** — same `--base_seed` + prompt on both; output should be visually identical (compile
   adds only ~5e-3 fp noise).
4. **Offload safety** — generation completes across the `.cpu()`/`.to(device)` offload cycle without
   address/cudagraph errors, confirming `max-autotune-no-cudagraphs` was the right mode.

## Caveats / scope

- **Single model only.** ti2v-5B is a single `WanModel` (not MoE). The A14B tasks have
  `low_noise_model`/`high_noise_model`; extending compile there means compiling both experts' blocks.
- **Flash-attn assumed.** The ~1.15–1.35x estimate assumes flash-attn is installed so the disabled
  `attention` island stays cheap. If it falls back to `scaled_dot_product_attention`
  ([attention.py:148-179](wan/modules/attention.py#L148-L179)), the eager island is heavier and the net
  gain shrinks.
- **`--dit_fsdp` not supported** on the compile path (sharded params); run single-GPU.
- **Composability.** This is orthogonal to Steps 1–3 (fewer steps, step caching, CFG batching) and
  stacks multiplicatively, but those are not part of this change.
