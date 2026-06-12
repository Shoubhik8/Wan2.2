# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Non-invasive torch.compile path for the ti2v-5B DiT (optimizations.md, Step 4).

Nothing in the existing source tree is modified. This module subclasses
``WanTI2V`` and, in ``_configure_model``, compiles each transformer block in
``model.blocks`` with ``mode="max-autotune-no-cudagraphs", dynamic=False``.

Per the graph-break probe (tools/compile_graph_break_probe.py), the only two
in-block blockers are ``rope_apply`` (unbacked-symint from ``grid_sizes.tolist()``)
and ``attention`` (flash-attn varlen dynamic slicing). Both are reachable as module
globals of ``wan.modules.model`` (``from .attention import attention``; ``rope_apply``
is defined there), and the blocks call them by bare name — so we contain them by
wrapping those two globals with ``torch.compiler.disable`` at runtime. No file edits.

Run via generate_compiled_optimized.py (compile ON). The stock generate.py is
untouched (compile OFF), so behavior there stays byte-identical.
"""
import logging
import time

import torch

import wan.modules.model as _model_mod

from .textimage2video import WanTI2V

# optimizations.md:58/60 — offload-safe (no cudagraph pinned addresses), best speedup.
COMPILE_MODE = "max-autotune-no-cudagraphs"

# Sentinel so the global monkey-patches are applied at most once per process.
_PATCH_SENTINEL = "_compiled_optimized_disable_applied"


def _apply_dynamo_disable_patches():
    """Wrap rope_apply / attention with torch.compiler.disable (eager islands).

    Patches the bindings in ``wan.modules.model`` — that is the namespace the
    WanAttentionBlock self/cross attention resolve at call time. Idempotent.
    """
    if getattr(_model_mod, _PATCH_SENTINEL, False):
        return
    _model_mod.rope_apply = torch.compiler.disable(_model_mod.rope_apply)
    _model_mod.attention = torch.compiler.disable(_model_mod.attention)
    setattr(_model_mod, _PATCH_SENTINEL, True)
    logging.info(
        "[compiled_optimized] wrapped rope_apply + attention with "
        "torch.compiler.disable (graph-break containment)")


def _compile_blocks(model, mode=COMPILE_MODE):
    """Compile each block's forward. The 30 identical blocks share one artifact.

    Compile is lazy: applying it while the model is still on CPU (ti2v default
    init_on_cpu=True) is fine — the artifact materializes on the first forward
    after model.to(device) and survives offload .to() moves.
    """
    n = 0
    for blk in model.blocks:
        blk.forward = torch.compile(blk.forward, mode=mode, dynamic=False)
        n += 1
    logging.info(
        f"[compiled_optimized] torch.compile applied to {n} DiT blocks "
        f"(mode={mode}, dynamic=False) — expect a one-time autotune warmup on "
        f"the first forward pass.")


def _wrap_model_timing(model):
    """Log wall-clock per WanModel.forward call (2 calls per denoising step).

    Self-contained instrumentation — avoids editing the t2v()/i2v() loop. The
    first call includes the one-time compile warmup; later calls are steady state.
    """
    orig_forward = model.forward
    state = {"n": 0}

    def timed_forward(*a, **kw):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = orig_forward(*a, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e3
        i = state["n"]
        tag = " (incl. compile warmup)" if i == 0 else ""
        logging.info(f"[compiled_optimized] DiT forward #{i}: {dt:.1f} ms{tag}")
        state["n"] = i + 1
        return out

    model.forward = timed_forward


class WanTI2VCompiled(WanTI2V):
    """ti2v-5B pipeline with torch.compile'd DiT blocks. Drop-in for WanTI2V."""

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        # Run the stock configuration first (eval, optional SP patch, dtype/device).
        # Compiling after preserves any SP-patched self_attn.forward (optimizations.md:60).
        model = super()._configure_model(model, use_sp, dit_fsdp, shard_fn,
                                         convert_model_dtype)

        if dit_fsdp:
            logging.warning(
                "[compiled_optimized] --dit_fsdp is on; skipping torch.compile "
                "(sharded params). Run single-GPU for the compile path.")
            return model

        _apply_dynamo_disable_patches()
        _compile_blocks(model, COMPILE_MODE)
        _wrap_model_timing(model)
        return model
