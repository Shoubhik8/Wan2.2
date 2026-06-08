# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
TeaCache-style step caching for ``WanModel`` (optimisations Step 2).

This module is a *side-by-side* optimisation: it does not modify ``model.py``.
Instead it provides three functions that are bound onto an already-constructed
``WanModel`` instance via ``types.MethodType`` (the same pattern the repo uses
for sequence-parallel in ``_configure_model``):

    init_cache_state(model, threshold)
    model.forward = types.MethodType(cached_dit_forward, model)
    ...
    reset_cache(model)   # called at the top of each generate()

The idea: between adjacent denoising steps the time-modulation embedding ``e0``
(the sole per-step control signal for every transformer block) often barely
moves. When the accumulated relative-L1 drift of ``e0`` since the last real
compute stays below ``threshold``, we skip the entire 32-block stack and reuse
the previously computed **residual** ``x_after_blocks - x_before_blocks``.

Correctness:
  * ``threshold == 0.0`` (default) makes ``cached_dit_forward`` numerically
    identical to ``WanModel.forward`` -- no residual is ever stored or reused
    and the drift signal is never computed (no extra device sync).
  * Cache state lives on the instance keyed by ``cache_tag`` ('cond'/'uncond')
    because the two CFG calls at a step share ``e0`` but produce different
    residuals.
  * Each MoE expert is a separate ``WanModel`` instance, so cache state is
    naturally isolated; ``e0_prev is None`` forces a real compute on the first
    call of each expert.
"""
import torch

# ``cached_dit_forward`` references ``sinusoidal_embedding_1d`` exactly as the
# original ``WanModel.forward`` does; import it from the unmodified module so we
# stay in lock-step with any upstream change.
from .model import sinusoidal_embedding_1d


def init_cache_state(self, threshold):
    r"""Attach TeaCache state to a WanModel instance.

    Args:
        threshold (float): accumulated-drift threshold. ``0.0`` disables
            caching (byte-identical to the original forward).
    """
    self._cache_threshold = float(threshold)
    self._cache_e0_prev = {}        # {tag: Tensor} e0 at the last real compute
    self._cache_accum = {}          # {tag: float}
    self._cache_residual = {}       # {tag: Tensor}


def reset_cache(self):
    r"""Clear cache state. Call once at the top of every generate()."""
    self._cache_e0_prev = {}
    self._cache_accum = {}
    self._cache_residual = {}


def _rescale_signal(dist):
    r"""Hook for the TeaCache calibration polynomial.

    v1 uses the identity (plain relative-L1); ``--cache_threshold`` absorbs the
    scale. A fitted ``numpy.poly1d`` can be dropped in here later without
    touching any call site.
    """
    return dist


def cached_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    cache_tag='cond',
    force_recompute=False,
):
    r"""Drop-in replacement for ``WanModel.forward`` with step caching.

    Verbatim copy of ``WanModel.forward`` except the block loop, which is
    wrapped so it can be skipped (reusing the cached residual) when ``e0`` has
    drifted little since the last real compute.

    Extra args (passed as explicit kwargs from the optimised pipeline):
        cache_tag (str): 'cond' or 'uncond' -- selects the residual slot.
        force_recompute (bool): force a real block-stack compute this call
            (used to protect the first/last steps and the MoE boundary).
    """
    if self.model_type == 'i2v':
        assert y is not None
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                  dim=1) for u in x
    ])

    # time embeddings
    if t.dim() == 1:
        t = t.expand(t.size(0), seq_len)
    with torch.amp.autocast('cuda', dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, seq_len)).float())
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)

    # --- TeaCache step-caching wrapper around the block loop ----------------
    # State is per cache_tag ('cond'/'uncond'): the two CFG calls share t (so
    # the same e0 each step) but produce different residuals AND may recompute
    # on different steps, so each tag tracks its own last-compute e0, drift
    # accumulator, and residual. With identical t and threshold the two tags
    # make the same decisions and stay in lockstep.
    e0_prev = self._cache_e0_prev.get(cache_tag)
    threshold = self._cache_threshold
    do_cache = (threshold > 0.0) and (not force_recompute) and (
        e0_prev is not None)
    recomputed = False

    if do_cache:
        dist = (e0 - e0_prev).abs().mean() / (e0_prev.abs().mean() + 1e-8)
        self._cache_accum[cache_tag] = self._cache_accum.get(
            cache_tag, 0.0) + _rescale_signal(dist.item())
        if (self._cache_accum[cache_tag] < threshold
                and cache_tag in self._cache_residual):
            # SKIP: reuse cached residual
            x = x + self._cache_residual[cache_tag]
        else:
            # RECOMPUTE + reset this tag's accumulator
            x_before = x
            for block in self.blocks:
                x = block(x, **kwargs)
            self._cache_residual[cache_tag] = (x - x_before).detach()
            self._cache_accum[cache_tag] = 0.0
            recomputed = True
    else:
        # FULL compute (caching off / forced / first call for this expert)
        x_before = x
        for block in self.blocks:
            x = block(x, **kwargs)
        if threshold > 0.0:
            self._cache_residual[cache_tag] = (x - x_before).detach()
            self._cache_accum[cache_tag] = 0.0
        recomputed = True

    # e0_prev advances only on a real compute, so the accumulator measures
    # drift since the last ACTUAL block-stack evaluation for this tag.
    if recomputed:
        self._cache_e0_prev[cache_tag] = e0.detach()
    # ------------------------------------------------------------------------

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    return [u.float() for u in x]
