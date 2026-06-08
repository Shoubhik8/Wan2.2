# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Optimised text-to-video pipeline with TeaCache-style step caching (Step 2).

``WanT2V_Optimised`` subclasses the original ``WanT2V`` and:
  * attaches caching state to both MoE experts after construction, binding
    ``cached_dit_forward`` onto each (the originals are never modified);
  * overrides ``generate()`` -- a verbatim copy of ``WanT2V.generate`` with a
    handful of edited lines: ``reset_cache`` before the loop, a per-step
    ``force`` flag (first/last 2 steps + expert swap), and ``cache_tag`` /
    ``force_recompute`` on the two model calls.

With ``cache_threshold == 0.0`` (default) behaviour is identical to ``WanT2V``.
"""
import gc
import math
import random
import sys
import types
from contextlib import contextmanager

import torch
import torch.distributed as dist
from tqdm import tqdm

from .text2video import WanT2V
from .modules.model_optimised import (
    cached_dit_forward,
    init_cache_state,
    reset_cache,
)
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V_Optimised(WanT2V):

    def __init__(self, *args, cache_threshold=0.0, **kwargs):
        if kwargs.get('use_sp', False):
            raise NotImplementedError(
                "WanT2V_Optimised step caching is single-GPU only for now; "
                "run without --ulysses_size > 1.")
        super().__init__(*args, **kwargs)
        self.cache_threshold = cache_threshold
        # Attach caching to each expert AFTER the originals are loaded and
        # configured; bind the cached forward in place (no reload).
        for model in (self.low_noise_model, self.high_noise_model):
            init_cache_state(model, cache_threshold)
            model.forward = types.MethodType(cached_dit_forward, model)

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""Identical to ``WanT2V.generate`` plus TeaCache step caching."""
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            # --- TeaCache: clear per-generation state on both experts --------
            reset_cache(self.low_noise_model)
            reset_cache(self.high_noise_model)
            prev_model = None
            # -----------------------------------------------------------------

            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                model = self._prepare_model_for_timestep(
                    t, boundary, offload_model)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                # Force a real compute on the first/last 2 steps (protect
                # structure + detail) and whenever the active expert changes.
                force = (i < 2) or (i >= len(timesteps) - 2) or (
                    model is not prev_model)
                prev_model = model

                noise_pred_cond = model(
                    latent_model_input, t=timestep, cache_tag='cond',
                    force_recompute=force, **arg_c)[0]
                noise_pred_uncond = model(
                    latent_model_input, t=timestep, cache_tag='uncond',
                    force_recompute=force, **arg_null)[0]

                noise_pred = noise_pred_uncond + sample_guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.low_noise_model.cpu()
                self.high_noise_model.cpu()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
