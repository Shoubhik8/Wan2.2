# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""
Entry point for the torch.compile'd ti2v-5B path (optimizations.md, Step 4).

This reuses all of generate.py's argument parsing and orchestration verbatim and
only swaps the ti2v pipeline class for the compiled subclass. Running THIS script
turns compile ON; running the stock generate.py leaves it OFF. No existing file is
modified.

Example:
    cd Wan2.2
    TORCH_LOGS=recompiles python generate_compiled_optimized.py \
        --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B \
        --offload_model True --convert_model_dtype \
        --sample_steps 25 --prompt "a cat playing piano"
"""
import logging

import generate
import wan
from wan.textimage2video_compiled_optimized import WanTI2VCompiled

# generate.py builds the ti2v pipeline via `wan.WanTI2V(...)` (module attribute
# access), so rebinding the attribute here routes ti2v through the compiled subclass.
wan.WanTI2V = WanTI2VCompiled

if __name__ == "__main__":
    args = generate._parse_args()
    if "ti2v" not in args.task:
        logging.warning(
            f"[compiled_optimized] task '{args.task}' is not a ti2v task; the "
            f"torch.compile path only applies to ti2v-5B. Other tasks run "
            f"exactly as in the stock generate.py.")
    generate.generate(args)
