# from: https://github.com/NVIDIA/TransformerEngine?tab=readme-ov-file#examples
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe


import pandas as pd
import numpy as np
import time


pd.set_option("display.precision", 4)

# data type
dtype = torch.bfloat16
# number of iterations after 3 warmup iterations
num_iters = 3
# checkpointing
ckpt_attn = False
# workspace optimization path for cuDNN attention
workspace_opt = True
# QKV memory layout
qkv_layout = "bshd_bshd_bshd"
# padding between sequences for qkv_format=thd
pad_between_seqs = False
# training mode
is_training = True
"""
model_configs = {
    #   test: batch_size, heads, num_gqa_heads, head_dim_qk, max_seqlen_q, max_seqlen_kv, dropout_p, mask, bias
    "test_0": ModelConfig(2, 16, 16, 64, 512, 512, 0.0, "no_mask", "no_bias"),  # short seq
    "test_1": ModelConfig(2, 16, 16, 128, 2048, 2048, 0.0, "causal", "no_bias"),  # longer seq, mask
    "test_2": ModelConfig(2, 16, 16, 128, 2048, 2048, 0.0, "causal", "post_scale_bias"),  # bias
    "test_3": ModelConfig(2, 32, 4, 128, 8192, 8192, 0.0, "causal", "no_bias"),  # GQA
}
"""

# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

# Init cuda profiler
torch.cuda.cudart().cudaProfilerStart()
torch.cuda.synchronize()
# dpa: Dot product attention
dpa_start = time.time()

# DotProduct with FP8
for i in range(num_iters):
# Enable autocasting for the forward pass
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        # Is this an op of dpa or a block of dpa?
        te.DotProductAttention(
            num_attention_heads=16,
            kv_channels=128,
            num_gqa_groups=16,
            attention_dropout=0.0,
            qkv_format=qkv_layout,
            attn_mask_type="causal"
        )
    
torch.cuda.synchronize()
dpa_total_time = (time.time() - dpa_start) * 1e3
dpa_indivual_time = dpa_total_time / num_iters
print(f"indidual time per dot product op: {dpa_indivual_time}ms")
print(f"Total time per {num_iters} iterations of dot product op: {dpa_total_time}ms")
