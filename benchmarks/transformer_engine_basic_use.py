# from: https://github.com/NVIDIA/TransformerEngine?tab=readme-ov-file#examples
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe


import pandas as pd
import numpy as np
import time
import nvtx


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
qkv_layout = "bshd"
# padding between sequences for qkv_format=thd
pad_between_seqs = False
# training mode
is_training = True

# Set dimensions.
seq_len = 128
b_size = 1
emb_size = 512
num_heads = 8
head_size = int(emb_size/num_heads)


rand_query = torch.randn(b_size, seq_len, num_heads, head_size, device="cuda")
rand_key = torch.randn(b_size, seq_len, num_heads, head_size, device="cuda")
rand_value = torch.randn(b_size, seq_len, num_heads, head_size, device="cuda")

# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

@nvtx.annotate(color="blue")
def DotProductAttention_operator():
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        with nvtx.annotate("dpa_loop", color="red"):
        # Is this an op of dpa or a block of dpa? -> te.DotProductAttention() init the block and .forward() performs the op
            te.DotProductAttention(
                num_attention_heads=8,
                # kv_channels: Head size of key and value tensors
                kv_channels= head_size,
                attention_dropout=0.0,
                qkv_format=qkv_layout,
                attn_mask_type="causal"
            ).forward(query_layer=rand_query, key_layer=rand_key, value_layer=rand_value)
                
# Init cuda profiler
torch.cuda.cudart().cudaProfilerStart()
torch.cuda.synchronize()

# dpa: Dot product attention
dpa_start = time.time()

# DotProduct with FP8
for i in range(num_iters):
# Enable autocasting for the forward pass
    DotProductAttention_operator()

torch.cuda.synchronize()

dpa_total_time = (time.time() - dpa_start) * 1e3
dpa_indivual_time = dpa_total_time / num_iters
print(f"indidual time per dot product op: {dpa_indivual_time}ms")
print(f"Total time per {num_iters} iterations of dot product op: {dpa_total_time}ms")


# nsys profile -t nvtx  python transformer_engine_basic_use.py
# ncu --section Occupancy -o occupancy python transformer_engine_basic_use.py