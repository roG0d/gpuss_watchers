import torch
import triton


# HBM Memory Bandwidth 

# Tensor size: 16GB
tensor_size = 16 * 1024**3 # 16GB in bytes
dtype = torch.float32 # Assuming FP32 (4 bytes per element)
num_elements = tensor_size // 4 # Calculate number of elements


# Allocate tensor on the device
a = torch.randn(num_elements, device="cuda", dtype=dtype)
b = torch.empty_like(a)

# Function to benchmark
def copy_tensor():
    b.copy_(a)

# Benchmark using Triton
time_ms = triton.testing.do_bench(copy_tensor)

# Calculate bandwidth
bandwitdh_gbps = (tensor_size * 2) / (time_ms * 1e-3) / 1e9 # Multiply by 2 for read + write

# Print results
print(f"Copy Bandwidth: {bandwitdh_gbps:.2f} GB/s (time: {time_ms:.2f} ms)")