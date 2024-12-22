import torch
# 设置要使用的 CUDA 设备索引
cuda_device = 0

# 检查是否有可用的 CUDA GPU
if torch.cuda.is_available():
    # 设置要使用的 CUDA 设备
    torch.cuda.set_device(cuda_device)
    print(f"Using CUDA device: {torch.cuda.get_device_name(cuda_device)}")
else:
    print("No CUDA GPUs available.")
