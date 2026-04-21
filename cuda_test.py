import torch
print(torch.version.cuda)          # PyTorch 编译时对应的 CUDA 版本
print(torch.cuda.is_available())   # 是否在用 GPU
