# -*- coding: utf-8 -*-
"""将 pytorch_model.bin 转换为 model.safetensors（transformers 5.x + torch<2.6 必需）"""
import torch
from safetensors.torch import save_file

P = r"E:\agent1\data\models\bge-m3\pytorch_model.bin"
OUT = r"E:\agent1\data\models\bge-m3\model.safetensors"

sd = torch.load(P, map_location="cpu", weights_only=True)
sd = {k: v.clone().contiguous() for k, v in sd.items()}
save_file(sd, OUT, metadata={"format": "pt"})
print("saved:", OUT, "tensors:", len(sd), flush=True)
