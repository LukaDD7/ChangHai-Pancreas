#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速验证 LLaVA-Med 模型下载
"""

import os
import sys

sys.path.insert(0, '/media/luzhenyang/project/ChangHai_PDA/data/models/LLaVA-Med')

print("="*70)
print("LLaVA-Med 模型验证")
print("="*70)

# 1. 检查文件
model_dir = "/media/luzhenyang/project/ChangHai_PDA/data/models/llava-med-v1.5-mistral-7b"
print(f"\n模型目录: {model_dir}")

required_files = [
    "config.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "tokenizer.model",
    "tokenizer_config.json",
]

print("\n文件检查:")
all_exist = True
for f in required_files:
    path = os.path.join(model_dir, f)
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    size = os.path.getsize(path) / 1024**3 if exists else 0
    print(f"  {status} {f} ({size:.2f} GB)" if exists else f"  {status} {f}")
    if not exists:
        all_exist = False

if not all_exist:
    print("\n❌ 模型文件不完整")
    sys.exit(1)

print("\n✅ 所有模型文件已下载")

# 2. 检查 GPU
print("\n" + "="*70)
print("GPU 检查")
print("="*70)

try:
    import torch
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    显存: {props.total_memory / 1024**3:.1f} GB")
        print("\n✅ GPU 检查通过")
    else:
        print("\n❌ CUDA 不可用")
        sys.exit(1)

except ImportError:
    print("❌ PyTorch 未安装")
    sys.exit(1)

print("\n" + "="*70)
print("结论")
print("="*70)
print("""
✅ LLaVA-Med v1.5 模型已成功下载到本地！

模型路径: /media/luzhenyang/project/ChangHai_PDA/data/models/llava-med-v1.5-mistral-7b
模型大小: ~15 GB

模型已准备就绪，可以进行本地推理。
要启动推理服务，请运行:

  conda activate llava-med
  python -m llava.serve.controller --host 0.0.0.0 --port 10000

  # 另一个终端
  conda activate llava-med
  python -m llava.serve.model_worker \
    --host 0.0.0.0 \
    --controller http://localhost:10000 \
    --port 40000 \
    --worker http://localhost:40000 \
    --model-path /media/luzhenyang/project/ChangHai_PDA/data/models/llava-med-v1.5-mistral-7b \
    --multi-modal
""")
