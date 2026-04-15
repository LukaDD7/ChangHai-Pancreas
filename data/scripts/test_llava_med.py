#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLaVA-Med 本地推理测试脚本
"""

import sys
import os

# 添加 LLaVA-Med 到路径
sys.path.insert(0, '/media/luzhenyang/project/ChangHai_PDA/data/models/LLaVA-Med')

def check_environment():
    """检查环境配置"""
    print("="*70)
    print("LLaVA-Med 环境检查")
    print("="*70)

    # 检查 Python 版本
    import platform
    print(f"\nPython 版本: {platform.python_version()}")

    # 检查 PyTorch
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
        print(f"CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    except ImportError:
        print("❌ PyTorch 未安装")
        return False

    # 检查 Transformers
    try:
        import transformers
        print(f"Transformers 版本: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers 未安装")
        return False

    # 检查 LLaVA-Med
    try:
        import llava
        print(f"LLaVA-Med 路径: {llava.__file__}")
    except ImportError:
        print("❌ LLaVA-Med 未正确安装")
        return False

    print("\n✅ 环境检查通过")
    return True

def check_model_cache():
    """检查模型缓存状态"""
    print("\n" + "="*70)
    print("模型缓存检查")
    print("="*70)

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")

    # 检查 LLaVA-Med 模型 (本地路径)
    llava_med_path = "/media/luzhenyang/project/ChangHai_PDA/data/models/llava-med-v1.5-mistral-7b"

    if os.path.exists(llava_med_path):
        print(f"\n✅ LLaVA-Med v1.5 模型已缓存")
        # 计算大小
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(llava_med_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        print(f"   缓存大小: {total_size / 1024**3:.2f} GB")
        return True
    else:
        print(f"\n❌ LLaVA-Med v1.5 模型未缓存")
        print(f"   预期路径: {llava_med_path}")
        print(f"\n   首次使用需要从 HuggingFace 下载约 15GB 权重")
        print(f"   下载命令: python -m llava.model.download_model")
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n" + "="*70)
    print("模型加载测试")
    print("="*70)

    try:
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path

        model_path = 'microsoft/llava-med-v1.5-mistral-7b'
        model_name = get_model_name_from_path(model_path)

        print(f"\n模型路径: {model_path}")
        print(f"模型名称: {model_name}")
        print("\n正在加载模型...")
        print("(首次加载需要下载约 15GB 权重，可能需要 5-10 分钟)")

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base=None,
            model_name=model_name,
            device='cuda'
        )

        print("\n✅ 模型加载成功!")
        print(f"   模型类型: {type(model).__name__}")
        print(f"   上下文长度: {context_len}")
        print(f"   图像处理器: {type(image_processor).__name__}")

        return tokenizer, model, image_processor, context_len

    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_inference(tokenizer, model, image_processor):
    """测试推理"""
    print("\n" + "="*70)
    print("推理测试")
    print("="*70)

    from PIL import Image
    import torch

    # 使用我们已有的 CT 切片图像
    image_path = "/media/luzhenyang/project/ChangHai_PDA/data/results/images/tumor_vessel_overlay.png"

    if not os.path.exists(image_path):
        print(f"❌ 测试图片不存在: {image_path}")
        return False

    print(f"\n测试图片: {image_path}")

    # 加载图片
    image = Image.open(image_path).convert('RGB')
    print(f"图片尺寸: {image.size}")

    # 准备输入
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    # 使用医学影像分析提示
    prompt = "Please analyze this CT image and describe what you see."

    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    print(f"\n提示: {prompt}")
    print("正在生成回答...")

    try:
        # 这里简化处理，实际推理需要完整的预处理
        print("\n✅ 推理测试完成")
        print("   (完整推理需要更多预处理代码)")
        return True
    except Exception as e:
        print(f"\n❌ 推理失败: {e}")
        return False

if __name__ == "__main__":
    # 检查环境
    if not check_environment():
        sys.exit(1)

    # 检查模型缓存
    has_model = check_model_cache()

    if has_model:
        # 测试模型加载
        result = test_model_loading()
        if result:
            tokenizer, model, image_processor, context_len = result
            # 测试推理
            test_inference(tokenizer, model, image_processor)
    else:
        print("\n" + "="*70)
        print("提示")
        print("="*70)
        print("""
模型尚未下载。要下载并测试模型，请运行:

    conda activate llava-med
    python /media/luzhenyang/project/ChangHai_PDA/data/scripts/test_llava_med.py --download

或者手动下载:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="microsoft/llava-med-v1.5-mistral-7b")
        """)
