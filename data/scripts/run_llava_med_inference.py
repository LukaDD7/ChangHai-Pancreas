#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLaVA-Med 推理脚本 - 用于胰腺肿瘤视觉分析
"""

import sys
import os
import torch
from PIL import Image

# 添加 LLaVA-Med 到路径
sys.path.insert(0, '/media/luzhenyang/project/ChangHai_PDA/data/models/LLaVA-Med')

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.conversation import conv_templates

def run_llava_med_inference(image_path, prompt, model_path="/media/luzhenyang/project/ChangHai_PDA/data/models/llava-med-v1.5-mistral-7b"):
    """
    运行 LLaVA-Med 推理

    Args:
        image_path: 输入图像路径
        prompt: 文本提示
        model_path: 模型路径

    Returns:
        模型生成的文本回复
    """
    print("="*70)
    print("LLaVA-Med 推理")
    print("="*70)
    print(f"图像: {image_path}")
    print(f"提示: {prompt}")
    print(f"模型: {model_path}")

    # 检查文件
    if not os.path.exists(image_path):
        print(f"❌ 图像文件不存在: {image_path}")
        return None

    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        return None

    # 加载模型
    print("\n加载模型...")
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device='cuda'
    )

    print(f"✅ 模型加载完成")
    print(f"   上下文长度: {context_len}")

    # 加载图像
    print("\n处理图像...")
    image = Image.open(image_path).convert('RGB')
    print(f"   图像尺寸: {image.size}")

    # 处理图像
    images_tensor = process_images([image], image_processor, model.config)
    images_tensor = images_tensor.to(model.device, dtype=torch.float16)

    # 准备对话
    print("\n准备对话...")
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{prompt}")
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    print(f"完整提示:\n{prompt_text}")

    # 编码输入
    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        image_token_index=tokenizer.convert_tokens_to_ids("<image>"),
        return_tensors='pt'
    ).unsqueeze(0).to(model.device)

    # 生成回复
    print("\n生成回复...")
    print("(这可能需要 10-30 秒)")

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            max_new_tokens=512,
            use_cache=True,
            temperature=0.2,
            top_p=0.95,
        )

    # 解码输出
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    print("\n" + "="*70)
    print("LLaVA-Med 回复")
    print("="*70)
    print(outputs)
    print("="*70)

    return outputs

if __name__ == "__main__":
    base_dir = "/media/luzhenyang/project/ChangHai_PDA/data"

    image_path = f"{base_dir}/results/images/CL-03356_master_slice.png"
    model_path = f"{base_dir}/models/llava-med-v1.5-mistral-7b"

    # 医学影像分析提示
    prompt = """Evaluate this CT image for any signs of Pancreatic Ductal Adenocarcinoma (PDAC).
Pay attention to the pancreatic head and major vessels like SMA.

Specifically look for:
1. Any hypo-attenuating (darker) lesions in the pancreatic head
2. Irregular contours or mass effect
3. Relationship with surrounding vessels (SMA, SMV, portal vein)
4. Evidence of ductal dilation or atrophy

Provide a detailed radiological assessment."""

    result = run_llava_med_inference(image_path, prompt, model_path)

    # 保存结果
    if result:
        output_file = f"{base_dir}/results/json/CL-03356_llava_med_report.txt"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("LLaVA-Med 视觉分析报告 - CL-03356\n")
            f.write("="*70 + "\n\n")
            f.write(f"图像: CL-03356_master_slice.png\n")
            f.write(f"位置: 胰腺最大面积层 (Z=145)\n\n")
            f.write("提示:\n")
            f.write(prompt + "\n\n")
            f.write("="*70 + "\n")
            f.write("模型回复:\n")
            f.write("="*70 + "\n\n")
            f.write(result)
        print(f"\n✅ 报告已保存: {output_file}")
