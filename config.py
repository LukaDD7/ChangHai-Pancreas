import os

# --- 1. THE BRAIN (Tongyi Qwen3.5-Plus) ---
BRAIN_MODEL_NAME = "qwen3.5-plus"  # 阿里云最新最强推理模型
BRAIN_API_KEY = os.getenv("DASHSCOPE_API_KEY")
BRAIN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --- 2. THE EYES (Tongyi Qwen-VL) ---
VLM_MODEL_NAME = "qwen3.5-plus"  # 处理复杂的 CT 影像分析
VLM_API_KEY = os.getenv("DASHSCOPE_API_KEY")
VLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --- 3. Medical Agent Python Path ---
MEDICAL_PYTHON_PATH = "/home/luzhenyang/anaconda3/envs/ChangHai/bin/python"
