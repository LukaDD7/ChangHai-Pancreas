import os
import sys
# 引入 LangChain 的封装，因为 DeepAgents 需要 LangChain 对象，而不是纯 OpenAI client
from langchain_openai import ChatOpenAI
from openai import OpenAI  # 保留原生 Client 给 VLM 用

# 动态加载根目录 config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def get_pdac_client_langchain():
    """
    [Main Agent] 获取支持 Qwen3.5-Plus 的 LangChain 对象
    """
    if not config.BRAIN_API_KEY:
        raise ValueError("❌ MISSING: DASHSCOPE_API_KEY in env.")

    return ChatOpenAI(
        model=config.BRAIN_MODEL_NAME,
        api_key=config.BRAIN_API_KEY,
        base_url=config.BRAIN_BASE_URL,
        temperature=0.6,
    )


def get_vlm_client():
    """
    [Vision Tool] 获取 Qwen-VL 的原生 Client (用于 VLM 分析)
    """
    if not config.VLM_API_KEY:
        key = config.VLM_API_KEY or os.getenv("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("❌ MISSING: DASHSCOPE_API_KEY (for Qwen-VL).")
        return OpenAI(api_key=key, base_url=config.VLM_BASE_URL)

    return OpenAI(
        api_key=config.VLM_API_KEY,
        base_url=config.VLM_BASE_URL
    )
