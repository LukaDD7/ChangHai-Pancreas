#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LLM Factory for ChangHai PDAC Agent
Provides LangChain-compatible model clients
"""

import os
from langchain_openai import ChatOpenAI
from openai import OpenAI


def get_pdac_client_langchain():
    """
    Get LangChain-compatible chat model for PDAC analysis

    Returns:
        ChatOpenAI: Configured LangChain chat model
    """
    # Using OpenRouter for model access (configurable)
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "Please set environment variable: export OPENROUTER_API_KEY='your-key'"
        )

    model = ChatOpenAI(
        model="anthropic/claude-sonnet-4-6",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=4096,
    )

    return model


def get_vlm_client():
    """
    Get OpenAI-compatible client for VLM (Vision-Language Model)

    Returns:
        OpenAI: Configured OpenAI client for image analysis
    """
    # Using DashScope for Qwen-VL (or alternative VLM provider)
    api_key = os.environ.get("DASHSCOPE_API_KEY")

    if not api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY not set. "
            "Please set environment variable: export DASHSCOPE_API_KEY='your-key'"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    return client
