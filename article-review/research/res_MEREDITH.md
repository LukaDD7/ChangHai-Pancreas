---
title: "Research: MEREDITH"
created: 2026-04-17
updated: 2026-04-17
paper: "MEREDITH: Precision Oncology Treatment Recommendation"
type: method_paper
domain: [Engineering, Interdisciplinary]
tags: [agent, oncology, RAG, treatment-recommendation, Gemini]
mocs: (Interdisciplinary Index)
---

## 🎯 Research Overview
- **Problem:** 精准肿瘤治疗推荐耗时长、依赖专家，分子肿瘤委员会 (MTB) 会议覆盖有限。
- **Contribution:** Gemini Pro + RAG + CoT，接入 PubMed、OncoKB、临床试验数据库。
- **Impact:** 与 MTB 94.7% 一致率 (90 例回顾验证)。JCO Precision Oncology, Oct 2024; ASCO 2024。

## 🔬 Core Method
- LLM backbone: Gemini Pro
- 知识源: PubMed API, OncoKB, DrugBank, 临床试验数据库
- 推理: Chain-of-Thought (CoT)
- 验证: 与分子肿瘤委员会 (MTB) 实际决策对比

## 🔗 Connections
- **参考价值:** [Pancreatic_Cancer Architecture] MDT 中 "肿瘤内科 Agent" 的治疗推荐模块
- **GitHub:** https://github.com/LammertJ/MEREDITH

