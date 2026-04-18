# 文献深度调研 SOP (Paper Deep-Review Standard Operating Procedure)

本 SOP 定义了在 `pancreatic-cancer-agent/article-review/` 中撰写**可投稿级**论文深度调研报告的标准流程。

**设计依据**: `deep-field-radar` SKILL Phase 1 (骨折级解构) + 学术同行评议标准
**适用范围**: 矩阵中任何论文的 `research/res_*.md` 精读页

---

## 六步法: 查 → 问 → 改 → 写 → 审 → 订

### Step 1: 查 (Search & Acquire)

**目标**: 确保拿到论文**全文** + 所有**视觉资产** (架构图/表格/结果图)

```
Checklist:
☐ 论文全文 (Markdown / PDF 视觉阅读)
☐ Supplementary Materials (如有)
☐ GitHub 仓库 (如有) — 读 README + 核心代码
☐ 架构图 [ARCH] 已提取 → article-review/_assets/<短名>_arch.png
☐ 结果图 [RESULT] 已识别
```

**来源优先级**: arXiv HTML > Nature OA > iPubMed PDF > 用户手动提供

### Step 2: 问 (Question & Decompose)

**目标**: 带着"如果给我代码和数据，我能复现吗？"的标准，提出结构化问题清单

**强制提问框架 (6 个维度)**:

| 维度 | 必答问题 |
|:-----|:---------|
| **数据** | 数据集规模? 公开/私有? 标注粒度? 标注者身份? 类别分布? |
| **模型** | 骨干网络? 损失函数? 优化器+学习率? Batch Size? 训练轮数? 硬件? |
| **Pipeline** | 几个模块? 每个模块的输入→输出格式? 模块间数据如何流转? |
| **评估** | 指标定义? 基线方法? 消融实验? 统计检验方法? |
| **临床** | 有无临床医生参与? 前瞻性验证? 参考标准 (Gold Standard) 是什么? |
| **复现** | 权重可用? 代码完整? 依赖链? 社区活跃度? |

### Step 3: 改 (Revise & Cross-Reference)

**目标**: 用外部证据**交叉验证**论文中的每一个关键断言

```
操作:
1. 对论文中引用的关键数据 (如 Kappa 值) → 查原始出处确认准确性
2. 对声称"首次" / "唯一" 的论断 → 搜索是否真的没有先例
3. 对"开源"声明 → 实际检查 GitHub Releases / HuggingFace
4. 对性能声称 → 与同期同任务的其他方法做横向对比
```

### Step 4: 写 (Write — 骨折级解构报告)

**目标**: 按照以下模板写入 `research/res_<短名>.md`

```markdown
# <论文短名>: <全标题> — DFR Phase 1 骨折级解构

![Architecture](../_assets/<短名>_arch.png)

---

## 1. 研究链路图谱 (Research Chain)
### 1.1 观察 (Observation) — 带量化数据 + 来源
### 1.2 假设 (Hypotheses) — 表格: #/假设/对应实验
### 1.3 关键领域概念解释 — 对非本领域读者的术语注释

## 2. 技术方案深度解构 (Technical Architecture)
### 2.0 总览 Pipeline — ASCII 流程图
### 2.N 模块 N — 表格: 输入/输出/算法/参数/代码文件/行数

## 3. 数据集构成与临床参与
### 3.1 数据集表格 — 规模/机构/公私/时间/用途
### 3.2 临床参与分析表格
### 3.3 核心性能指标表格

## 4. 代码与可复现性

## 5. 未披露细节清单 — 表格: #/内容/影响程度/为什么重要

## 6. 对我们项目的迁移价值
### ✅ 可直接复用
### 🔧 需要改造
### ❌ 明确不适用
```

**写作红线** (违反任何一条即不合格):
- ❌ 不得出现无来源的数据断言 (全程强制溯源)
- ❌ 不得使用模糊语言: "大量数据" "效果显著提升" "一定程度上"
- ❌ 不得混淆 Clinical Gap 和 Technical Gap
- ❌ 不得将单篇论文的 Limitation 等同于 Knowledge Gap

### Step 5: 审 (Review & Audit)

**目标**: 自查报告质量

```
Audit Checklist:
☐ 每个数据断言都有 (来源: 作者, 年份, 期刊/DOI)?
☐ 技术细节达到"可复现"粒度? (输入维度、Loss、LR、Batch 全部标注)
☐ Pipeline 流程图清晰, 模块间数据流转明确?
☐ 未披露细节是否逐条列出并评估影响?
☐ 架构图已嵌入且路径使用相对路径?
☐ 性能表格包含与基线的对比?
☐ 对项目的迁移价值分为三级 (✅/🔧/❌)?
```

### Step 6: 订 (Finalize & Sync)

**目标**: 同步到 literature_matrix.md + 确认链接完整性

```
操作:
1. 更新 literature_matrix.md 中对应行的"技术细节"列
   — 从一句话摘要 → 换为关键技术参数的精简版
2. 确认 Wiki Doc 链接指向更新后的 res_*.md
3. 确认架构图链接在 Obsidian Preview 中正常渲染
4. git add + commit 提交变更
```

---

## 质量基准: res_PAN-VIQ.md

`res_PAN-VIQ.md` 是本 SOP 的**标杆实现 (Gold Standard)**。

所有后续的 `res_*.md` 文件在深度、粒度、格式上不应低于该文件。

---

## 与 deep-field-radar SKILL 的关系

| 维度 | deep-field-radar | 本 SOP |
|:-----|:----------------|:-------|
| 范围 | 以种子论文为锚点, 扫描整个研究领域 (7 Phases) | 仅做单篇论文的 Phase 1 级深度解构 |
| 输出 | 领域图谱 + Gap + Standing Point | 单篇骨折级报告 |
| 使用时机 | 首次探索一个新领域 | 对矩阵中每篇论文做精读 |
| 互补性 | DFR Phase 1 → 调用本 SOP 的模板 | 本 SOP 的输出 → 喂入 DFR Phase 4/6 |
