# Agentic RAG — PDF Knowledge QA System

> 基于 LangGraph 构建的多节点 Agentic RAG 系统，实现对本地 PDF 文档的智能问答。
> 项目经历了从手动 Pipeline 到 LlamaIndex 框架重构、从单路向量检索到混合检索的完整演进，核心目标是将 RAG 作为**系统设计问题**而非 API 调用来对待。

---

## 系统架构

```
用户输入
   │
   ▼
┌──────────┐
│  Planner │  判断意图：需要检索 or 直接回答
└──────────┘
   │
   ▼
┌──────────┐   调用工具     ┌──────────────────────┐
│  Agent   │ ────────────▶ │  Tools               │
│  执行层   │               │  search_pdf_tool     │
└──────────┘ ◀──────────── │  submit_final_answer │
   │          工具结果      └──────────────────────┘
   │
   │  submit_final_answer
   ▼
┌──────────┐
│ Evaluator│  Faithfulness 检验，不合格回退 Planner
└──────────┘
   │
   ├── pass → END
   └── fail → Planner（最多3轮）
```

| 节点 | 职责 |
|------|------|
| **Planner** | 只看当前用户输入，判断是否需要检索，指定工具名。最小权限设计，不接触历史消息，不生成答案 |
| **Agent** | 根据 Planner 决策执行。`action=tool` 时使用 `llm_with_tools`，`action=respond` 时使用 `llm_plain`，避免 Forced Tool Calling |
| **Tools** | 执行 RAG 检索，校验工具调用是否符合 Planner 预期，拒绝越权调用并将原因写回消息历史 |
| **Evaluator** | 独立 LLM 做 Faithfulness 检验，通过 `tool_call_id` 精确匹配检索文档，不合格时回退 Planner 重新规划 |

---

## RAG Pipeline

### 混合检索（Hybrid Search）

```
Query
  │
  ├── Multi-query 改写（3个子问题）
  │
  ├─▶ BGE-M3 向量检索（Bi-Encoder）× 4路 → 合并去重
  │
  ├─▶ BM25 关键词检索（jieba 中文分词）
  │
  ├── RRF 算法融合两路排名
  │
  ├─▶ BGE-Reranker-v2-m3 精排（Cross-Encoder）→ Top-5
  │
  └── Context Expansion（扩展前后相邻 chunk）
        │
        ▼
      返回 Agent
```

**为什么是混合检索：**

- BGE-M3（Bi-Encoder）：Query 和文档独立编码，向量相似度计算，速度快，适合语义召回
- BM25：关键词精确匹配，弥补向量检索对专有术语、型号、公式的不足
- RRF：两路结果排名融合，无需对齐分数量纲，是混合检索的工业标准做法
- BGE-Reranker-v2-m3（Cross-Encoder）：Query 和文档拼接后做完整 Attention 计算，精度高，只用于精排少量候选

**为什么 Multi-query：**

单一 query 召回覆盖率有限，复杂问题容易遗漏相关段落。将原始问题改写为 3 个不同角度的子问题，分别检索后合并去重，再统一精排。

### 语义分块

使用 LlamaIndex `SemanticSplitterNodeParser`，自适应百分位数切割：

- `breakpoint_percentile_threshold=95`：只切语义跳跃最突兀的 5% 位置
- 相比固定阈值，在不同风格文档间无需重新调参
- 避免关键信息跨 chunk 断裂（Context Fragmentation）

### 向量存储

Qdrant 内存模式，BGE-M3 输出经 L2 归一化后存储。归一化后 Cosine 相似度等价于点积计算，降低检索开销。

> **当前限制**：内存模式不持久化，重启后需重新建索引。单文件单线程，不支持并发。

---

## 关键设计决策

### 为什么引入 Planner

初版直接使用绑定工具的 LLM 作为 Agent。实际运行中，当用户提问不涉及 PDF 内容时，LLM 仍会尝试生成工具调用格式的输出，触发 provider 的 `BadRequestError`（Forced Tool Calling 问题）。

引入独立 Planner 节点，使用未绑定工具的 `llm_plain`，将意图判断与工具执行解耦。Planner 只看当前问题，不传入历史消息，避免历史中的工具调用模式干扰判断。

### 为什么 Evaluator 回退到 Planner 而非 Agent

Evaluator 判定质量不合格时，根本原因可能是检索策略本身错误（搜错了方向），而不只是生成质量差。直接回退 Agent 只是重试，不改变检索策略。回退 Planner 让整个决策链重新运行，有机会修正更上游的问题。

### Faithfulness 检验替代自评置信度

原始设计依赖模型自评 `confidence` 字段触发重试，存在自我评估偏差（Self-evaluation Bias）。当前版本引入独立 Evaluator LLM，将回答和检索文档一起传入，判断每个陈述是否有文档依据。通过 `tool_call_id` 精确匹配对应的 `ToolMessage`，而非搜索消息内容字符串。

### 初版 → 当前版本的演进

| 版本 | 检索方式 | 分块方式 | Evaluator |
|------|---------|---------|-----------|
| v1 | 手动 BGE-M3 单路向量检索 | Chonkie SemanticChunker（固定阈值） | 模型自评 confidence |
| v2 | LlamaIndex 单路向量检索 | LlamaIndex SemanticSplitter（自适应） | 模型自评 confidence |
| v3（当前）| 混合检索 + RRF + Multi-query | LlamaIndex SemanticSplitter | 独立 Evaluator Faithfulness 检验 |

---

## 已知局限与后续方向

**当前局限：**

- Qdrant 内存模式，重启需重新索引
- Evaluator 仍依赖 LLM 判断，缺乏量化评估指标
- 不支持多文档对比问答
- 上下文扩展基于 node 顺序而非页面布局，跨页内容可能引入噪音

**后续方向：**

- 接入 RAGAS 框架，量化评估 Faithfulness / Answer Relevance / Context Precision / Context Recall
- Memory as RAG：历史对话向量化存储，按需检索，替代 MemorySaver 全量上下文
- Multi-agent 架构：Orchestrator 调度独立 Retrieval Worker 和 QA Worker
- Qdrant 持久化模式，支持增量索引更新

---

## 技术栈

| 组件 | 选型 |
|------|------|
| Agent 框架 | LangGraph |
| RAG 框架 | LlamaIndex |
| Embedding 模型 | BAAI/bge-m3 |
| Reranker 模型 | BAAI/bge-reranker-v2-m3 |
| 关键词检索 | BM25Okapi + jieba |
| 向量数据库 | Qdrant（内存模式） |
| LLM | Qwen-plus（OpenAI 兼容接口） |
| PDF 解析 | PyMuPDF |
| 终端渲染 | Rich |

---

## 快速开始

```bash
git clone https://github.com/APushingBoy/agentic_rag_langgraph.git
cd agentic-rag-pdf

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 运行
python main.py
# 根据提示输入 PDF 完整路径
```

---