# Agentic RAG 系统 V8 版本 - 深度混合检索与离线元数据驱动架构

本项目是一个基于 LangGraph 构建的高性能 Agentic RAG 系统。V8 版本通过“离线化索引”与“混合检索”双引擎驱动，配合智能路由与反思评估机制，实现了极高的工业级问答准确度，并原生支持 LangSmith 全链路监控。

##  核心特性

### 1. 深度混合检索引擎 (Hybrid Search)
系统集成了双路检索机制，确保在各种查询场景下的召回率：
- **BM25 关键词检索**：针对专有名词、产品型号、缩写等精确匹配场景表现卓越。
- **Vector 语义检索**：基于 HuggingFace 嵌入模型，捕捉用户问题的深层语义意图。
- **重排序 (Rerank)**：采用 Cross-Encoder 对双路结果进行二次打分，确保最相关的上下文排在首位。

### 2. V8 离线化元数据架构
- **Manifest 驱动**：彻底放弃实时路径输入。系统通过 `manifest.json` 管理文档库，包含每篇文档的摘要、关键词和核心要点。
- **智能路由 (Smart Routing)**：Planner 节点在检索前会根据元数据自动锁定目标 `doc_id`，避免在无关文档中浪费检索额度。

### 3. Agentic 闭环评估机制
- **Planner 节点**：负责多步规划，决定是直接回复、列出目录还是执行深度混合检索。
- **Evaluator 节点**：独立的反思模块，对生成的答案进行“事实性冲突”检验。若检索结果为空或答案置信度不足，系统将触发重试或严格执行拒绝回答策略。

### 4. 全方位可观测性 (LangSmith)
- 原生集成 **LangSmith** 监控，只需配置环境变量即可记录完整的 Agent 决策链路、工具调用参数及 Token 消耗。

##  技术栈

- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
- **Framework**: [LangChain](https://github.com/langchain-ai/langchain)
- **Monitoring**: [LangSmith](https://www.langchain.com/langsmith)
- **Search Engine**: BM25 + Vector (Qdrant)
- **LLM**: 通义千问 `qwen-plus-2025-12-01`
- **Hardware**: NVIDIA RTX 4090 (支持 CUDA 加速加速推理与嵌入)
- **Terminal UI**: [Rich](https://github.com/Textualize/rich)

##  系统架构流程

1. **意图解析 (Planner)**：分析问题，判断是否需要调用工具。
2. **路由寻址 (Router)**：通过 `manifest.json` 匹配最相关的文档。
3. **混合检索 (Hybrid Search)**：在选定文档中并行执行 BM25 与向量检索。
4. **答案生成 (Agent)**：融合检索片段生成带引用的回答。
5. **质量自检 (Evaluator)**：评估回答是否包含幻觉，不合格则返回 Planner 重新执行。

##  快速开始

### 1. 配置环境
在项目根目录创建 `.env` 文件：

```env
# LLM 访问
OPENAI_API_KEY=your_api_key
OPENAI_API_BASE=https://...

# LangSmith 监控 (建议开启)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=Agentic_RAG_V8

# 调试模式
DEBUG_MODE=true