## 放一个mermaid图看看

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        agent(agent)
        tools(tools)
        evaluator(evaluator)
        __end__([<p>__end__</p>]):::last
        __start__ --> agent;
        agent -. &nbsp;end&nbsp; .-> __end__;
        agent -. &nbsp;reflect&nbsp; .-> evaluator;
        agent -. &nbsp;continue&nbsp; .-> tools;
        evaluator -. &nbsp;end&nbsp; .-> __end__;
        evaluator -. &nbsp;re-think&nbsp; .-> agent;
        tools --> agent;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```

##  项目简介

本项目实现了一个面向 PDF 文档问答的 **Agentic RAG（检索增强生成）系统**。

相比基础的 RAG pipeline，本项目重点在于：

* 引入 **可观测的检索流程**
* 区分 **召回（recall）与精排（precision）**
* 结合 **Agent + 反思机制（reflection loop）**
* 将 RAG 作为一个**系统设计问题**进行实现，而不仅仅是调用 API

---

##  当前功能

###  两阶段检索（Retrieval Pipeline）

* 使用 **BGE-M3** 进行向量检索（高召回）
* 使用 **BGE Reranker** 进行精排（高精度）
* 采用 **Top-K 策略**，替代固定阈值，提升稳定性

---

###  Agent 工作流（基于 LangGraph）

* 基于 **LangGraph** 构建多节点流程
* 支持：

  * 工具调用（PDF 检索）
  * 多轮推理
  * 反思与重试机制

---

###  反思机制（Reflection Loop）

* 引入 evaluator 节点对回答质量进行评估
* 在以下情况触发重试：

  * 置信度较低
  * 回答内容不足

---

###  检索过程可观测

提供详细调试信息：

* 向量召回结果（Top-K）
* Rerank 排序结果及分数

用于分析：

* 检索是否命中正确内容
* rerank 是否合理
* query 是否存在偏差

---

###  文档处理流程

* PDF → 文本（MarkItDown）
* 语义分块（Semantic Chunking）
* 向量化并存入 Qdrant（内存模式）

---

###  模块化设计

* 检索逻辑（RAGEngine）与 Agent 解耦
* 支持后续独立优化：

  * retrieval
  * agent
  * evaluation

---

##  设计思路

本项目的核心目标是：

> **让 RAG 系统从“黑盒调用”变为“可分析、可控制的流程”**

具体体现为：

* 明确区分 embedding 与 rerank 的职责
* 引入反思机制控制回答质量
* 增强检索过程的可解释性
* 避免依赖模型自带的“置信度幻觉”

---

##  后续优化方向

###  Retrieval 方向

* Multi-query（查询扩展 / 分解）
* 动态 Top-K（基于分数分布）
* 更结构化的 chunk（基于章节 / 布局）
* 引入 metadata（页码、章节等）

---

###  Agent 方向

* 基于检索质量的决策（而非简单规则）
* 区分失败类型（检索失败 / 推理失败 / 幻觉）
* 更智能的重试策略

---

###  评估与可靠性

* 自动化评估 pipeline（正确性 / 引用一致性）
* 更可靠的置信度机制
* 系统级日志与指标分析

---

###  多模态扩展（未来方向）

* 支持 PDF 中的图片 / 表格
* 引入多模态 embedding

---

##  项目状态

当前版本为一个**功能完整的原型系统**，正在逐步向：

> **更稳定、可控的 Agentic RAG 系统**

演进。

