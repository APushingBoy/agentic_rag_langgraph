import os
import json
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import torch
from rank_bm25 import BM25Okapi
import jieba

# LlamaIndex 组件
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import QueryBundle, NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore

from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# 这里导入我们在管道中定义的全局索引类，确保架构对齐
# 如果在一个文件内，请确保 GlobalBM25Index 类在 RAGEngine 之前定义
from ingestion_pipeline import GlobalBM25Index

# ==========================================
# 配置常量
# ==========================================
MANIFEST_PATH = Path("./manifest.json")
QDRANT_PATH = "./qdrant_storage"
BM25_INDEX_PATH = Path("./bm25_global_index.pkl")
LLM_MODEL = "qwen-plus-2025-12-01"


# ==========================================
# 核心 RAG 引擎类 (V8 版本 - 对齐全局索引架构)
# ==========================================
class RAGEngine:
    def __init__(self):
        """
        初始化 RAG 引擎。
        我在这里不仅加载了向量库，还引入了真·全局 BM25 索引，
        确保我们的混合检索是基于全量语料库统计的。
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DEBUG: [RAG 引擎启动] 使用设备: {self.device}")
        
        # 1. 初始化 LLM（用于查询扩展和改写）
        self.query_llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        
        # 2. 配置 LlamaIndex 全局嵌入模型 (BGE-M3)
        # 我在初始化时会显式指定设备，避免面试时被问到资源调度问题
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            device=self.device
        )
        
        # 3. 初始化 Qdrant 向量库客户端
        self.qdrant_client = QdrantClient(path=QDRANT_PATH)
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="expert_collection"
        )
        self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        
        # 4. 加载真·全局 BM25 索引
        # 这是本次修改的核心：直接加载 ingestion 生成的全局池，不再扫描散装 pkl
        self.global_bm25 = GlobalBM25Index(BM25_INDEX_PATH)
        
        # 5. 初始化重排序模型 (Rerank)
        # 使用 bge-reranker-v2-m3 能够极大提升 Top-K 召回后的精度
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3",
            top_n=5,
            device=self.device
        )

    def _expand_query(self, query: str) -> List[str]:
        """
        利用 LLM 进行查询改写与扩展。
        我会生成多个维度的子问题，以应对向量检索在复杂表述下的召回不足。
        """
        print(f"DEBUG: [查询扩展] 原始 Query: {query}")
        prompt = f"""你是一个搜索专家。请将用户的提问拆解或改写为 3 个不同的搜索短语，
        以便从知识库中检索到更全面的内容。
        要求：
        1. 涵盖原始意图。
        2. 尝试不同的侧重点（概念、原理、应用）。
        3. 只输出短语，每行一个。

        用户提问：{query}
        """
        
        try:
            response = self.query_llm.invoke(prompt)
            expanded_queries = [line.strip() for line in response.content.split("\n") if line.strip()]
            # 我一定会把原始查询放在首位，确保基准召回
            all_queries = [query] + expanded_queries[:2] 
            return all_queries
        except Exception as e:
            print(f"ERROR: [查询扩展失败] {e}")
            return [query]

    def _bm25_search(self, query: str, doc_id: str = "all", top_k: int = 20) -> List[Tuple[float, Any]]:
        """
        基于全局 BM25 模型进行关键词检索。
        
        修改点：
        1. 不再循环多个文件，而是直接对全局模型打分。
        2. 支持 doc_id 过滤：如果指定了某个文档，我们会从全局结果中筛选出该文档的节点。
        """
        if not self.global_bm25.bm25_model:
            print("WARNING: [BM25] 全局模型未就绪")
            return []

        # 分词对齐：中英文分词逻辑必须与 Ingestion 阶段完全一致
        if any(ord(c) > 127 for c in query):
            query_tokens = jieba.lcut(query)
        else:
            query_tokens = query.lower().split()

        # 获取全局打分
        scores = self.global_bm25.bm25_model.get_scores(query_tokens)
        all_nodes = self.global_bm25.data.get("all_nodes", [])

        # 封装结果并应用 doc_id 过滤
        results = []
        for i, score in enumerate(scores):
            if score > 0:
                node = all_nodes[i]
                # 如果用户限定了文档，则只保留该文档的节点
                if doc_id != "all" and node.metadata.get("doc_id") != doc_id:
                    continue
                results.append((score, node))

        # 按分值排序
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def _expand_context(self, node: Any, doc_id: str) -> str:
        """
        上下文扩展：解决 Chunk 截断导致的语义缺失。
        既然我们有全局节点池，我可以轻松通过元数据找到该节点在原文档中的前后邻居。
        """
        # 从全局池中过滤出属于该文档的所有节点，并按顺序排列
        all_doc_nodes = [
            n for n in self.global_bm25.data.get("all_nodes", [])
            if n.metadata.get("doc_id") == doc_id
        ]
        
        # 寻找当前节点在文档流中的位置
        try:
            current_idx = -1
            for i, n in enumerate(all_doc_nodes):
                if n.node_id == node.node_id:
                    current_idx = i
                    break
            
            if current_idx == -1:
                return node.get_content()

            # 向上向下各取 1 个节点作为缓冲，增强语义完整性
            start_idx = max(0, current_idx - 1)
            end_idx = min(len(all_doc_nodes), current_idx + 2)
            
            expanded_text = ""
            for i in range(start_idx, end_idx):
                prefix = "[...继续上一段...] " if i == current_idx - 1 else ""
                suffix = " [...接下一段...]" if i == current_idx + 1 else ""
                content = all_doc_nodes[i].get_content()
                expanded_text += f"{prefix}{content}{suffix}\n\n"
            
            return expanded_text.strip()
        except Exception as e:
            print(f"DEBUG: [上下文扩展异常] {e}")
            return node.get_content()

    def _rrf_merge(self, vector_nodes: List[NodeWithScore], bm25_results: List[Tuple[float, Any]], k: int = 60) -> List[Any]:
        """
        互惠排名融合 (Reciprocal Rank Fusion, RRF)。
        这在面试中是必考点。我用它来平衡向量检索和 BM25 检索的结果。
        """
        rrf_scores = {}
        
        # 处理向量检索排名
        for rank, node_with_score in enumerate(vector_nodes):
            node_id = node_with_score.node.node_id
            rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1.0 / (k + rank + 1)
        
        # 处理 BM25 检索排名
        for rank, (score, node) in enumerate(bm25_results):
            node_id = node.node_id
            rrf_scores[node_id] = rrf_scores.get(node_id, 0) + 1.0 / (k + rank + 1)
        
        # 汇总去重后的节点字典，方便后续查找
        node_lookup = {n.node.node_id: n.node for n in vector_nodes}
        for _, node in bm25_results:
            if node.node_id not in node_lookup:
                node_lookup[node.node_id] = node
                
        # 按 RRF 分数重排
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        merged_nodes = []
        for node_id, _ in sorted_ids:
            # 统一封装为 NodeWithScore 类型供下游使用
            merged_nodes.append(NodeWithScore(node=node_lookup[node_id], score=rrf_scores[node_id]))
            
        return merged_nodes

    def search(self, query: str, doc_id: str = "all") -> str:
        """
        核心检索入口：多路召回 -> RRF 融合 -> Rerank 重排序 -> 上下文增强。
        """
        print(f"\nDEBUG: [开始检索] Query: {query} | Scope: {doc_id}")
        
        # 1. 查询扩展
        queries = self._expand_query(query)
        
        all_vector_results = []
        all_bm25_results = []
        
        # 对每一个扩展后的子查询进行检索
        for q in queries:
            # --- 向量路 (Vector Path) ---
            vector_query_filter = None
            if doc_id != "all":
                vector_query_filter = qdrant_models.Filter(
                    must=[qdrant_models.FieldCondition(
                        key="doc_id", match=qdrant_models.MatchValue(value=doc_id)
                    )]
                )
            
            # 向量检索召回
            v_results = self.index.as_retriever(
                similarity_top_k=15, 
                vector_store_kwargs={"filter": vector_query_filter}
            ).retrieve(q)
            all_vector_results.extend(v_results)
            
            # --- BM25 路 (BM25 Path) ---
            b_results = self._bm25_search(q, doc_id=doc_id, top_k=15)
            all_bm25_results.extend(b_results)
            
        # 2. 混合融合 (RRF)
        # RRF 会自动处理多轮子查询带来的重复节点
        merged_nodes = self._rrf_merge(all_vector_results, all_bm25_results)
        print(f"DEBUG: [多路召回完成] 融合后节点数: {len(merged_nodes)}")
        
        # 3. 重排序 (Rerank)
        # 哪怕召回了一堆，我也只取最相关的 Top-N，宁缺毋滥
        if merged_nodes:
            reranked_nodes = self.reranker.postprocess_nodes(merged_nodes, query_bundle=QueryBundle(query_str=query))
        else:
            return ""
            
        # 4. 上下文扩展与最终文本组装
        # 我会尝试把检索到的片段与其所在的文档上下文串联起来，使 LLM 回答更准确
        final_context_blocks = []
        for i, n_s in enumerate(reranked_nodes):
            # 获取节点所在的真实 doc_id（防止搜索 all 时来源混乱）
            actual_doc_id = n_s.node.metadata.get("doc_id", "unknown")
            source_file = n_s.node.metadata.get("source_file", "未知文件")
            
            expanded_content = self._expand_context(n_s.node, actual_doc_id)
            
            block = f"--- [来源 {i+1}: {source_file}] ---\n{expanded_content}"
            final_context_blocks.append(block)
            
        return "\n\n".join(final_context_blocks)


# ==========================================
# 单例引擎获取 (确保全局只加载一次显存)
# ==========================================
_rag_engine_instance = None

def get_rag_engine():
    global _rag_engine_instance
    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()
    return _rag_engine_instance


# ==========================================
# LangChain Tools 定义
# ==========================================

@tool
def rag_search_tool(query: str, doc_id: str = "all"):
    """
    根据用户问题，从本地知识库中检索高度相关的文档片段。
    
    参数：
    - query: 用户的提问或需要搜索的关键词。
    - doc_id: 文档 ID。如果设为 "all"，则搜索全库；如果指定具体 ID，则仅搜索该文档。
    
    返回：
    - 检索到的相关文本片段，按相关性排序，带有来源标识。
    """
    engine = get_rag_engine()
    result = engine.search(query, doc_id)
    
    if not result or result.strip() == "":
        return "【检索结果为空】在当前知识库中未找到相关内容，建议尝试更换关键词或搜索全库。"
    
    return result


@tool
def list_docs_tool():
    """
    获取当前知识库的完整文档概览列表。
    
    使用场景：
    - 确认当前加载了哪些文档。
    - 用户询问"你有关于什么的资料"或"有哪些 PDF"时。
    
    返回：
    - JSON 格式的文档列表。
    """
    if not MANIFEST_PATH.exists():
        return json.dumps({
            "status": "empty",
            "message": "文档库暂时没有内容。请先运行 ingestion_pipeline.py 导入文档。",
            "documents": []
        }, ensure_ascii=False)
    
    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        # 抽取核心元数据，方便 Agent 决策
        simplified = []
        for doc in manifest:
            simplified.append({
                "doc_id": doc.get("doc_id", "unknown"),
                "title": doc.get("title", "Untitled"),
                "summary": doc.get("summary", ""),
                "key_points": doc.get("key_points", [])
            })
        
        return json.dumps({
            "status": "success",
            "count": len(simplified),
            "documents": simplified
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"ERROR: 读取文档库清单失败 - {e}"

# ==========================================
# 调试入口
# ==========================================
if __name__ == "__main__":
    # 快速冒烟测试
    engine = get_rag_engine()
    test_query = "什么是 DPO 训练？"
    res = engine.search(test_query)
    print("\n--- 最终检索预览 ---")
    print(res[:500] + "...")