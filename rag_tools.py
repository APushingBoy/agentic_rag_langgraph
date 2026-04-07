import os
import torch
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import jieba

# --- [新引入 LlamaIndex 相关组件] ---
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.schema import QueryBundle
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.readers.file import PyMuPDFReader

# --- [保留原有的工具库] ---
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# ==========================================
# 1. 核心 RAG 引擎类 (LlamaIndex 增强版)
# ==========================================
class RAGEngine:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DEBUG: [RAG 引擎启动] 正在加载模型到 {self.device} (4090 模式)...")
        
        # claude建议定义一个query_llm
        self.query_llm = ChatOpenAI(model="qwen-plus", temperature=0)
        # --- [旧代码：手动加载 SentenceTransformer] ---
        # self.embed_model = SentenceTransformer(...) 
        
        # --- [新代码：配置 LlamaIndex 全局模型设置] ---
        # 使用 HuggingFaceEmbedding 直接对接 BGE-M3，充分利用 4090
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-m3",
            device=self.device,
            embed_batch_size=128 # 4090 显存大，直接把 Batch Size 开大
        )
        # 注意：LlamaIndex 默认会处理 LLM 节点，如果我们只做检索，可以暂不配置 Settings.llm

        # --- [旧代码：手动初始化 CrossEncoder] ---
        # self.rerank_model = CrossEncoder(...)
        
        # --- [新代码：使用 LlamaIndex 内置的 Reranker 插件] ---
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-v2-m3",
            top_n=5, # 相当于你原来的 final_results[:5]
            device=self.device
        )

        # 初始化 Qdrant
        self.client = QdrantClient(":memory:")
        
        # --- [新逻辑：初始化数据存储上下文] ---
        self.vector_store = QdrantVectorStore(
            client=self.client, 
            collection_name="expert_collection"
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # 初始化 BM25 相关
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_nodes = []
        
        self._process_document()

    def _extract_text(self):
        """
        [旧代码：手动使用 MarkItDown]
        md = MarkItDown()
        return md.convert(self.file_path).text_content
        """
        # [新代码：使用 LlamaIndex 适配器，它能更好地保留元数据]
        reader = PyMuPDFReader()
        return reader.load_data(file_path=self.file_path)

    def _process_document(self):
        print(f"DEBUG: [RAG] 正在解析文档: {os.path.basename(self.file_path)}")
        
        # 1. 提取文档
        documents = self._extract_text() 
        
        # 2. 语义切分 (替代原来的 Chonkie 手动切分)
        # LlamaIndex 的 SemanticSplitter 会自动利用 Settings.embed_model 进行语义聚类切片
        splitter = SemanticSplitterNodeParser(
            buffer_size=1, 
            breakpoint_percentile_threshold=95,
            embed_model=Settings.embed_model
        )
        
        # --- [旧代码：手动 chunks -> embeddings -> upload_collection] ---
        # 这部分现在被一行代码取代：索引构建
        # LlamaIndex 会自动完成：切片、计算向量、批量上传 Qdrant
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            transformations=[splitter],
            show_progress=True
        )
        
        # 3. 构建 BM25 索引
        print(f"DEBUG: [RAG] 正在构建 BM25 索引...")
        # 获取所有节点
        self.bm25_nodes = []
        for doc in documents:
            nodes = splitter.get_nodes_from_documents([doc])
            self.bm25_nodes.extend(nodes)
        
        # 准备 BM25 语料库
        self.bm25_corpus = []
        for node in self.bm25_nodes:
            # 对中文文本进行分词
            if any(ord(c) > 127 for c in node.text):
                tokens = jieba.lcut(node.text)
            else:
                tokens = node.text.lower().split()
            self.bm25_corpus.append(tokens)
        
        # 构建 BM25 索引
        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)
            print(f"DEBUG: [RAG] BM25 索引构建成功，包含 {len(self.bm25_nodes)} 个节点")
        
        print(f"DEBUG: [RAG] LlamaIndex 索引构建成功")

    def _expand_query(self, query: str) -> List[str]:
        """
        将原始查询改写成多个子查询
        """
        try:
            # 创建 LLM 实例 ❌ 每次都创建实例太低效了
            # llm = ChatOpenAI(model="qwen-plus", temperature=0)
            
            # 构建系统提示
            system_prompt = """
            你是一个查询改写器。
            将用户的问题改写成 3 个不同角度的子问题，用于从学术论文中检索不同层面的信息。
            每个子问题应该关注原问题的不同方面。
            只输出 JSON 数组：["子问题1", "子问题2", "子问题3"]
            不要输出任何其他内容。
            """
            
            # 发起调用
            response = self.query_llm.invoke(system_prompt + f"\n\n原始问题：{query}")
            
            # 解析 JSON 结果
            import json
            # sub_queries = json.loads(response.content)
            # 防止千问的回复中包含json相关的markdown代码块的内容
            content = response.content.strip().replace("```json", "").replace("```", "").strip()
            sub_queries = json.loads(content)
            
            # 确保返回的是列表且长度为 3
            if isinstance(sub_queries, list) and len(sub_queries) == 3:
                return sub_queries
            else:
                return [query]
        except Exception as e:
            # 失败时退化为返回原始查询
            print(f"DEBUG: [RAG] 查询改写失败: {str(e)}")
            return [query]

    def _bm25_search(self, query: str, top_k: int = 15) -> List[Tuple[float, any]]:
        """使用 BM25 进行关键词搜索"""
        if not self.bm25_index:
            return []
        
        # 对查询进行分词
        if any(ord(c) > 127 for c in query):
            query_tokens = jieba.lcut(query)
        else:
            query_tokens = query.lower().split()
        
        # BM25 搜索
        scores = self.bm25_index.get_scores(query_tokens)
        
        # 排序并返回 Top K
        results = []
        for i, score in enumerate(scores):
            if score > 0:
                results.append((score, self.bm25_nodes[i]))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
    
    def _rrf_merge(self, vector_results: List[any], bm25_results: List[Tuple[float, any]], k: int = 60) -> Tuple[List[any], Dict[str, float]]:
        """使用 RRF 算法合并向量搜索和 BM25 搜索结果"""
        # 构建文档到排名的映射
        doc_rank = {}
        
        # 处理向量搜索结果
        for rank, node in enumerate(vector_results):
            if node.node_id not in doc_rank:
                doc_rank[node.node_id] = 0
            doc_rank[node.node_id] += 1 / (rank + k)
        
        # 处理 BM25 搜索结果
        for rank, (score, node) in enumerate(bm25_results):
            if node.node_id not in doc_rank:
                doc_rank[node.node_id] = 0
            doc_rank[node.node_id] += 1 / (rank + k)
        
        # 按 RRF 得分排序
        sorted_docs = sorted(doc_rank.items(), key=lambda x: x[1], reverse=True)
        
        # 构建结果列表
        node_map = {node.node_id: node for node in vector_results}
        for _, node in bm25_results:
            if node.node_id not in node_map:
                node_map[node.node_id] = node
        
        results = []
        rrf_scores = {}
        for node_id, score in sorted_docs:
            if node_id in node_map:
                node = node_map[node_id]
                # 确保所有节点都是 NodeWithScore 类型
                from llama_index.core.schema import NodeWithScore
                if not isinstance(node, NodeWithScore):
                    # 创建一个 NodeWithScore 对象
                    node = NodeWithScore(node=node, score=0.0)
                results.append(node)
                rrf_scores[node_id] = score
        
        return results, rrf_scores
    
    def _expand_context(self, node: any, context_size: int = 500) -> str:
        """扩展节点上下文，包含前后各 500 tokens"""
        if not self.bm25_nodes:
            return node.text
        
        # 找到当前节点在列表中的位置
        node_index = -1
        for i, n in enumerate(self.bm25_nodes):
            if n.node_id == node.node_id:
                node_index = i
                break
        
        if node_index == -1:
            return node.text
        
        # 收集前后节点
        start_idx = max(0, node_index - 2)
        end_idx = min(len(self.bm25_nodes), node_index + 3)
        
        # 合并上下文
        context_parts = []
        for i in range(start_idx, end_idx):
            if i != node_index:
                context_parts.append(f"[上下文] {self.bm25_nodes[i].text}")
            else:
                context_parts.append(f"[核心内容] {self.bm25_nodes[i].text}")
        
        return "\n___\n".join(context_parts)
    
    def search(self, query: str) -> str:
        # --- [旧代码：手动计算 query_vector, 调用 query_points, 手动循环 Rerank] ---
        # 下面是 LlamaIndex 的全自动化链路：
        
        # 1. 扩展查询
        sub_queries = self._expand_query(query)
        # 包含原始查询
        all_queries = [query] + sub_queries
        
        # 2. 创建检索器
        retriever = self.index.as_retriever(similarity_top_k=15)
        
        # 3. 执行多查询检索（召回阶段）
        all_nodes = []
        for q in all_queries:
            nodes = retriever.retrieve(q)
            all_nodes.extend(nodes)
        
        # 4. 去重
        unique_nodes = {}
        for node in all_nodes:
            if node.node_id not in unique_nodes:
                unique_nodes[node.node_id] = node
        vector_nodes = list(unique_nodes.values())
        
        # 5. 执行 BM25 搜索
        bm25_results = self._bm25_search(query, top_k=15)
        
        # 6. 使用 RRF 合并结果
        merged_nodes, rrf_scores = self._rrf_merge(vector_nodes, bm25_results)
        
        # 7. 执行精排（Rerank 阶段）
        # 使用我们 init 时定义的 self.reranker
        ranked_nodes = self.reranker.postprocess_nodes(merged_nodes, query_bundle=QueryBundle(query_str=query))
        
        # 8. 扩展上下文
        expanded_nodes = []
        for node in ranked_nodes:
            expanded_text = self._expand_context(node)
            expanded_nodes.append(expanded_text)
        
        # --- [保留原有的 Debug 输出逻辑] ---
        print(f"\n{'#'*30} RAG 检索诊断 (LlamaIndex 版) {'#'*30}")
        print(f" 原始查询: {query}")
        print(f" 改写子查询: {sub_queries}")
        print(f" 向量检索 Top 3:")
        for i, node in enumerate(vector_nodes[:3]):
            print(f"  [{i}] {node.text[:60]}...")
        
        print(f" BM25 检索 Top 3:")
        for i, (score, node) in enumerate(bm25_results[:3]):
            print(f"  [{i}] Score: {score:.4f} | {node.text[:60]}...")
        
        print(f" RRF 合并后 Top 3:")
        for i, node in enumerate(merged_nodes[:3]):
            rrf_score = rrf_scores.get(node.node_id, 0)
            print(f"  [{i}] RRF Score: {rrf_score:.4f} | {node.text[:60]}...")
        
        print(f" 精排阶段 Top 3:")
        for i, node in enumerate(ranked_nodes[:3]):
            print(f"  [{i}] Score: {node.score:.4f} | {node.text[:60]}...")
        print(f"{'#'*75}\n")

        # 返回合并后的文本
        return "\n___\n".join(expanded_nodes)

# ==========================================
# 2. 定义工具 (保持不变，LangGraph 无感知替换)
# ==========================================
_global_rag_engine = None

def init_rag_engine(file_path: str):
    global _global_rag_engine
    _global_rag_engine = RAGEngine(file_path)

@tool
def search_pdf_tool(query: str):
    """
    Search the PDF document for information regarding specific technical terms, 
    abstracts, or methodologies. Use this whenever the user asks about the 
    content of the uploaded paper.
    
    支持关键词精确匹配。当用户提到具体的章节标题、页码或带有引号的术语时，必须优先使用此工具。
    """
    if _global_rag_engine is None:
        return "错误：RAG 引擎未初始化。"
    return _global_rag_engine.search(query)
