import os
import torch
from typing import List
from markitdown import MarkItDown
from chonkie import SemanticChunker, AutoEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_core.tools import tool

# ==========================================
# 1. 核心 RAG 引擎类 (负责模型管理和数据处理)
# ==========================================
class RAGEngine:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DEBUG: [RAG 引擎启动] 正在加载模型到 {self.device} (使用 FP16)...")

        # 初始化 BGE-M3 (Embedding)
        self.embed_model = SentenceTransformer(
            "BAAI/bge-m3", 
            device=self.device,
            model_kwargs={"torch_dtype": torch.float16}
        )
        
        # 初始化 BGE-Reranker-v2-m3
        self.rerank_model = CrossEncoder(
            "BAAI/bge-reranker-v2-m3", 
            device=self.device
        )

        # 初始化 Qdrant (内存模式)
        self.client = QdrantClient(":memory:")
        self._process_document()

    def _extract_text(self) -> str:
        md = MarkItDown()
        return md.convert(self.file_path).text_content

    def _create_chunks(self, raw_text: str):
        # 包装已经加载的模型，避免重复加载
        wrapped_embeddings = AutoEmbeddings.get_embeddings(self.embed_model)
        chunker = SemanticChunker(
            embedding_model=wrapped_embeddings,
            threshold=0.6,
            chunk_size=512,
            min_sentences=1
        )
        return chunker.chunk(raw_text)

    def _process_document(self):
        print(f"DEBUG: [RAG] 正在解析文档: {os.path.basename(self.file_path)}")
        raw_text = self._extract_text()
        chunks = self._create_chunks(raw_text)
        
        docs = [chunk.text for chunk in chunks]
        embeddings = self.embed_model.encode(
            docs, 
            batch_size=32, 
            show_progress_bar=True,
            normalize_embeddings=True  # 确保余弦相似度计算更稳定
        )

        # --- 优化点 2: 注入 Metadata (为以后溯源做准备) ---
        payloads = [
            {
                "content": doc,
                "source": os.path.basename(self.file_path),
                "chunk_id": i,
                "total_chunks": len(chunks)
            } for i, doc in enumerate(docs)
        ]

        if not self.client.collection_exists("expert_collection"):
            self.client.create_collection(
                collection_name="expert_collection",
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

        self.client.upload_collection(
            collection_name="expert_collection",
            vectors=embeddings,
            payload=payloads,
            ids=list(range(len(chunks)))
        )
        print(f"DEBUG: [RAG] 索引构建成功，片段数: {len(chunks)}")

    def search(self, query: str) -> str:
        # 1. 向量化检索 (海选)
        # --- 优化点 3: Query 也需要 normalize ---
        query_vector = self.embed_model.encode(query, normalize_embeddings=True).tolist()
        try:
            # 使用 query_points 检索
            response = self.client.query_points(
                collection_name="expert_collection",
                query=query_vector,
                limit=15
            )
            initial_results = response.points 
        except Exception as e:
            return f"工具内部错误：无法调用检索接口。详情: {str(e)}"

        if not initial_results:
            return "未在文档中找到相关匹配内容。"

        # --- 【关键修复点】：兼容元组和对象两种返回格式 ---
        pairs = []
        valid_results = []
        
        for res in initial_results:
            # 如果是对象，用 res.payload；如果是元组，通常 payload 在索引 2 或 3
            # 我们用 hasattr 探测，最稳妥
            if hasattr(res, 'payload') and res.payload:
                content = res.payload.get('content', "")
            elif isinstance(res, (tuple, list)):
                # 在元组模式下，Qdrant 结果通常是 (id, score, payload, ...)
                # 我们直接找寻包含 'content' 键的那个元素
                content = next((item['content'] for item in res if isinstance(item, dict) and 'content' in item), "")
            else:
                content = ""

            if content:
                pairs.append([query, content])
                valid_results.append(content)

        if not pairs:
            return "检索到的原始数据格式异常，无法解析内容。"

        # 2. Rerank (精排)
        scores = self.rerank_model.predict(pairs, batch_size=16)

        # 3. 排序与过滤
        reranked = []
        for i in range(len(valid_results)):
            reranked.append({
                "content": valid_results[i],
                "score": float(scores[i])
            })
        
        reranked.sort(key=lambda x: x["score"], reverse=True)
        
        # --- 优化点 4: 动态 Top-K 策略 (弃用硬编码阈值) ---
        # 即使分数低，我们也给 Top 3，让 Agent 自己判断是否有用
        # 但如果分数断层（比如第1名 5分，第2名 -5分），后续会通过逻辑优化
        final_results = reranked[:5]
        
        # --- 优化点 5: 深度 Debug 输出 (核心生产力) ---
        print(f"\n{'#'*30} RAG 检索诊断 {'#'*30}")
        print(f" 原始查询: {query}")
        print(f" 召回阶段 (Vector Search Top 3):")
        for i, res in enumerate(valid_results[:3]):
            print(f"  [{i}] {res[:60]}...")
        
        print(f" 精排阶段 (Reranker Top 3):")
        for i, res in enumerate(final_results[:3]):
            print(f"  [{i}] Score: {res['score']:.4f} | {res['content'][:60]}...")
        print(f"{'#'*75}\n")

        return "\n___\n".join([r["content"] for r in final_results])

# ==========================================
# 2. 定义原生 LangGraph 工具
# ==========================================

# 我们在外部初始化这个引擎，这样模型只会加载一次
# 注意：在实际运行中，SELECTED_PDF_PATH 由你的 main 脚本提供
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
    """
    if _global_rag_engine is None:
        return "错误：RAG 引擎未初始化，请检查 PDF 路径。"
    return _global_rag_engine.search(query)