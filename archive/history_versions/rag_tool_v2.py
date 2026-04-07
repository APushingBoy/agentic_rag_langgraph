import os
import torch
from typing import List

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

# ==========================================
# 1. 核心 RAG 引擎类 (LlamaIndex 增强版)
# ==========================================
class RAGEngine:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"DEBUG: [RAG 引擎启动] 正在加载模型到 {self.device} (4090 模式)...")

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
        
        print(f"DEBUG: [RAG] LlamaIndex 索引构建成功")

    def search(self, query: str) -> str:
        # --- [旧代码：手动计算 query_vector, 调用 query_points, 手动循环 Rerank] ---
        # 下面是 LlamaIndex 的全自动化链路：
        
        # 1. 创建检索器
        retriever = self.index.as_retriever(similarity_top_k=15)
        
        # 2. 执行检索（召回阶段）
        nodes = retriever.retrieve(query)
        
        # 3. 执行精排（Rerank 阶段）
        # 使用我们 init 时定义的 self.reranker
        ranked_nodes = self.reranker.postprocess_nodes(nodes, query_bundle=QueryBundle(query_str=query))
        
        # --- [保留原有的 Debug 输出逻辑] ---
        print(f"\n{'#'*30} RAG 检索诊断 (LlamaIndex 版) {'#'*30}")
        print(f" 原始查询: {query}")
        print(f" 召回阶段 Top 3:")
        for i, node in enumerate(nodes[:3]):
            print(f"  [{i}] {node.text[:60]}...")
            
        print(f" 精排阶段 Top 3:")
        for i, node in enumerate(ranked_nodes[:3]):
            print(f"  [{i}] Score: {node.score:.4f} | {node.text[:60]}...")
        print(f"{'#'*75}\n")

        # 返回合并后的文本
        return "\n___\n".join([n.text for n in ranked_nodes])

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
    """
    if _global_rag_engine is None:
        return "错误：RAG 引擎未初始化。"
    return _global_rag_engine.search(query)