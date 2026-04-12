"""
离线入库流水线 - 最终修复完整版 (Final Refactored Ingestion Pipeline)
职责：
1. 监控/扫描指定文档库目录，识别新增或修改的 PDF 文件。
2. 利用 LLM 提取文档画像（标题、作者、摘要、关键词）。
3. 进行语义分块（Semantic Splitting）并显式嵌入。
4. 真·全局 BM25：将全库所有节点聚合，统一计算全局 IDF。
5. 向量库清理：入库前自动物理删除旧 doc_id 数据，确保数据唯一性。
"""

import os
import json
import hashlib
import pickle
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import jieba
from rank_bm25 import BM25Okapi

# LlamaIndex 组件
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Qdrant 客户端
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_openai import ChatOpenAI

# ==========================================
# 日志配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("ingestion_pipeline.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# 配置管理
# ==========================================
@dataclass
class PipelineConfig:
    """Pipeline 核心配置类，支持从环境变量读取"""
    
    # 基础路径
    documents_dir: Path = Path("./documents")
    qdrant_path: str = "./qdrant_storage"
    bm25_index_path: Path = Path("./bm25_global_index.pkl")
    metadata_path: Path = Path("./ingestion_metadata.json")
    manifest_path: Path = Path("./manifest.json")
    
    # LLM 分析器配置
    llm_model: str = "qwen-plus-2025-12-01"
    llm_temperature: float = 0.0
    
    # 嵌入模型配置 (默认使用 BGE-M3)
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32
    
    # 语义分块配置
    semantic_split_buffer: int = 1
    semantic_split_threshold: float = 0.95
    
    # 功能开关
    bm25_enabled: bool = True

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """从环境变量快速初始化"""
        return cls(
            documents_dir=Path(os.getenv("DOCS_DIR", "./documents")),
            qdrant_path=os.getenv("QDRANT_PATH", "./qdrant_storage"),
            bm25_index_path=Path(os.getenv("BM25_INDEX_PATH", "./bm25_global_index.pkl")),
            metadata_path=Path(os.getenv("METADATA_PATH", "./ingestion_metadata.json")),
            manifest_path=Path(os.getenv("MANIFEST_PATH", "./manifest.json")),
            llm_model=os.getenv("LLM_MODEL", "qwen-plus-2025-12-01"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
        )


@dataclass
class ProcessingStats:
    """处理统计容器"""
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    total_nodes_ingested: int = 0
    total_time_seconds: float = 0.0
    skipped_documents: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "total_documents": self.total_documents,
            "successful_documents": self.successful_documents,
            "failed_documents": self.failed_documents,
            "skipped_documents": self.skipped_documents,
            "total_nodes_ingested": self.total_nodes_ingested,
            "total_time_seconds": f"{self.total_time_seconds:.2f}s",
            "success_rate": f"{(self.successful_documents / self.total_documents * 100) if self.total_documents > 0 else 0:.2f}%"
        }


# ==========================================
# 文档分析器 (LLM Profiler)
# ==========================================
class DocumentProfiler:
    """利用 LLM 对文档进行内容理解和结构化分析"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("[WARNING] OPENAI_API_KEY 未设置，分析器可能无法工作")
        
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=config.llm_model,
            temperature=config.llm_temperature
        )
    
    def analyze(self, text_content: str) -> Dict:
        """分析文本，返回 JSON 格式的文档画像"""
        # 限制分析的文本长度以节省 Token
        sample_text = text_content[:8000] if len(text_content) > 8000 else text_content
        
        system_prompt = """
        你是一个文档分析专家。请分析给定的文档内容，提取以下信息并以 JSON 格式返回：
        要求：
        1. title: 文档标题（若无则拟定一个）
        2. author: 作者名（若无则返回 Unknown）
        3. summary: 核心内容摘要（100字以内）
        4. key_points: 关键词列表（最多5个）
        
        只返回 JSON 代码块，不要有解释性文字。
        """
        
        try:
            response = self.llm.invoke([
                ("system", system_prompt),
                ("human", f"文档内容片段：\n{sample_text}")
            ])
            
            content = response.content.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            logger.error(f"[ERROR] LLM 画像分析异常: {e}")
            return {
                "title": "Untitled Document",
                "author": "Unknown",
                "summary": "解析失败，建议手动核查",
                "key_points": []
            }


# ==========================================
# 全局 BM25 索引管理
# ==========================================
class GlobalBM25Index:
    """
    真·全局 BM25 索引管理类
    相比散文件，本类维护一个全局的节点池（Node Pool），
    确保在检索时使用的是基于全量数据的 IDF（逆文档频率）统计。
    """
    
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self.data = self._load()
        self.bm25_model: Optional[BM25Okapi] = None
        # 初始化时尝试构建一次模型
        self._refresh_model()
    
    def _load(self) -> Dict:
        """从磁盘加载持久化的全局节点数据"""
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    logger.info(f"[INFO] 已加载全局 BM25 索引，包含节点: {len(data.get('all_nodes', []))}")
                    return data
            except Exception as e:
                logger.warning(f"[WARNING] BM25 索引加载失败: {e}，将创建新索引")
        
        return {
            "version": 2,
            "all_nodes": [],
            "last_updated": datetime.now().isoformat()
        }
    
    def _refresh_model(self):
        """[核心] 重建全局统计模型"""
        nodes = self.data.get("all_nodes", [])
        if not nodes:
            self.bm25_model = None
            return

        logger.info(f"[INFO] 正在为 {len(nodes)} 个节点构建全局 BM25 模型...")
        corpus = []
        for node in nodes:
            text = node.get_content()
            # 针对中文分词
            if any(ord(c) > 127 for c in text):
                tokens = jieba.lcut(text)
            else:
                tokens = text.lower().split()
            corpus.append(tokens)
        
        self.bm25_model = BM25Okapi(corpus)
    
    def sync_documents(self, doc_id: str, new_nodes: List):
        """同步文档：先剔除全局池中属于该 doc_id 的旧节点，再合入新节点"""
        self.data["all_nodes"] = [
            node for node in self.data["all_nodes"] 
            if node.metadata.get("doc_id") != doc_id
        ]
        self.data["all_nodes"].extend(new_nodes)
        self._refresh_model()
    
    def save(self):
        """保存全局数据到磁盘"""
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.index_path, "wb") as f:
            pickle.dump(self.data, f)
        logger.info("[INFO] 全局 BM25 数据已持久化")

    def clear_all(self):
        """重置所有数据"""
        self.data["all_nodes"] = []
        self.bm25_model = None


# ==========================================
# 主入库流水线 (Ingestion Pipeline)
# ==========================================
class IngestionPipeline:
    """自动化入库流水线类"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig.from_env()
        logger.info(f"[INIT] 启动入库流水线，工作目录: {self.config.documents_dir}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 设置 Embedding 模型
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.config.embedding_model,
            device=self.device,
            embed_batch_size=self.config.embedding_batch_size
        )
        
        # 2. 初始化各组件
        self.profiler = DocumentProfiler(self.config)
        self.qdrant_client = QdrantClient(path=self.config.qdrant_path)
        self._ensure_collection_exists()
        
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name="expert_collection"
        )
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # 3. 初始化全局索引与元数据
        self.bm25_index = GlobalBM25Index(self.config.bm25_index_path)
        self.metadata = self._load_json(self.config.metadata_path, {"processed_files": {}, "version": 1})
        self.manifest = self._load_json(self.config.manifest_path, [])
    
    def _load_json(self, path: Path, default: Any) -> Any:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return default

    def _save_json(self, path: Path, data: Any):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _ensure_collection_exists(self):
        """确保向量集合已按正确维度初始化"""
        collections = self.qdrant_client.get_collections().collections
        if "expert_collection" not in [c.name for c in collections]:
            logger.info("[INFO] 正在创建 Qdrant 集合: expert_collection")
            self.qdrant_client.create_collection(
                collection_name="expert_collection",
                vectors_config=VectorParams(
                    size=self.config.embedding_dim, 
                    distance=Distance.COSINE
                )
            )

    def _calculate_md5(self, file_path: Path) -> str:
        """计算文件 MD5 用于检测文件变化"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def process_document(self, file_path: Path) -> Optional[Dict]:
        """
        处理单个文档的完整流程：提取 -> 分块 -> 嵌入 -> 向量清理 -> 向量入库 -> BM25 同步
        """
        start_time = time.time()
        current_md5 = self._calculate_md5(file_path)
        # 生成基于文件名和 MD5 的唯一 ID
        doc_id = "".join(c if c.isalnum() else "_" for c in file_path.stem) + "_" + current_md5[:8]
        
        try:
            # 1. 文本读取
            reader = PyMuPDFReader()
            documents = reader.load_data(file_path=str(file_path))
            full_text = "\n".join([doc.text for doc in documents])
            
            # 2. 提取画像
            logger.info(f"[STAGE 1] 正在提取画像: {file_path.name}")
            profile = self.profiler.analyze(full_text)
            
            # 3. 语义分块
            logger.info(f"[STAGE 2] 正在执行语义分块...")
            splitter = SemanticSplitterNodeParser(
                buffer_size=self.config.semantic_split_buffer,
                breakpoint_percentile_threshold=self.config.semantic_split_threshold,
                embed_model=Settings.embed_model
            )
            
            all_nodes = []
            for doc in documents:
                nodes = splitter.get_nodes_from_documents([doc])
                for node in nodes:
                    node.metadata["doc_id"] = doc_id
                    node.metadata["source_file"] = file_path.name
                all_nodes.extend(nodes)

            # 4. 显式嵌入管理 (Embedding)
            logger.info(f"[STAGE 3] 正在生成 {len(all_nodes)} 个节点的向量嵌入...")
            texts = [node.text for node in all_nodes]
            embeddings = Settings.embed_model.get_text_embedding_batch(texts)
            for node, emb in zip(all_nodes, embeddings):
                node.embedding = emb

            # 5. 向量库清理 (防止增量重复)
            logger.info(f"[STAGE 4] 清理旧向量数据 (doc_id: {doc_id})...")
            # 这里其实有一个涉及到文件版本和文件本身是否变化的问题
            # 如果文件名和文件内容都变化了，代码要怎么判断是不是同一个文件？
            # 这是一个忒修斯之船问题。所以完美的做法取决于我们根据业务状况做出的选择
            self.qdrant_client.delete(
                collection_name="expert_collection",
                points_selector=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                ),
            )

            # 6. 向量入库
            logger.info(f"[STAGE 5] 正在执行向量入库...")
            VectorStoreIndex(
                nodes=all_nodes,
                storage_context=self.storage_context,
                show_progress=False
            )

            # 7. 全局 BM25 同步
            if self.config.bm25_enabled:
                logger.info(f"[STAGE 6] 正在同步全局 BM25 索引...")
                self.bm25_index.sync_documents(doc_id, all_nodes)
                self.bm25_index.save()

            # 8. 更新处理状态记录
            file_key = str(file_path.name)
            self.metadata["processed_files"][file_key] = {
                "md5": current_md5,
                "doc_id": doc_id,
                "processed_at": datetime.now().isoformat()
            }
            
            elapsed = time.time() - start_time
            logger.info(f"[SUCCESS] {file_path.name} 处理完成，耗时 {elapsed:.2f}s")
            
            return {
                "doc_id": doc_id,
                "title": profile.get("title", "Untitled"),
                "author": profile.get("author", "Unknown"),
                "summary": profile.get("summary", ""),
                "key_points": profile.get("key_points", []),
                "num_nodes": len(all_nodes),
                "status": "success",
                "processed_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"[ERROR] 文档 {file_path.name} 处理过程中崩溃: {e}", exc_info=True)
            return {"status": "failed", "file": file_path.name}

    def scan_and_process(self) -> ProcessingStats:
        """主入口：扫描目录执行增量入库"""
        if not self.config.documents_dir.exists():
            logger.error(f"[ERROR] 目录不存在: {self.config.documents_dir}")
            return ProcessingStats()

        pdf_files = list(self.config.documents_dir.rglob("*.pdf"))
        stats = ProcessingStats(total_documents=len(pdf_files))
        start_time = time.time()
        
        new_entries = []
        for pdf_file in pdf_files:
            file_key = str(pdf_file.name)
            current_md5 = self._calculate_md5(pdf_file)
            
            # 增量检查
            if file_key in self.metadata["processed_files"]:
                if self.metadata["processed_files"][file_key]["md5"] == current_md5:
                    logger.info(f"[SKIP] 文件无变化，跳过: {file_key}")
                    stats.skipped_documents += 1
                    continue

            logger.info(f"\n[START] 开始处理新文件: {file_key}")
            entry = self.process_document(pdf_file)
            
            if entry and entry.get("status") == "success":
                new_entries.append(entry)
                stats.successful_documents += 1
                stats.total_nodes_ingested += entry["num_nodes"]
            else:
                stats.failed_documents += 1
        
        # 处理结果持久化
        if new_entries or stats.skipped_documents < stats.total_documents:
            processed_ids = {e["doc_id"] for e in new_entries}
            # 更新 Manifest：移除旧条目，加入新条目
            self.manifest = [m for m in self.manifest if m["doc_id"] not in processed_ids]
            self.manifest.extend(new_entries)
            
            self._save_json(self.config.manifest_path, self.manifest)
            self._save_json(self.config.metadata_path, self.metadata)
            
        stats.total_time_seconds = time.time() - start_time
        return stats

    def reset_all(self):
        """危险操作：清空所有入库数据重新开始"""
        logger.warning("[RESET] 正在清理所有存储数据...")
        try:
            self.qdrant_client.delete_collection("expert_collection")
            self._ensure_collection_exists()
            self.bm25_index.clear_all()
            self.bm25_index.save()
            self.metadata = {"processed_files": {}, "version": 1}
            self.manifest = []
            self._save_json(self.config.manifest_path, [])
            self._save_json(self.config.metadata_path, self.metadata)
            logger.info("[RESET] 清理完成")
        except Exception as e:
            logger.error(f"[ERROR] 重置失败: {e}")


# ==========================================
# 运行脚本
# ==========================================
def main():
    """主程序入口"""
    print("="*60)
    print("Homie, RAG 自动化入库流水线准备就绪")
    print("="*60)
    
    # 确保文档目录存在
    Path("./documents").mkdir(exist_ok=True)
    
    pipeline = IngestionPipeline()
    
    # 执行扫描
    stats = pipeline.scan_and_process()
    
    # 输出简报
    print(f"\n处理简报:")
    print("-" * 20)
    res = stats.to_dict()
    for k, v in res.items():
        print(f"{k}: {v}")
    print("-" * 20)
    
    # 打印当前文档库摘要
    manifest = pipeline.manifest
    if manifest:
        print("\n当前文档库目录:")
        for i, m in enumerate(manifest, 1):
            print(f"{i}. [{m['title']}] - 作者: {m['author']} ({m['num_nodes']} 节点)")

if __name__ == "__main__":
    main()