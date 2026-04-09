项目迭代指令：RAG 系统 V8 版本 - 离线化与元数据驱动架构
当前任务
将现有的“实时上传-解析”RAG 系统重构为“计算与存储分离”的工业级架构。引入离线预处理流水线和动态文档目录（Data Catalog）机制。
1. 新增功能模块：离线入库流水线 (ingestion_pipeline.py)
职责：监控/扫描指定文档库目录（如 ./documents），对新文档进行异步解析并持久化。
逻辑流程：
扫描与过滤：扫描目录下的 PDF 文件，支持增量处理（通过 MD5 校验或文件修改时间避免重复解析）。
生成文档画像 (Document Profile)：利用 LLM (qwen-plus) 提取标题、作者、核心摘要、若干个关键词。
持久化存储：
向量入库：将文档切片存入 Qdrant 向量数据库，必须配置 path 参数以支持本地磁盘持久化。
关键词索引：构建并持久化保存 BM25 索引（如使用 .pkl 或 .json 格式存储），确保重启后无需重新扫描。
元数据更新：同步更新项目根目录下的 manifest.json。
2. 在文档库目录下新增配置文件：文档目录索引 (manifest.json)
作用：作为 Agent 的“全知视角”，用于回复关于文档库本身的问题，并辅助智能路由。
格式要求：JSON 数组，存储每个文档的 doc_id, title, summary, key_points, file_path。
内容约束：
标题 (title)：若原文档无标题，由 LLM 拟定，不超过 10 个词。
摘要 (summary)：不超过 100 个词。
关键词 (key_points)：不超过 5 个关键词。
路径 (file_path)：相对于 ./documents 的相对路径。
3. 重构 RAG 工具模块 (rag_tools.py)
删除：去除从指定文件路径实时读取、解析并存入向量库的逻辑。
修改：search_pdf_tool：
参数：包含 doc_id (默认值为 all)。
物理过滤：若 doc_id 为特定值，底层 Qdrant 查询必须使用 models.Filter 进行 Pre-filtering，确保检索范围严格锁定在该文档内。
混合检索：加载持久化的 BM25 索引，同样需支持按 doc_id 过滤。
新增：list_docs_tool：读取 manifest.json 并返回当前知识库的概览字段。
4. 重构主 Agent 逻辑 (main.py)
交互变更：彻底删除“请输入 PDF 路径”的交互步骤，程序启动即自动加载现有库。
系统提示词优化：程序启动时即调用 list_docs_tool，将清单以格式化的 Markdown 注入 System Message（作为“文档库概览”）。
智能路由逻辑（核心）：
冷启动引导：若用户问题模糊（如“你能做什么”），Agent 需基于概览进行回复并提供提问建议。
自动识别 doc_id：当用户提出具体问题时，Agent 必须先根据系统提示词中的概览进行判断：
若明显指向特定文档，自动锁定 doc_id 并调用 search_pdf_tool。
若涉及多文档或无法确定，将 doc_id 设为 all 进行全局检索。
调试透明度：在调试模式下，Agent 必须在终端显式输出其判断出的 doc_id，例如[DEBUG] Detected Intent: Specific Question | Target doc_id: sam2_paper 或者 [DEBUG] Detected Intent: No Specific Question | Target doc_id: all
输出原则（禁止幻觉）：
严格闭环回复：若检索工具未返回相关内容，Agent 必须如实回复“抱歉，在当前知识库中没有检索到相关内容”，严禁调用 LLM 自身通用知识进行脑补。
5. 技术约束
向量库：Qdrant (Local Persistence Mode)。
LLM：统一使用 qwen-plus-2025-12-01。
解耦要求：ingestion_pipeline.py 必须能够作为独立脚本运行，不依赖对话主程序。
