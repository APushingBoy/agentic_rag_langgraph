"""
RAG 系统 V8 版本 - 离线化与元数据驱动架构
- 移除实时 PDF 路径输入
- 引入文档目录索引 (manifest.json)
- 智能路由：自动识别目标文档
- 严格闭环回复：无检索结果时禁止脑补
"""
# ==========================================
# 1. 基础环境与配置
# ==========================================
from dotenv import load_dotenv
import os
import json
import operator
from typing import Annotated, TypedDict, List, Literal, Optional, Dict
from pydantic import BaseModel, Field
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- 引入 Rich 美化库 ---
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()

# --- 导入工具 ---
from rag_tools import rag_search_tool, list_docs_tool, get_rag_engine

load_dotenv()

# 常量配置
LLM_MODEL = "qwen-plus-2025-12-01"
MANIFEST_PATH = Path("./manifest.json")
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"


# ==========================================
# 2. 全局状态：文档库概览
# ==========================================
_docs_catalog = None
_docs_catalog_markdown = ""


def load_document_catalog() -> tuple:
    """
    加载文档库目录，返回 (catalog_dict, catalog_markdown)
    """
    global _docs_catalog, _docs_catalog_markdown
    
    # 调用 list_docs_tool 获取文档列表
    result = list_docs_tool.func()
    catalog_data = json.loads(result)
    
    if catalog_data.get("status") != "ok" or not catalog_data.get("documents"):
        _docs_catalog = {"documents": [], "count": 0}
        _docs_catalog_markdown = "**文档库为空**，请先运行 `ingestion_pipeline.py` 导入文档。"
        return _docs_catalog, _docs_catalog_markdown
    
    _docs_catalog = catalog_data
    
    # 生成 Markdown 格式的概览
    md_parts = ["### 当前知识库概览\n"]
    md_parts.append(f"共收录 **{catalog_data['count']}** 篇文档：\n")
    
    for i, doc in enumerate(catalog_data["documents"], 1):
        title = doc.get("title", "Untitled")
        doc_id = doc.get("doc_id", "unknown")
        summary = doc.get("summary", "")[:80]
        keywords = ", ".join(doc.get("key_points", [])[:3])
        
        md_parts.append(f"{i}. **{title}** (ID: `{doc_id}`)")
        md_parts.append(f"   - 摘要：{summary}...")
        if keywords:
            md_parts.append(f"   - 关键词：{keywords}")
        md_parts.append("")
    
    _docs_catalog_markdown = "\n".join(md_parts)
    return _docs_catalog, _docs_catalog_markdown


def detect_target_doc_id(user_query: str) -> tuple:
    """
    智能识别用户查询可能指向的文档
    返回: (detected_doc_id, confidence, reasoning)
    """
    if not _docs_catalog or not _docs_catalog.get("documents"):
        return "all", 0.0, "文档库为空"
    
    # 构建用于 LLM 判断的上下文
    docs_info = []
    for doc in _docs_catalog["documents"]:
        docs_info.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "key_points": doc.get("key_points", [])
        })
    
    system_prompt = """
    你是一个文档路由专家。分析用户问题，判断它最可能指向哪个文档。
    
    可用文档列表：
    {docs_list}
    
    判断规则：
    1. 如果问题明确提到某个文档的标题、关键词或主题，返回对应的 doc_id
    2. 如果问题涉及多个文档或无法确定具体文档，返回 "all"
    3. 如果问题是关于系统能力、问候等通用问题，返回 "none"（表示不需要检索）
    
    输出格式（严格 JSON）：
    {{
        "target_doc_id": "doc_id 或 all 或 none",
        "confidence": 0.0-1.0,
        "reasoning": "简要说明判断理由"
    }}
    
    只输出 JSON，不要包含其他内容。
    """.format(docs_list=json.dumps(docs_info, ensure_ascii=False, indent=2))
    
    try:
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        response = llm.invoke([
            ("system", system_prompt),
            ("human", f"用户问题：{user_query}")
        ])
        
        content = response.content.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        
        doc_id = result.get("target_doc_id", "all")
        confidence = result.get("confidence", 0.5)
        reasoning = result.get("reasoning", "")
        
        return doc_id, confidence, reasoning
        
    except Exception as e:
        if DEBUG_MODE:
            console.print(f"[yellow][DEBUG] 文档路由分析失败: {e}[/yellow]")
        return "all", 0.0, f"分析失败: {e}"


# ==========================================
# 3. 结构化输出与工具绑定
# ==========================================
class Plan(BaseModel):
    action: Literal["tool", "respond"]
    tool_name: Optional[str] = None
    tool_args: Optional[Dict] = None
    reason: Optional[str] = None


class FinalAnswer(BaseModel):
    answer: str = Field(description="针对问题的正式回答")
    sources: List[str] = Field(description="依据的文档具体原文或章节")
    confidence: float = Field(description="置信度 0-1")


@tool
def submit_final_answer(output: FinalAnswer) -> str:
    """提交最终经过反思的结构化答案。"""
    return "SUCCESS"


# 工具列表
tools = [rag_search_tool, list_docs_tool, submit_final_answer]
llm_plain = ChatOpenAI(model=LLM_MODEL, temperature=0)
llm_with_tools = llm_plain.bind_tools(tools)


# ==========================================
# 4. 状态与节点定义
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    retry_count: int
    plan: dict
    target_doc_id: str  # 新增：目标文档ID


def planner(state: AgentState):
    """
    规划器：判断用户意图，决定是否需要调用工具
    """
    messages = state["messages"]
    current_message = messages[-1]
    user_input = current_message.content
    
    # 提取工具信息
    simplified_tools = []
    for t in tools:
        if t.name != "submit_final_answer":  # 不向 planner 展示内部工具
            args_desc = {k: v.get('description', '无描述') for k, v in t.args.items()}
            simplified_tools.append(f"工具: {t.name}\n参数: {args_desc}")
    
    tools_context = "\n\n".join(simplified_tools)
    
    # 检测纠错倾向关键词
    user_input_lower = user_input.lower()
    correction_keywords = ["你确定吗", "再搜一下", "明明有", "在xx章节", "在章节", "页码", "第.*章"]
    has_correction = any(kw in user_input_lower for kw in correction_keywords)
    
    # 检测检索锚点
    has_anchor = '"' in user_input or '“' in user_input or '”' in user_input
    has_anchor = has_anchor or any(kw in user_input_lower for kw in ["章节", "页码", "section", "chapter", "page"])
    
    # 智能路由：检测目标文档
    target_doc_id, confidence, reasoning = detect_target_doc_id(user_input)
    
    if DEBUG_MODE:
        console.print(f"[cyan][DEBUG] Detected Intent: {'Specific Question' if target_doc_id != 'none' else 'No Specific Question'} | Target doc_id: {target_doc_id} (confidence: {confidence:.2f})[/cyan]")
    
    # 构建 System Prompt
    system_prompt_content = f"""
    你是一个规划器（Planner）。

    当前知识库概览：
    {_docs_catalog_markdown}

    可用工具：
    {tools_context}

    判断规则：
    1. 如果用户问题是关于系统能力、问候或文档库概览（如"你能做什么"、"有哪些文档"），调用 list_docs_tool
    2. 如果用户问题涉及具体技术内容、术语、方法论，调用 rag_search_tool
    3. 如果用户包含纠错倾向或新的检索锚点，必须调用 rag_search_tool
    4. 如果问题与文档内容无关，可以直接回答（respond）
    
    当前智能路由判断：
    - 目标文档ID: {target_doc_id}
    - 置信度: {confidence:.2f}
    - 理由: {reasoning}
    
    输出要求：
    - action: "tool" 或 "respond"
    - tool_name: 要调用的工具名（action=tool 时必填）
    - tool_args: 工具参数（action=tool 时必填，必须包含 doc_id 参数）
    - reason: 判断理由
    """
    
    try:
        # 强制调用工具的情况
        if has_correction or has_anchor:
            plan = Plan(
                action="tool",
                tool_name="rag_search_tool",
                tool_args={"query": user_input, "doc_id": target_doc_id if target_doc_id != "none" else "all"},
                reason="检测到纠错倾向或检索锚点，需要验证"
            )
            console.print(f"[cyan]Planner 输出: {plan.action} | {plan.tool_name} | doc_id={target_doc_id}[/cyan]")
            return {
                "plan": plan.dict(),
                "target_doc_id": target_doc_id
            }
        
        # 使用 LLM 判断
        plan = llm_plain.with_structured_output(Plan).invoke([
            SystemMessage(content=system_prompt_content),
            current_message
        ])
        
        # 确保 tool_args 包含 doc_id
        if plan.action == "tool" and plan.tool_name == "rag_search_tool":
            if not plan.tool_args:
                plan.tool_args = {}
            if "doc_id" not in plan.tool_args:
                plan.tool_args["doc_id"] = target_doc_id if target_doc_id != "none" else "all"
        
        console.print(f"[cyan]Planner 输出: {plan.action} | {plan.tool_name}[/cyan]")
        return {
            "plan": plan.dict(),
            "target_doc_id": target_doc_id
        }
        
    except Exception as e:
        console.print(f"[yellow]Planner 异常，fallback 到 respond: {e}[/yellow]")
        return {
            "plan": {"action": "respond", "reason": f"异常: {e}"},
            "target_doc_id": "all"
        }


def call_model(state: AgentState):
    """
    调用模型生成回复或工具调用
    """
    plan = state.get("plan", {})
    target_doc_id = state.get("target_doc_id", "all")
    
    # 构建系统提示词，注入文档库概览
    sys_prompt = f"""你是一个专业的文档分析助手。

{_docs_catalog_markdown}

重要原则：
1. 你只能基于检索到的文档内容回答问题
2. 如果检索结果为空或无法回答用户问题，必须回复"抱歉，在当前知识库中没有检索到相关内容"
3. 严禁使用自身通用知识进行脑补或推测
4. 回答必须注明依据来源
"""
    
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    
    # 检查是否已有工具结果
    has_tool_result = any(isinstance(msg, ToolMessage) for msg in state["messages"])
    last_tool_result = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            last_tool_result = msg.content
            break
    
    # 检查检索结果是否为空
    retrieval_empty = False
    if last_tool_result and "【检索结果为空】" in last_tool_result:
        retrieval_empty = True
    
    if plan.get("action") == "tool" or has_tool_result:
        tool_hint = """
执行流程：
1. 如果还没有调用工具，先调用 rag_search_tool 获取相关信息
2. 如果已经收到 rag_search_tool 的结果：
   - 若结果为空，直接调用 submit_final_answer，answer 字段填写"抱歉，在当前知识库中没有检索到相关内容"
   - 若结果不为空，基于检索内容生成答案，调用 submit_final_answer
3. 答案必须包含：回答内容、依据来源和置信度

严禁：在没有检索结果的情况下使用自身知识回答问题！
"""
        messages = messages + [SystemMessage(content=tool_hint)]
        
        # 如果检索为空，直接返回拒绝回答的提示
        if retrieval_empty:
            messages = messages + [SystemMessage(content="注意：检索结果为空，你必须回答'抱歉，在当前知识库中没有检索到相关内容'")]
        
        for attempt in range(3):
            try:
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                if "400" in str(e) or "JSON" in str(e):
                    console.print(f"[yellow]Tool call format error, retrying... ({attempt+1}/3)[/yellow]")
                    continue
                raise
        
        # 降级为普通回答
        console.print(f"[red]Tool call failed after 3 attempts, falling back.[/red]")
        response = llm_plain.invoke(messages)
    else:
        response = llm_plain.invoke(messages)
    
    return {"messages": [response]}


def execute_tools(state: AgentState):
    """
    执行工具调用
    """
    last_message = state["messages"][-1]
    tool_responses = []
    tool_map = {tool.name: tool for tool in tools}
    plan = state.get("plan", {})
    
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_obj = tool_map.get(tool_name)
            
            # 检查是否与 plan 一致
            expected_tool_name = plan.get("tool_name")
            if plan.get("action") == "respond" and tool_name != "list_docs_tool":
                console.print(f"[yellow]Planner decided to not use tools, but {tool_name} was called.[/yellow]")
                tool_responses.append(
                    ToolMessage(
                        content=f"Tool rejected: Planner decided to respond directly.",
                        tool_call_id=tool_call["id"]
                    )
                )
                continue
            
            if not tool_obj:
                console.print(f"[red]Unknown Tool: {tool_name}[/red]")
                continue
            
            try:
                # 调用工具
                args = tool_call["args"]
                # 确保 doc_id 参数存在
                if tool_name == "rag_search_tool" and "doc_id" not in args:
                    args["doc_id"] = state.get("target_doc_id", "all")
                
                res = tool_obj.invoke(args)
            except Exception as e:
                res = f"Tool error: {str(e)}"
            
            tool_responses.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
    
    if len(tool_responses) == 0:
        return {"messages": [ToolMessage(content="Error: No valid tool calls identified.", tool_call_id="unknown")]}
    
    return {"messages": tool_responses}


def reflector(state: AgentState):
    """
    反思节点：评估答案质量
    """
    last_ai_msg = None
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            last_ai_msg = m
            break
    
    if not last_ai_msg:
        return {"retry_count": 0}
    
    # 提取最终答案
    final_output = None
    if last_ai_msg.tool_calls:
        for tc in last_ai_msg.tool_calls:
            if tc["name"] == "submit_final_answer":
                args = tc["args"]
                final_output = args.get("output") if isinstance(args, dict) else args
                break
    
    if not final_output:
        return {"retry_count": 0}
    
    ans = final_output.get("answer", "") if isinstance(final_output, dict) else ""
    
    # 检查是否是"无相关内容"的合规回答
    if "没有检索到相关内容" in ans or "抱歉" in ans:
        if DEBUG_MODE:
            console.print(f"[green][DEBUG] 答案为合规的'无结果'回复，通过评估[/green]")
        return {"retry_count": 0}
    
    # 找到对应的检索结果
    search_tool_call_id = None
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if tc["name"] == "rag_search_tool":
                    search_tool_call_id = tc["id"]
                    break
        if search_tool_call_id:
            break
    
    retrieved_content = ""
    if search_tool_call_id:
        for m in state["messages"]:
            if isinstance(m, ToolMessage) and m.tool_call_id == search_tool_call_id:
                retrieved_content = m.content
                break
    
    # 进行 Faithfulness 检验
    if retrieved_content and ans and "【检索结果为空】" not in retrieved_content:
        try:
            system_prompt = SystemMessage(content="""
你是一个回答质量评估器，专注于检测"事实性冲突"。

评估准则：
1. 核心事实：回答中的关键结论、数据、公式、专有名词是否与原文冲突？（冲突则不合格）
2. 幻觉定义：只有当回答编造了原文未提及的模型名、实验数据、或完全无关的结论时，才判定为幻觉。
3. 允许范围：允许合理的逻辑串联、结构化排版和语义转述。
4. 完整性：回答是否解决了用户的核心疑问？

只输出 JSON：{"pass": true} 或 {"pass": false, "reason": "具体原因"}
""")
            
            human_prompt = HumanMessage(content=f"文档片段：\n{retrieved_content[:3000]}\n\n回答：\n{ans}")
            
            evaluation_result = llm_plain.invoke([system_prompt, human_prompt])
            
            eval_json = json.loads(evaluation_result.content.replace("```json", "").replace("```", "").strip())
            
            if not eval_json.get("pass", False):
                reason = eval_json.get("reason", "回答与文档内容不符")
                feedback = f"【质量评估】: 回答存在问题：{reason}。请重新基于文档内容生成回答。"
                console.print(f"[yellow]反思节点: 答案未通过评估 - {reason}[/yellow]")
                return {
                    "messages": [HumanMessage(content=feedback)],
                    "retry_count": state.get("retry_count", 0) + 1
                }
            else:
                console.print(f"[green]反思节点: 答案质量过关。[/green]")
                
        except Exception as e:
            console.print(f"[yellow]Faithfulness 检验失败: {e}[/yellow]")
            # 退化为置信度检查
            conf = final_output.get("confidence", 0) if isinstance(final_output, dict) else 0
            if conf < 0.6 or len(ans) < 10:
                feedback = f"【自我反思驱动】: 当前回答置信度仅为 {conf}，请重新深入检索。"
                return {
                    "messages": [HumanMessage(content=feedback)],
                    "retry_count": state.get("retry_count", 0) + 1
                }
    elif "【检索结果为空】" in retrieved_content:
        # 检索为空时，检查是否正确回应
        if "没有检索到相关内容" not in ans and "抱歉" not in ans:
            feedback = "【质量评估】: 检索结果为空，但你未按规则回复。必须回复'抱歉，在当前知识库中没有检索到相关内容'。"
            return {
                "messages": [HumanMessage(content=feedback)],
                "retry_count": state.get("retry_count", 0) + 1
            }
    
    return {"retry_count": 0}


def router_after_planner(state: AgentState):
    """规划器后的路由"""
    plan = state.get("plan", {})
    if plan.get("action") in ["tool", "respond"]:
        return "agent"
    return "agent"


def should_continue(state: AgentState):
    """判断流程是否继续"""
    last_message = state["messages"][-1]
    
    # 检查是否是工具执行结果
    if isinstance(last_message, ToolMessage):
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                if any(tc["name"] == "rag_search_tool" for tc in msg.tool_calls):
                    return "continue"
                if any(tc["name"] == "submit_final_answer" for tc in msg.tool_calls):
                    return "reflect"
                break
        return "end"
    
    # 检查是否有工具调用
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    
    # 检查是否调用了 submit_final_answer
    if any(tc["name"] == "submit_final_answer" for tc in last_message.tool_calls):
        return "reflect"
    
    return "continue"


def check_reflection(state: AgentState):
    """检查反思结果"""
    if state.get("retry_count", 0) > 0 and state["retry_count"] < 3:
        return "re-think"
    return "end"


# ==========================================
# 5. 构建 LangGraph 工作流
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.add_node("evaluator", reflector)

workflow.add_edge(START, "planner")
workflow.add_conditional_edges(
    "planner",
    router_after_planner,
    {"agent": "agent"}
)
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "reflect": "evaluator",
        "end": END
    }
)
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges(
    "evaluator",
    check_reflection,
    {
        "re-think": "planner",
        "end": END
    }
)

app = workflow.compile(checkpointer=MemorySaver())

# 打印工作流图
mermaid_code = app.get_graph().draw_mermaid()
print(mermaid_code)


# ==========================================
# 6. 交互运行
# ==========================================
if __name__ == "__main__":
    # --- 初始化 RAG 引擎 ---
    with console.status("[bold green]正在初始化 RAG 引擎...", spinner="dots"):
        get_rag_engine()
    
    # --- 加载文档目录 ---
    with console.status("[bold green]正在加载文档库目录...", spinner="dots"):
        catalog, catalog_md = load_document_catalog()
    
    # 显示系统就绪信息
    doc_count = catalog.get("count", 0)
    console.print(Panel(
        f"[bold green]PDF 专家已就绪[/bold green]\n"
        f"[文档数量]: [cyan]{doc_count}[/cyan] 篇\n"
        f"[设备]: [magenta]NVIDIA RTX 4090[/magenta]\n"
        f"[模型]: [yellow]{LLM_MODEL}[/yellow]",
        title="System Ready",
        border_style="green"
    ))
    
    # 显示文档库概览
    if doc_count > 0:
        console.print(Markdown(catalog_md))
    else:
        console.print(Panel(
            "[yellow]文档库为空[/yellow]\n"
            "请先运行: [cyan]python ingestion_pipeline.py[/cyan]",
            title="文档库状态",
            border_style="yellow"
        ))
    
    config = {"configurable": {"thread_id": "gen_pdf_agent_v8"}}
    
    while True:
        user_input = console.input("\n[bold yellow]User ➤ [/bold yellow]")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "retry_count": 0,
            "target_doc_id": "all"
        }
        
        final_answer_displayed = False
        
        with console.status("[bold blue]AI 正在处理...", spinner="bouncingBall") as status:
            for event in app.stream(initial_state, config, stream_mode="updates"):
                for node_name, values in event.items():
                    status.update(f"[bold blue]当前环节: {node_name}...")
                    
                    if "messages" in values and len(values["messages"]) > 0:
                        msg = values["messages"][-1]
                        
                        if node_name == "agent" and isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    if tc['name'] == 'submit_final_answer':
                                        raw_args = tc['args']
                                        final_data = raw_args.get('output', raw_args) if isinstance(raw_args, dict) else raw_args
                                        
                                        # 格式化最终答案
                                        report_md = f"### 深度分析报告\n\n"
                                        report_md += f"**【回答】**\n{final_data.get('answer', '无')}\n\n"
                                        
                                        src_list = final_data.get('sources', [])
                                        if src_list:
                                            src_str = "\n".join([f"- {s}" for s in src_list]) if isinstance(src_list, list) else str(src_list)
                                            report_md += f"**【依据来源】**\n{src_str}\n\n"
                                        
                                        conf = final_data.get('confidence', 'N/A')
                                        report_md += f"---\n*置信度评分: `{conf}`*"
                                        
                                        console.print(Markdown(report_md))
                                        final_answer_displayed = True
        
        if not final_answer_displayed:
            # 如果没有显示最终答案，显示最后一条消息
            final_state = app.get_state(config)
            if final_state and final_state.values.get("messages"):
                last_msg = final_state.values["messages"][-1]
                if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                    console.print(Markdown(last_msg.content))
