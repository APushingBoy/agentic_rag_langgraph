# ==========================================
# 1. 基础环境与 RAG 引擎准备
# ==========================================
from dotenv import load_dotenv
import os
import operator
import time
from typing import Annotated, TypedDict, List
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- 引入 Rich 美化库 ---
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

console = Console()

# --- 关键修改：导入原生工具和初始化函数 ---
from rag_tool import search_pdf_tool, init_rag_engine

load_dotenv()

# --- 文件选择逻辑 ---
def select_file():
    console.print("\n[bold cyan]项目助手:[/bold cyan] 请手动输入或粘贴 PDF 路径")
    path = console.input("[bold yellow]路径 ➤ [/bold yellow]").strip().replace("'", "").replace('"', "")
    if os.path.exists(path) and path.lower().endswith(".pdf"):
        return path
    return None


SELECTED_PDF_PATH = select_file()
if not SELECTED_PDF_PATH:
    console.print("[bold red] 未选择有效文件，程序退出。[/bold red]")
    exit()

# ==========================================
# 2. 结构化输出与工具绑定
# ==========================================
class FinalAnswer(BaseModel):
    answer: str = Field(description="针对问题的正式回答")
    sources: List[str] = Field(description="依据的文档具体原文或章节")
    confidence: float = Field(description="置信度 0-1")

from langchain_core.tools import tool
@tool
def submit_final_answer(output: FinalAnswer) -> str:
    """提交最终经过反思的结构化答案。"""
    return "SUCCESS"

tools = [search_pdf_tool, submit_final_answer]
llm = ChatOpenAI(model="qwen-plus", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. 状态与节点定义 (完全保留原逻辑)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    retry_count: int

def call_model(state: AgentState):
    sys_prompt = f"你正在分析：{os.path.basename(SELECTED_PDF_PATH)}。请严谨回答。"
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def execute_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_responses = []
    tool_map = {tool.name: tool for tool in tools}
    
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_obj = tool_map.get(tool_call["name"])
            if tool_obj:
                res = tool_obj.invoke(tool_call["args"])
                tool_responses.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
    return {"messages": tool_responses}

def reflector(state: AgentState):
    last_ai_msg = None
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            last_ai_msg = m
            break
    if not last_ai_msg: return {"retry_count": 0}

    final_output = None
    if last_ai_msg.tool_calls:
        for tc in last_ai_msg.tool_calls:
            if tc["name"] == "submit_final_answer":
                args = tc["args"]
                final_output = args.get("output") if isinstance(args, dict) else args
                break
    
    if final_output:
        conf = final_output.get("confidence", 0) if isinstance(final_output, dict) else 0
        ans = final_output.get("answer", "") if isinstance(final_output, dict) else ""
        if conf < 0.6 or len(ans) < 10:
            feedback = f"【自我反思驱动】: 当前回答置信度仅为 {conf}，内容较单薄。请重新深入检索 PDF，挖掘更多细节以增强回答的权威性。"
            return {"messages": [HumanMessage(content=feedback)], "retry_count": state.get("retry_count", 0) + 1}
    return {"retry_count": 0}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    if any(tc["name"] == "submit_final_answer" for tc in last_message.tool_calls):
        return "reflect"
    return "continue"

def check_reflection(state: AgentState):
    if state.get("retry_count", 0) > 0 and state["retry_count"] < 3:
        return "re-think"
    return "end"

# ==========================================
# 7. 构建图 (保持不变)
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.add_node("evaluator", reflector)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "reflect": "evaluator", "end": END})
workflow.add_edge("tools", "agent")
workflow.add_conditional_edges("evaluator", check_reflection, {"re-think": "agent", "end": END})
app = workflow.compile(checkpointer=MemorySaver())


# ==========================================
# 8. 交互运行 (终极防吞字修复版)
# ==========================================
if __name__ == "__main__":
    # --- 初始化 RAG 引擎 ---
    with console.status(f"[bold green]正在为 {os.path.basename(SELECTED_PDF_PATH)} 构建索引...", spinner="dots"):
        init_rag_engine(SELECTED_PDF_PATH)
    
    console.print(Panel(
        f" [bold green]PDF 专家已就绪[/bold green]\n[文件]: [cyan]{os.path.basename(SELECTED_PDF_PATH)}[/cyan]\n[设备]: [magenta]NVIDIA RTX 4090[/magenta]",
        title="System Ready", border_style="green"
    ))
    
    config = {"configurable": {"thread_id": "gen_pdf_agent_004"}} # 换个新ID避免缓存干扰

    while True:
        user_input = console.input("\n[bold yellow]User ➤ [/bold yellow]")
        if user_input.lower() in ["exit", "quit"]: break
        
        initial_state = {"messages": [HumanMessage(content=user_input)], "retry_count": 0}
        
        # 启动状态条，交给 rich 自动管理，我们内部绝对不去干涉它
        with console.status("[bold blue]AI 正在处理...", spinner="bouncingBall") as status:
            for event in app.stream(initial_state, config, stream_mode="updates"):
                for node_name, values in event.items():
                    # 动态更新文字
                    status.update(f"[bold blue]当前环节: {node_name}...")
                    
                    if "messages" in values:
                        msg = values["messages"][-1]

                        if node_name == "agent" and isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    if tc['name'] == 'submit_final_answer':
                                        raw_args = tc['args']
                                        final_data = raw_args.get('output', raw_args) if isinstance(raw_args, dict) else raw_args
                                        
                                        report_md = f"###  深度分析报告\n\n"
                                        report_md += f"**【回答】**\n{final_data.get('answer', '无')}\n\n"
                                        src_list = final_data.get('sources', [])
                                        src_str = "\n".join([f"- {s}" for s in src_list]) if isinstance(src_list, list) else str(src_list)
                                        report_md += f"**【依据来源】**\n{src_str}\n\n"
                                        report_md += f"---\n*置信度评分: `{final_data.get('confidence', 'N/A')}`*"
                                        
                                        # 【修复点】：直接 print，不要停状态条
                                        console.print(Panel(Markdown(report_md), title="Final Structured Report", border_style="green", padding=(1, 2)))
                                    else:
                                        console.print(f"  [dim] [Agent 决策] 调用工具: {tc['name']}...[/dim]")
                            
                            elif msg.content:
                                # 【修复点】：日常对话直接 print，瞬间渲染精美的 Markdown 框
                                console.print(Panel(Markdown(msg.content), title="Assistant", border_style="blue", padding=(1, 2)))

                        # 处理反思节点
                        if node_name == "evaluator":
                            current_retry = values.get("retry_count", 0)
                            if current_retry > 0:
                                console.print(f"  [bold orange1] 反思节点: 质量不合格 (第 {current_retry} 次打回)，正在优化检索...[/bold orange1]")
                            else:
                                console.print(f"  [bold green] 反思节点: 答案质量过关。[/bold green]")
