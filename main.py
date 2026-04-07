# ==========================================
# 1. 基础环境与 RAG 引擎准备
# ==========================================
from dotenv import load_dotenv
import os
import operator
import time
from typing import Annotated, TypedDict, List, Literal, Optional, Dict  
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
from rag_tools import search_pdf_tool, init_rag_engine

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
class Plan(BaseModel):
    action: Literal["tool", "respond"]
    # 这里的 tool_name 不再是普通的 str，而是被限制在工具名枚举里
    tool_name: Optional[str] = None
    # tool_args: Optional[Dict] = None 
    reason: Optional[str] = None


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
llm_plain = ChatOpenAI(model="qwen-plus", temperature=0)
llm_with_tools = llm_plain.bind_tools(tools)

# ==========================================
# 3. 状态与节点定义 (完全保留原逻辑)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    retry_count: int
    plan: dict  #  新增

def planner(state: AgentState):
    simplified_tools = []
    for t in tools:
        # 仅提取参数名和描述，过滤掉冗余的 Pydantic 校验信息
        args_desc = {k: v.get('description', '无描述') for k, v in t.args.items()}
        simplified_tools.append(f"工具: {t.name}\n参数: {args_desc}")

    tools_context = "\n\n".join(simplified_tools)
    messages = state["messages"]
    current_message = messages[-1]
    
    # 检查用户输入是否包含纠错倾向或新的检索锚点
    user_input = current_message.content.lower()
    
    # 检测纠错倾向关键词
    correction_keywords = ["你确定吗", "再搜一下", "明明有", "在xx章节", "在章节", "页码", "第.*章"]
    has_correction = any(keyword in user_input for keyword in correction_keywords)
    
    # 检测新的检索锚点（如引号中的术语、具体章节标题）
    has_anchor = False
    if '"' in user_input or '“' in user_input or '”' in user_input:
        has_anchor = True
    if any(keyword in user_input for keyword in ["章节", "页码", "section", "chapter", "page"]):
        has_anchor = True
    
    # 构建增强的 System Prompt
    system_prompt = SystemMessage(content=f"""
    你是一个规划器（planner）。

    你的任务：
    判断用户问题是否需要调用工具。
                                 
    可选的工具和参数如下：
    {tools_context}

    规则：
    - 如果问题明显涉及到PDF，那么优先使用 tool
    - 如果问题和PDF并无直接联系，则可以直接回答，请使用 respond
    - 铁律：如果用户的Query包含纠错倾向（如"你确定吗？"、"再搜一下"、"明明有"、"在XX章节"），严禁直接回答，必须强制调用 search_pdf_tool 进行二次验证
    - 自我纠正：如果用户提供了新的检索锚点（如引号中的术语、具体章节标题、页码），必须立即触发 search_pdf_tool
    - 只输出结构化结果

    不要解释，不要输出多余文本
    """)

    try:
        # 如果检测到纠错倾向或新的检索锚点，直接生成调用工具的计划
        if has_correction or has_anchor:
            plan = Plan(action="tool", tool_name="search_pdf_tool", reason="检测到纠错倾向或新的检索锚点，需要进行验证")
            console.print(f"[cyan]Planner 输出: {plan}[/cyan]")
            return {"plan": plan.dict()}
        
        # 否则使用 LLM 进行判断
        plan = llm_plain.with_structured_output(Plan).invoke(
            [system_prompt, current_message]
        )
        console.print(f"[cyan]Planner 输出: {plan}[/cyan]")
        return {"plan": plan.dict()} # 我发现这里“plan”其实是Pydantic类型，所以要转换成字典
    except Exception:
        return {"plan": {"action": "respond"}}

def call_model(state: AgentState):
    plan = state.get("plan", {})

    sys_prompt = f"你正在分析：{os.path.basename(SELECTED_PDF_PATH)}。请严谨回答。"
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]

    # 检查是否已经有工具执行结果
    has_tool_result = any(isinstance(msg, ToolMessage) for msg in state["messages"])
    
    if plan.get("action") == "tool" or has_tool_result:
        tool_hint = f"""
        执行流程：
        1. 如果还没有调用工具，先调用 search_pdf_tool 获取相关信息
        2. 如果已经收到 search_pdf_tool 的结果，立即调用 submit_final_answer 工具提交最终结构化答案
        3. 答案必须包含：回答内容、依据来源和置信度
        
        请按照这个流程执行，不要重复调用 search_pdf_tool
        """
        messages = messages + [SystemMessage(content=tool_hint)]
        for attempt in range(3):  # 最多重试3次
            try:
                response = llm_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                if "400" in str(e) or "JSON" in str(e):
                    console.print(f"[yellow]Tool call format error, retrying... ({attempt+1}/3)[/yellow]")
                    continue
                raise  # 其他错误直接抛出
        
        # 3次都失败，降级为普通回答
        console.print(f"[red]Tool call failed after 3 attempts, falling back to direct response.[/red]")
        response = llm_plain.invoke(messages)
    else:
        response = llm_plain.invoke(messages)

    return {"messages": [response]}

def execute_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_responses = []
    tool_map = {tool.name: tool for tool in tools}
    plan = state.get("plan", {})
    
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_obj = tool_map.get(tool_call["name"])
            expected_tool_name = plan.get("tool_name")
            if plan.get("action") == "respond":
                console.print(f"[yellow]Planner decided to not use tools.[/yellow]")
                tool_responses.append(
                    ToolMessage(
                        content=f"Planner decided to not use tools at the current step.",
                        tool_call_id=tool_call["id"]
                    )
                ) # 加入一条tool message，让后面llm知道之前发生了什么，而不是稀里糊涂就跳过了
                continue
            if expected_tool_name and tool_call["name"] != expected_tool_name: # 不允许LLM试图调用不同的工具，LLM必须听从planner的控制
                console.print(f"Tool Mismatch! planner: {expected_tool_name}, actual: {tool_call['name']}") # 外面有双引号时内部应该用单引号
                tool_responses.append(
                    ToolMessage(
                        content=f"Tool rejected: expected {expected_tool_name}, got {tool_call['name']} . Please correct your tool call and try again.",
                        tool_call_id=tool_call["id"]
                    )
                ) # 加入一条tool message，让后面llm知道之前发生了什么，而不是稀里糊涂就跳过了
                continue

            if not tool_obj:
                console.print(f"[red]Unknown Tool: {tool_call['name']}[/red]")
                continue
            
            try:
                res = tool_obj.invoke(tool_call["args"])
            except Exception as e:
                res = f"Tool error: {str(e)}"
            tool_responses.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
    if len(tool_responses)==0:
        return {"messages": [ToolMessage(content="Error: No valid tool calls identified.", tool_call_id="unknown")]}
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
        ans = final_output.get("answer", "") if isinstance(final_output, dict) else ""
        
        # 1. 找到最近一次 search_pdf_tool 返回的 ToolMessage
        # search_tool_message = None
        # for m in reversed(state["messages"]):
        #     if isinstance(m, ToolMessage) and "search_pdf_tool" in m.content:
        #         search_tool_message = m
        #         break
        # 以上是Trae的代码，claude说是错的，说要改成下面这个
        # 先找到 search_pdf_tool 的 tool_call_id
        search_tool_call_id = None
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    if tc["name"] == "search_pdf_tool":
                        search_tool_call_id = tc["id"]
                        break
            if search_tool_call_id:
                break
        
        # 2. 提取检索到的原始文档内容
        # retrieved_content = ""
        # if search_tool_message:
        #     retrieved_content = search_tool_message.content
        # 再用 id 找对应的 ToolMessage
        retrieved_content = ""
        if search_tool_call_id:
            for m in state["messages"]:
                if isinstance(m, ToolMessage) and m.tool_call_id == search_tool_call_id:
                    retrieved_content = m.content
                    break

        
        # 3. 进行 Faithfulness 检验
        if retrieved_content and ans:
            try:
                system_prompt = SystemMessage(content="""
                你是一个回答质量评估器，专注于检测“事实性冲突”。

                评估准则：
                1. 核心事实：回答中的关键结论、数据、公式、专有名词是否与原文冲突？（冲突则不合格）
                2. 幻觉定义：只有当回答编造了原文未提及的模型名、实验数据、或完全无关的结论时，才判定为幻觉。
                3. 允许范围：允许合理的逻辑串联、结构化排版和语义转述。不要因为措辞略微不同或增加了引导语（如“根据文档”）而判定不合格。
                4. 完整性：回答是否解决了用户的核心疑问？（如果文档中有答案但回答没提，判定为不合格）。

                只输出 JSON：{"pass": true} 或 {"pass": false, "reason": "具体原因"}
                """)
                
                human_prompt = HumanMessage(content=f"文档片段：\n{retrieved_content}\n\n回答：\n{ans}")
                
                evaluation_result = llm_plain.invoke([system_prompt, human_prompt])
                
                # 解析 JSON 结果
                import json
                eval_json = json.loads(evaluation_result.content)
                
                if not eval_json.get("pass", False):
                    reason = eval_json.get("reason", "回答与文档内容不符")
                    feedback = f"【质量评估】: 回答存在问题：{reason}。请重新基于文档内容生成回答，确保所有陈述都有依据。"
                    return {"messages": [HumanMessage(content=feedback)], "retry_count": state.get("retry_count", 0) + 1}
                
            except Exception as e:
                # 异常处理：退化为原来的 confidence 逻辑
                console.print(f"[yellow]Faithfulness 检验失败，退化为置信度判断: {str(e)}[/yellow]")
                conf = final_output.get("confidence", 0) if isinstance(final_output, dict) else 0
                if conf < 0.6 or len(ans) < 10:
                    feedback = f"【自我反思驱动】: 当前回答置信度仅为 {conf}，内容较单薄。请重新深入检索 PDF，挖掘更多细节以增强回答的权威性。"
                    return {"messages": [HumanMessage(content=feedback)], "retry_count": state.get("retry_count", 0) + 1}
        else:
            # 如果没有检索内容，退化为检查回答长度
            if len(ans) < 10:
                feedback = "【质量评估】: 回答内容严重不足。请重新生成更详细的回答。"
                return {"messages": [HumanMessage(content=feedback)], "retry_count": state.get("retry_count", 0) + 1}
    return {"retry_count": 0}

def router_after_planner(state: AgentState):
    plan = state.get("plan", {})

    if plan.get("action") == "tool" or plan.get("action") == "respond": 
        return "agent"
    else:
        plan = {"action": "respond"} # 当planner抽风时，fallback到respond
        state["plan"] = {"action": "respond"} # 其实在planner里面已经有exception兜底，确保fallback到respond，但工程上考虑这里也写上比较好
        return "agent"
"""
TODO
当 Planner 说 respond，但 Agent 还是调用了工具，execute_tools 写回了一条拒绝的 ToolMessage。
然后流程回到 agent，Agent 看到工具被拒绝，重新生成了一个普通文本回答。
这时候 should_continue 看到最后一条消息是 AIMessage，没有 tool_calls，直接返回 "end"。
表面上正常结束了，但这条回答没有经过 Evaluator 评估。
Planner 说 respond 的路径，永远绕过了反思环路。
"""
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    
    # 检查是否是工具执行结果
    if isinstance(last_message, ToolMessage):
        # 检查之前的消息是否有工具调用
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                # 如果之前调用了 search_pdf_tool，现在应该调用 submit_final_answer
                if any(tc["name"] == "search_pdf_tool" for tc in msg.tool_calls):
                    return "agent"  # 回到 agent 节点生成 submit_final_answer
                # 如果之前调用了 submit_final_answer，现在应该进行反思
                if any(tc["name"] == "submit_final_answer" for tc in msg.tool_calls):
                    return "reflect"  # 进入反思节点
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
    if state.get("retry_count", 0) > 0 and state["retry_count"] < 3:
        return "re-think"
    return "end"

# ==========================================
# 7. 构建图 (保持不变)
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
    {
        "agent": "agent", 
        "end": END
    }
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
mermaid_code=app.get_graph().draw_mermaid()
print(mermaid_code)


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
                    
                    if "messages" in values and len(values["messages"]) > 0:
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
