# ==========================================
# 1. 基础环境与 RAG 引擎准备
# ==========================================
from dotenv import load_dotenv
import os
import operator
from typing import Annotated, TypedDict, List
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- 关键修改：导入原生工具和初始化函数 ---
from rag_tool import search_pdf_tool, init_rag_engine

load_dotenv()

# --- 文件选择逻辑 (保持不变) ---
def select_file():
    print("\n--- Linux 模式：请手动输入或粘贴 PDF 路径 ---")
    path = input("请输入 PDF 完整路径: ").strip().replace("'", "").replace('"', "")
    if os.path.exists(path) and path.lower().endswith(".pdf"):
        return path
    return None

SELECTED_PDF_PATH = select_file()
if not SELECTED_PDF_PATH:
    print("未选择文件，程序退出。")
    exit()

# --- 【核心步骤】：在程序启动前，初始化 4090 上的 RAG 引擎 ---
print("正在初始化 RAG 引擎，请稍候...")
init_rag_engine(SELECTED_PDF_PATH)

# ==========================================
# 2. 结构化输出与工具绑定
# ==========================================
class FinalAnswer(BaseModel):
    answer: str = Field(description="针对问题的正式回答")
    sources: List[str] = Field(description="依据的文档具体原文或章节")
    confidence: float = Field(description="置信度 0-1")

# 原生装饰器定义提交工具
from langchain_core.tools import tool
@tool
def submit_final_answer(output: FinalAnswer) -> str:
    """提交最终经过反思的结构化答案。"""
    return "SUCCESS"

# 这里的 search_pdf_tool 是从 custom_tool.py 导入的原生工具
tools = [search_pdf_tool, submit_final_answer]
llm = ChatOpenAI(model="qwen-plus", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 3. 状态与节点定义 (保持你的逻辑，仅修正工具调用)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    retry_count: int

def call_model(state: AgentState):
    # 系统提示词中动态加入文件名
    sys_prompt = f"你正在分析：{os.path.basename(SELECTED_PDF_PATH)}。请严谨回答。"
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def execute_tools(state: AgentState):
    """节点 B: 工具执行 (Worker)"""
    last_message = state["messages"][-1]
    tool_responses = []
    
    # 查找工具映射表
    tool_map = {tool.name: tool for tool in tools}
    
    if last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_obj = tool_map.get(tool_call["name"])
            if tool_obj:
                # 直接通过 invoke 调用原生工具
                res = tool_obj.invoke(tool_call["args"])
                tool_responses.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
    return {"messages": tool_responses}

def reflector(state: AgentState):
    """节点 C: 评价者 (Evaluator) - 核心反思逻辑"""
    
    # --- 修复点 1：稳妥地获取最后一条 AIMessage ---
    last_ai_msg = None
    # 从后往前找，确保拿到的是 AIMessage
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            last_ai_msg = m
            break
            
    if not last_ai_msg:
        return {"retry_count": 0}

    # 找到 FinalAnswer 的参数
    final_output = None
    if last_ai_msg.tool_calls:
        for tc in last_ai_msg.tool_calls:
            if tc["name"] == "submit_final_answer":
                # 注意：有些模型返回的 args 已经是字典，有些是字符串，这里做个兼容
                args = tc["args"]
                # 这里的 key 取决于你的 FinalAnswer 模型在 submit_final_answer 里的参数名
                # 如果你定义的是 def submit_final_answer(output: FinalAnswer)，那么 key 就是 'output'
                final_output = args.get("output") if isinstance(args, dict) else args
                break
    
    # --- 修复点 2：反思逻辑增强 ---
    if final_output:
        # 即使模型没传信心值，我们也给个默认值防止崩溃
        conf = final_output.get("confidence", 0) if isinstance(final_output, dict) else 0
        ans = final_output.get("answer", "") if isinstance(final_output, dict) else ""
        
        # 触发反思的条件：置信度低，或者回答太敷衍
        if conf < 0.6 or len(ans) < 10:
            feedback = f"【自我反思驱动】: 当前回答置信度仅为 {conf}，内容较单薄。请重新深入检索 PDF，挖掘更多细节以增强回答的权威性。"
            # 返回 HumanMessage 作为“鞭策”指令发给 agent 节点
            return {"messages": [HumanMessage(content=feedback)], "retry_count": state.get("retry_count", 0) + 1}
    
    return {"retry_count": 0}

# ==========================================
# 6. 路由逻辑
# ==========================================
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    
    # 如果最后一条消息没有工具调用，说明它已经给出了文本回答，直接结束
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    
    # 如果调用了提交答案的工具，去反思节点
    if any(tc["name"] == "submit_final_answer" for tc in last_message.tool_calls):
        return "reflect"
        
    # 其他工具（如 search_pdf_tool）继续运行工具节点
    return "continue"

def check_reflection(state: AgentState):
    # 检查最后一条消息是不是反馈信息
    if state.get("retry_count", 0) > 0 and state["retry_count"] < 3:
        return "re-think" # 回到 agent 重新思考
    return "end"

# ==========================================
# 7. 构建带有反思环路的图
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)
workflow.add_node("evaluator", reflector) # 新增评价节点
workflow.add_edge(START, "agent")

# 第一层路由：决定是去调工具还是去评价
workflow.add_conditional_edges(
    "agent", 
    should_continue, 
    {"continue": "tools", "reflect": "evaluator", "end": END}
)

workflow.add_edge("tools", "agent")

# 第二层路由：评价完后，决定是结束还是打回重做
workflow.add_conditional_edges(
    "evaluator",
    check_reflection,
    {"re-think": "agent", "end": END}
)

app = workflow.compile(checkpointer=MemorySaver())

# ==========================================
# 8. 交互运行
# ==========================================
if __name__ == "__main__":
    # --- 关键修改 1：在此处初始化 RAG 引擎 (4090 显存加载点) ---
    print(f"DEBUG: 正在为 {os.path.basename(SELECTED_PDF_PATH)} 构建索引...")
    # from rag_tool import init_rag_engine
    init_rag_engine(SELECTED_PDF_PATH)
    
    config = {"configurable": {"thread_id": "gen_pdf_agent_002"}} 
    print(f"\n---  PDF 专家已就绪: {os.path.basename(SELECTED_PDF_PATH)} ---")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        # initial_state 只传当前问题，历史记录由 Checkpointer (MemorySaver) 自动管理
        initial_state = {"messages": [HumanMessage(content=user_input)], "retry_count": 0}
        
        # 使用 stream 模式
        for event in app.stream(initial_state, config, stream_mode="updates"):
            for node_name, values in event.items():
                if "messages" in values:
                    msg = values["messages"][-1]

                    # --- 分类打印逻辑 ---
                    if node_name == "agent" and isinstance(msg, AIMessage):
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                if tc['name'] == 'submit_final_answer':
                                    # --- 关键修改 2：更稳健的参数提取 ---
                                    # 有些模型会把结果放在 tc['args']['output']，有些直接放 tc['args']
                                    raw_args = tc['args']
                                    final_data = raw_args.get('output', raw_args) if isinstance(raw_args, dict) else raw_args
                                    
                                    print(f"\n{'='*20} [ AI 结构化报告 ] {'='*20}")
                                    print(f"【回答】: {final_data.get('answer', '（未生成回答）')}")
                                    print(f"【来源】: {final_data.get('sources', '（未标注来源）')}")
                                    print(f"【置信度】: {final_data.get('confidence', 'N/A')}")
                                    print(f"{'='*56}\n")
                                else:
                                    print(f"DEBUG: [Agent 决策] 调用工具: {tc['name']} -> 参数: {tc['args']}")
                        
                        elif msg.content:
                            # 处理模型直接对话的情况
                            print(f"Assistant: {msg.content}")

                    # 处理评价结果
                    if node_name == "evaluator":
                        current_retry = values.get("retry_count", 0)
                        if current_retry > 0:
                            print(f" DEBUG: [反思节点] 质量不合格 (第 {current_retry} 次打回)，正在重新挖掘文档...")
                        else:
                            print(f" DEBUG: [反思节点] 答案质量过关。")