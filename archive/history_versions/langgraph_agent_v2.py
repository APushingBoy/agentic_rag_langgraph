# ==========================================
# 1. 导入必要的包 (新增了 MemorySaver 和 SystemMessage)
# ==========================================
from dotenv import load_dotenv
import os
import operator
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage # 新增 SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver # 新增：持久化内存

# 导入你之前的工具
from src.agentic_rag.tools.custom_tool import DocumentSearchTool

load_dotenv()

# ==========================================
# 2. 定义系统提示词 (System Prompt) - Agent 的"灵魂协议"
# ==========================================
SYSTEM_PROMPT = """你是一名专业的 AI 应用工程师面试官助手。
你的核心任务是根据上传的 PDF 简历内容回答问题。

遵守以下原则：
1. **优先检索**：在回答任何关于候选人经历的问题前，必须先调用 `search_pdf_tool`。
2. **诚实原则**：如果 PDF 中没有相关信息，请直接回答“简历中未提及相关内容”，严禁基于常识幻觉（例如：不要因为候选人是 AI 方向就默认他会写 Python）。
3. **结构化回答**：回答时请条理清晰，重要信息（如时间、项目名）请加粗。
4. **多轮对话**：你可以参考之前的聊天记录来理解用户现在的意图（例如用户说“他呢？”，你应该知道“他”指的是简历本人）。
"""

# ==========================================
# 3. 定义状态 (保持不变)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add] 

# ==========================================
# 4. 定义工具与大脑 (保持不变)
# ==========================================
pdf_searcher = DocumentSearchTool(file_path=r"C:\Users\SYK84\Documents\史玉坤-AI应用工程师(LLM_RAG_Agent方向).pdf") 

@tool
def search_pdf_tool(query: str) -> str:
    """PDF 搜索工具。用于查询候选人的简历内容、项目细节、工作时间等。"""
    result = pdf_searcher._run(query)
    return result

tools = [search_pdf_tool]
llm = ChatOpenAI(model="qwen-plus", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 5. 定义节点 (更新了 call_model 以注入 SystemPrompt)
# ==========================================

def call_model(state: AgentState):
    """思考节点：负责注入系统指令并调用 LLM。"""
    # 1. 构造消息流：系统提示词 + 所有的历史对话记录
    # 注意：我们不把 SystemMessage 存入 state，而是每次调用时"动态注入"
    # 这样可以节省 token 且保证 Agent 的目标始终清晰
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def execute_tools(state: AgentState):
    """工具节点 (保持原有逻辑)"""
    last_message = state["messages"][-1]
    tool_responses = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            action_name = tool_call["name"]
            action_args = tool_call["args"]
            
            if action_name == "search_pdf_tool":
                result_str = search_pdf_tool.invoke(action_args)
                tool_msg = ToolMessage(content=str(result_str), tool_call_id=tool_call["id"])
                tool_responses.append(tool_msg)
                
    return {"messages": tool_responses}

# ==========================================
# 6. 定义路由 (保持不变)
# ==========================================
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

# ==========================================
# 7. 拼接图并挂载持久化内存 (Checkpointer)
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")

# 【核心更新】：实例化内存保存器
memory = MemorySaver()

# 【核心更新】：编译时加入 checkpointer
app = workflow.compile(checkpointer=memory)

# ==========================================
# 8. 运行持久化交互测试 (Run Interactive)
# ==========================================
if __name__ == "__main__":
    # 定义一个线程 ID（在实际应用中，每个用户或每个会话应该有不同的 ID）
    config = {"configurable": {"thread_id": "syk_interview_001"}}
    
    print("--- 史玉坤简历 AI 助手 (已开启持久化记忆) ---")
    print("输入 'exit' 或 'quit' 退出对话。")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        # 构造输入
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        
        # 运行图流。注意这里传入了 config 字典，它包含了 thread_id
        for event in app.stream(initial_state, config, stream_mode="values"):
            # 我们只看每一步产出的最后一条消息
            last_msg = event["messages"][-1]
            
            if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
                 print(f"Assistant: {last_msg.content}")
            elif isinstance(last_msg, AIMessage) and last_msg.tool_calls:
                 print(f"DEBUG: [Agent 决定搜索 PDF... 参数: {last_msg.tool_calls[0]['args']}]")
            elif isinstance(last_msg, ToolMessage):
                 print(f"DEBUG: [工具已返回 {len(last_msg.content)} 个字符的参考资料]")