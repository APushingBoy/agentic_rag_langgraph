# ==========================================
# 1. 导入必要的包 (新增 pydantic)
# ==========================================
from dotenv import load_dotenv
import os
import operator
import json
from typing import Annotated, TypedDict, List, Optional
from pydantic import BaseModel, Field # 新增：用于结构化输出

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from src.agentic_rag.tools.custom_tool import DocumentSearchTool

load_dotenv()

# ==========================================
# 2. 结构化输出定义 (Structured Output Schema)
# ==========================================
class FinalAnswer(BaseModel):
    """最终回复的结构化模型"""
    answer: str = Field(description="针对用户问题的详细回答")
    sources: List[str] = Field(description="回答所依据的简历具体片段或章节名称")
    confidence: float = Field(description="回答的置信度，0到1之间")
    needs_follow_up: bool = Field(description="是否需要面试官进一步追问或确认")

# ==========================================
# 3. 系统提示词升级 (注入结构化要求)
# ==========================================
SYSTEM_PROMPT = """你是一名专业的 AI 应用工程师面试官助手。
你的任务是基于简历 PDF 提供准确的信息。

工作流规则：
1. **必须检索**：对于任何事实性问题，先调用 `search_pdf_tool`。
2. **处理错误**：如果工具返回错误信息，请尝试理解错误或告知用户无法读取特定内容。
3. **结构化结束**：当你准备好回答时，必须通过 `FinalAnswer` 工具来输出你的结果。
4. **禁止幻觉**：如果检索结果为空，请在 `FinalAnswer` 中如实说明，不要编造经历。
"""

# ==========================================
# 4. 状态与工具准备 (增加异常捕获逻辑)
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add] 

pdf_searcher = DocumentSearchTool(file_path=r"C:\Users\SYK84\Documents\史玉坤-AI应用工程师(LLM_RAG_Agent方向).pdf") 

@tool
def search_pdf_tool(query: str) -> str:
    """PDF 搜索工具。用于查询简历内容、项目细节、工作时间等。"""
    # 注意：这里的异常捕获放在了 _run 内部或此处
    try:
        if not query:
            return "错误：搜索查询词不能为空，请输入具体的关键词。"
        result = pdf_searcher._run(query)
        return result
    except Exception as e:
        return f"工具执行过程中发生故障: {str(e)}"

# 将最终回复也定义为一个工具，这是强制结构化输出的一种高级技巧
@tool
def submit_final_answer(output: FinalAnswer) -> str:
    """当你完成所有检索并准备好给出最终答案时，调用此工具。"""
    # 这个工具其实不需要执行什么，它的存在是为了让模型输出符合 FinalAnswer 的格式
    return "已提交最终答案"

# 包含搜索工具和结构化输出工具
tools = [search_pdf_tool, submit_final_answer]
llm = ChatOpenAI(model="qwen-plus", temperature=0)

# 让 LLM 知道它有两个工具可用
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 5. 节点定义 (增强容错性)
# ==========================================

def call_model(state: AgentState):
    """思考节点"""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def execute_tools(state: AgentState):
    """工具节点：增加了对模型生成的参数进行校验和报错处理"""
    last_message = state["messages"][-1]
    tool_responses = []
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            # 1. 获取工具名
            action_name = tool_call["name"]
            
            # 2. 尝试执行
            try:
                if action_name == "search_pdf_tool":
                    # 执行搜索
                    result = search_pdf_tool.invoke(tool_call["args"])
                elif action_name == "submit_final_answer":
                    # 最终回复工具直接返回参数内容
                    result = f"结构化回复已生成"
                else:
                    result = f"错误：未知的工具名称 '{action_name}'。"
            except Exception as e:
                # 【关键】：如果不幸崩溃，捕获它并传回给 Agent
                result = f"工具调用发生异常: {str(e)}。请检查参数格式并重试。"
                
            tool_msg = ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            tool_responses.append(tool_msg)
                
    return {"messages": tool_responses}

# ==========================================
# 6. 路由逻辑 (基于工具名判断)
# ==========================================
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    
    # 如果没有工具调用，说明模型可能直接输出了文字（这违反了我们的协议）
    if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
        return "end"
    
    # 如果模型调用了 submit_final_answer，则结束循环
    for call in last_message.tool_calls:
        if call["name"] == "submit_final_answer":
            return "end"
            
    # 其他情况（如调用 search_pdf_tool）继续循环
    return "continue"

# ==========================================
# 7. 构建图
# ==========================================
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", execute_tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")

app = workflow.compile(checkpointer=MemorySaver())

# ==========================================
# 8. 运行测试
# ==========================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "structured_test_001"}}
    
    print("--- 史玉坤简历 AI 助手 (结构化防御版) ---")

    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        
        for event in app.stream(initial_state, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            
            # 这里的打印逻辑也要升级，以便看到结构化数据
            if isinstance(last_msg, AIMessage):
                if last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        if tc['name'] == 'submit_final_answer':
                            # 当调用这个工具时，意味着结构化答案就在 tc['args'] 里
                            ans = tc['args']['output']
                            print(f"\nAssistant (Structured Answer):")
                            print(f"【回答】: {ans['answer']}")
                            print(f"【来源】: {ans['sources']}")
                            print(f"【置信度】: {ans['confidence']}")
                        else:
                            print(f"DEBUG: [Agent 正在调用 {tc['name']}...]")
            elif isinstance(last_msg, ToolMessage):
                # 检查工具返回是否包含“错误”字样
                if "错误" in last_msg.content or "异常" in last_msg.content:
                    print(f"DEBUG: [ 工具调用异常报告]: {last_msg.content}")
                else:
                    print(f"DEBUG: [ 工具数据返回成功]")