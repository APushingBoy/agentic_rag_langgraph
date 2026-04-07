# ==========================================
# 1. 导入必要的包 (依赖准备)
# ==========================================
from dotenv import load_dotenv
import os                                   # 用于设置环境变量 (API Key等)
import operator                             # Python内置的运算符模块，这里用到加法 operator.add
from typing import Annotated, TypedDict     # 用于定义静态类型提示，让代码结构更严谨
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage # 导入 LangChain 的标准消息格式
from langchain_openai import ChatOpenAI     # 导入 OpenAI 格式的模型调用接口
from langgraph.graph import StateGraph, START, END # 导入 LangGraph 的核心图构建组件
from langchain_core.tools import tool       # 导入用于把普通函数变成 AI 工具的装饰器

# 这里假设你之前的 custom_tool.py 依然可用，并且能正常实例化
from src.agentic_rag.tools.custom_tool import DocumentSearchTool

# 配置 API 环境变量 (这里依然使用阿里千问的兼容模式)
# os.environ["OPENAI_API_BASE"] = ""
# os.environ["OPENAI_API_KEY"] = "" # 请替换为你的真实 Key

# 如今改用了.env文件，直接读取环境变量
load_dotenv()

# ==========================================
# 2. 定义状态 (State) - 这是整个图的"血液"
# ==========================================
# TypedDict 是一种特殊的字典，它规定了这个字典里只能有哪些键(Key)，以及它们的值(Value)是什么类型。
class AgentState(TypedDict):
    # 【最难懂的一行】：Annotated 和 operator.add 的组合
    # 意思：定义一个名为 messages 的变量，它的类型是 BaseMessage(消息) 组成的列表(list)。
    # 为什么需要 operator.add？
    # 默认情况下，LangGraph 如果接收到新数据，会直接"覆盖"旧数据。
    # 加上 operator.add 后，它告诉框架："当有新消息来时，不要覆盖，而是把新消息加到原列表的末尾 (append)。"
    # 这样我们的聊天记录和工具调用记录就能一直累积下去。
    messages: Annotated[list[BaseMessage], operator.add] 

# ==========================================
# 3. 定义工具 (Tools) - Agent 的"手"
# ==========================================
# 实例化你之前写好的 PDF 搜索工具 (为了测试，假设路径已经写死或通过参数传入)
pdf_searcher = DocumentSearchTool(file_path=r"C:\Users\SYK84\Documents\史玉坤-AI应用工程师(LLM_RAG_Agent方向).pdf") 

# @tool 装饰器是 LangChain 的魔法，它把下面这个普通的 Python 函数，
# 包装成了一个带有标准 JSON Schema 说明书的工具，大模型就能看懂它了。
@tool
def search_pdf_tool(query: str) -> str:
    """这是一个 PDF 搜索工具。当你需要回答关于简历、经历或文档内部的问题时，请使用此工具。"""
    # 这一行调用你 custom_tool.py 里的搜索逻辑
    result = pdf_searcher._run(query)
    return result

# 把工具打包成一个列表，待会儿要挂载到大模型身上
tools = [search_pdf_tool]

# ==========================================
# 4. 初始化大脑 (LLM)
# ==========================================
# 初始化千问模型，这里不需要写 "openai/" 前缀，直接写模型名即可
llm = ChatOpenAI(model="qwen-plus", temperature=0)

# 【关键步骤】：把工具列表绑定给大模型。
# 这相当于给大模型发了一本工具使用手册。之后大模型就知道自己不仅能说话，还能调用这些工具了。
llm_with_tools = llm.bind_tools(tools)

# ==========================================
# 5. 定义节点 (Nodes) - 流程图上的工作站
# ==========================================

# 节点1：大模型思考节点
def call_model(state: AgentState):
    """这个节点负责让大模型看着历史消息，决定下一步说什么，或者决定调用什么工具。"""
    
    # 1. 从状态里拿出所有的历史消息
    messages = state["messages"]
    
    # 2. 把消息喂给绑定了工具的大模型，得到回复
    response = llm_with_tools.invoke(messages)
    
    # 3. 返回新的状态。因为上面定义了 operator.add，这里返回的单条消息会自动追加到总列表里。
    return {"messages": [response]}


# 节点2：工具执行节点 (核心手动逻辑)
def execute_tools(state: AgentState):
    """这个节点专门负责执行代码。只有当大模型说"我要用工具"时，才会走到这里。"""
    
    # 1. 拿到最后一条消息 (大模型刚刚发出的指令)
    last_message = state["messages"][-1]
    
    # 我们准备一个空列表，用来存工具执行的结果
    tool_responses = []
    
    # 2. 检查大模型是不是发出了工具调用指令 (tool_calls)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 遍历所有的工具调用指令 (有时候大模型会一次性要求调用多个工具)
        for tool_call in last_message.tool_calls:
            # 拿到大模型想调用的工具名称，比如 "search_pdf_tool"
            action_name = tool_call["name"]
            # 拿到大模型给这个工具传入的参数，比如 {"query": "史玉坤经历"}
            action_args = tool_call["args"]
            
            # 我们做个简单的路由：如果是搜索工具，就执行搜索
            if action_name == "search_pdf_tool":
                # 【执行你的Python代码】：把参数传进去，拿到检索到的字符串
                result_str = search_pdf_tool.invoke(action_args)
                
                # 【非常重要】：把结果包装成 ToolMessage，大模型才能认出这是工具的返回结果
                # tool_call_id 是为了让大模型知道，这个结果对应它刚才发出的哪个指令
                tool_msg = ToolMessage(content=str(result_str), tool_call_id=tool_call["id"])
                
                # 存入列表
                tool_responses.append(tool_msg)
                
    # 3. 把工具返回的结果追加到状态池里
    return {"messages": tool_responses}


# ==========================================
# 6. 定义边与条件路由 (Edges & Routing)
# ==========================================

def should_continue(state: AgentState) -> str:
    """这是一个"红绿灯"逻辑。判断大模型是想结束对话，还是要调用工具。"""
    
    # 拿出最后一条消息
    last_message = state["messages"][-1]
    
    # 如果最后一条消息里包含 tool_calls，说明大模型想要用工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue" # 返回字符串 "continue"，告诉图走向工具节点
    
    # 如果没有，说明大模型直接给出了最终答案，没用工具
    return "end" # 返回 "end"，告诉图直接结束


# ==========================================
# 7. 拼接图 (Build the Graph) - 把所有的零件组装起来
# ==========================================
# 1. 创建图的骨架，告诉它我们要用 AgentState 作为状态传递规范
workflow = StateGraph(AgentState)

# 2. 把我们写的函数当作节点(Node)挂载上去
workflow.add_node("agent", call_model)      # 命名为 "agent" 节点
workflow.add_node("tools", execute_tools)   # 命名为 "tools" 节点

# 3. 画出明确的连接线 (Edges)
# 程序的起点(START) 永远是指向 agent 节点，让大模型先看一眼问题
workflow.add_edge(START, "agent")

# 4. 加上条件分支 (红绿灯)
# 当 agent 节点运行完后，去问 should_continue 函数
workflow.add_conditional_edges(
    "agent",           # 从哪个节点出发
    should_continue,   # 运行哪个判断函数
    {
        "continue": "tools", # 如果函数返回 "continue"，就走到 "tools" 节点
        "end": END           # 如果函数返回 "end"，就走到终点 (END) 结束程序
    }
)

# 5. 把工具节点和 agent 节点连成一个圈 (循环)
# 工具执行完后，必须把结果再交回给 agent，让大模型根据结果总结答案
workflow.add_edge("tools", "agent")

# 6. 编译成可执行的图应用
app = workflow.compile()

# ==========================================
# 8. 运行测试 (Run the App)
# ==========================================
if __name__ == "__main__":
    print("Agent 初始化完成，开始测试...")
    
    # 模拟用户输入的问题
    user_input = "请总结一下史玉坤在 Datatree Inc. 的工作经历时间段"
    
    # 构造初始状态字典，放入一条 HumanMessage
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    
    # 启动图的运行！ stream 方法可以让我们看到每一步的输出
    for event in app.stream(initial_state):
        # 打印当前走到哪个节点了
        for node_name, node_state in event.items():
            print(f"\n--- 当前刚刚运行完节点: {node_name} ---")
            
            # 打印这个节点产出的最后一条消息的内容
            last_msg = node_state["messages"][-1]
            if isinstance(last_msg, AIMessage):
                if last_msg.tool_calls:
                    print(f" 大模型决定调用工具: {last_msg.tool_calls}")
                else:
                    print(f" 大模型最终回答: {last_msg.content}")
            elif isinstance(last_msg, ToolMessage):
                 print(f" 工具执行结果返回了 {len(last_msg.content)} 个字符")