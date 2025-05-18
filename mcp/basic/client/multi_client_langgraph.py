"""
LangGraph 与 MCP 多服务器客户端集成示例

本示例展示了如何使用 LangGraph 的状态图与 MCP 多服务器客户端集成，
实现对多个 MCP 服务器提供的工具的调用。
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import os
import asyncio
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_model():
    """创建并返回LLM模型实例"""
    # 设置 API 密钥和基础 URL
    os.environ["OPENAI_API_KEY"] = "sk-46e3a3bf58f545beaca174a0eab590a3"
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
    
    # 创建并返回模型
    return ChatOpenAI(model="deepseek-chat")


async def format_response(response):
    """格式化并打印响应结果"""
    print("\n===== 响应详情 =====")
    
    if isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
        
        # 打印消息流程
        print("\n----- 消息流程 -----")
        for i, message in enumerate(messages):
            if isinstance(message, HumanMessage):
                print(f"[用户问题 {i+1}] {message.content}")
            elif isinstance(message, AIMessage):
                # 检查是否有工具调用
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    print(f"[AI调用工具 {i+1}] 调用了 {len(message.tool_calls)} 个工具")
                    for tool_call in message.tool_calls:
                        print(f"  - 工具: {tool_call['name']}, 参数: {tool_call['args']}")
                # 检查内容
                if message.content:
                    print(f"[AI回复 {i+1}] {message.content}")
            elif isinstance(message, ToolMessage):
                print(f"[工具结果 {i+1}] {message.name}: {message.content}")
        
        # 提取最终AI回复
        final_messages = [msg for msg in messages if isinstance(msg, AIMessage) and msg.content]
        if final_messages:
            print("\n----- 最终回复 -----")
            print(final_messages[-1].content)


def create_mcp_client():
    """创建并返回MCP多服务器客户端"""
    return MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mcp/basic/server/math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
        }
    )


def create_langgraph(model, tools):
    """创建并返回LangGraph状态图"""
    # 定义调用模型的函数
    def call_model(state: MessagesState):
        """处理状态并调用模型"""
        response = model.bind_tools(tools).invoke(state["messages"])
        return {"messages": state["messages"] + [response]}
    
    # 构建状态图
    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools",ToolNode(tools))
    
    # 添加边和条件边
    builder.add_edge(START, "call_model")
    
    # 添加条件边
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")
    
    # 编译图
    return builder.compile()


async def main():
    """主函数，运行LangGraph与MCP客户端集成示例"""
    try:
        # 获取模型
        model = get_model()
        
        # 创建MCP客户端
        client = create_mcp_client()
        
        # 获取可用工具
        tools = await client.get_tools()
        if not tools:
            print("未获取到任何工具，请确保MCP服务器正在运行")
            return
        
        print(f"可用工具: {[tool.name for tool in tools]}")
        
        # 创建LangGraph状态图
        graph = create_langgraph(model, tools)
        
        # 执行数学查询
        print("\n执行数学查询: what's (3 + 5) x 12?")
        math_response = await graph.ainvoke({"messages": [HumanMessage(content="what's (3 + 5) x 12?")]})
        await format_response(math_response)
        
        # 执行天气查询
        print("\n执行天气查询: what is the weather in New York?")
        weather_response = await graph.ainvoke({"messages": [HumanMessage(content="what is the weather in New York?")]})
        await format_response(weather_response)
        
        # 执行复合查询（同时使用数学和天气工具）
        print("\n执行复合查询: What is the weather in China and what would be (25 + 5) x 2?")
        complex_response = await graph.ainvoke({
            "messages": [HumanMessage(content="What is the weather in China and what would be (25 + 5) x 2?")]
        })
        await format_response(complex_response)
        
    except Exception as e:
        print(f"错误: {str(e)}")


# 主程序入口点
if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
