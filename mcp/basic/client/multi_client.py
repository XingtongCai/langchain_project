from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import os
import asyncio
import json
from typing import Dict, Any, List


def get_model():
    """
    创建并返回LLM模型实例
    """
    # 设置 API 密钥和基础 URL
    os.environ["OPENAI_API_KEY"] = "sk-"
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
    # 创建并返回模型
    return ChatOpenAI(model="deepseek-chat")


async def format_response(response: Dict[Any, Any]) -> None:
    """
    格式化并打印响应结果
    """
    try:
        print("\n===== 响应详情 =====")
        
        # 处理消息列表
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
                else:
                    # 处理字典形式的消息
                    if isinstance(message, dict):
                        if "role" in message and "content" in message:
                            print(f"[{message['role']} {i+1}] {message['content']}")
                        else:
                            print(f"[消息 {i+1}] {message}")
                    else:
                        print(f"[其他消息 {i+1}] {message}")
            
            # 提取最终AI回复
            final_messages = [msg for msg in messages if isinstance(msg, AIMessage) and msg.content]
            if final_messages:
                print("\n----- 最终回复 -----")
                print(final_messages[-1].content)
        else:
            print(f"未识别的响应格式: {type(response)}")
            print(response)
            
    except Exception as e:
        print(f"处理响应时出错: {str(e)}")
        print(f"响应类型: {type(response)}")
        # 直接打印响应，避免序列化
        print(response)


async def run_mcp_client():
    """
    运行MCP客户端，连接到多个服务器并执行查询
    """
    try:
        # 创建多服务器MCP客户端
        client = MultiServerMCPClient(
            {
                "math": {
                    "command": "python",
                    # 确保使用math_server.py文件的完整路径
                    "args": ["mcp/basic/server/math_server.py"],
                    "transport": "stdio",
                },
                "weather": {
                    # 确保在端口8000上启动weather服务器
                    "url": "http://127.0.0.1:8000/weather-mcp",
                    "transport": "streamable_http",
                },
            }
        )

        # 获取可用工具
        tools = await client.get_tools()
        print(f"可用工具: {[tool.name for tool in tools]}")

        # 创建模型和代理
        model = get_model()
        agent = create_react_agent(model, tools)

        # 执行数学查询
        print("\n执行数学查询: what's (3 + 5) x 12?")
        math_response = await agent.ainvoke({"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]})
        await format_response(math_response)

        # 执行天气查询
        print("\n执行天气查询: what is the weather in New York?")
        weather_response = await agent.ainvoke({"messages": [{"role": "user", "content": "what is the weather in New York?"}]})
        await format_response(weather_response)

        # 执行复合查询（同时使用数学和天气工具）
        print("\n执行复合查询: What is the weather in China and what would be (25 + 5) x 2?")
        complex_response = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What is the weather in China and what would be (25 + 5) x 2?"}]
        })
        await format_response(complex_response)

    except Exception as e:
        print(f"错误: {str(e)}")
    finally:
        # MultiServerMCPClient 没有 close 方法，所以这里不需要关闭连接
        print("\nMCP客户端任务完成")


# 主函数入口点
if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(run_mcp_client())
