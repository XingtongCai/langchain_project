from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
import asyncio


async def main():
    # 创建多服务器MCP客户端
    client = MultiServerMCPClient(
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

    # 设置API密钥和基础URL
    os.environ["OPENAI_API_KEY"] = "sk-46e3a3bf58f545beaca174a0eab590a3"
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"  # 使用DeepSeek API
    
    # 创建模型
    model = ChatOpenAI(model="deepseek-chat")
    
    # 获取可用工具并创建代理
    tools = await client.get_tools()
    print(f"可用工具: {[tool.name for tool in tools]}")
    agent = create_react_agent(model, tools)

    # 执行数学查询
    print("\n执行数学查询: what's (3 + 5) x 12?")
    math_response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]
    })
    print_final_answer(math_response)

    # 执行天气查询
    print("\n执行天气查询: what is the weather in New York?")
    weather_response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "what is the weather in New York?"}]
    })
    print_final_answer(weather_response)


def print_final_answer(response):
    """打印最终回答"""
    if isinstance(response, dict) and "messages" in response:
        messages = response["messages"]
        # 找到最后一条AI消息
        for message in reversed(messages):
            if hasattr(message, "content") and message.content:
                print(f"回答: {message.content}")
                break


if __name__ == "__main__":
    asyncio.run(main())
