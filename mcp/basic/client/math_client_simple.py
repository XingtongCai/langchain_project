from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
import os

# 创建服务器参数
server_params = StdioServerParameters(
    command="python",
    args=["mcp/basic/server/math_server.py"],
)

def getModel():
    # 设置 API 密钥和基础 URL
    os.environ["OPENAI_API_KEY"] = "sk-46e3a3bf58f545beaca174a0eab590a3"
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
    # 创建并返回模型
    return ChatOpenAI(model="deepseek-chat")

async def main():
    # 初始化模型
    model = getModel()
    
    # 连接到MCP服务器
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化会话
            await session.initialize()
            
            # 加载MCP工具
            tools = await load_mcp_tools(session)
            
            # 创建代理
            agent = create_react_agent(model, tools)
            
            # 调用代理
            agent_response = await agent.ainvoke({
                "messages": [
                    {"role": "user", "content": "what's (3 + 5) x 12?"}
                ]
            })
            
            # 打印最终答案
            final_message = agent_response["messages"][-1]
            if hasattr(final_message, 'content'):
                print(f"最终答案: {final_message.content}")
            
            return agent_response

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())
