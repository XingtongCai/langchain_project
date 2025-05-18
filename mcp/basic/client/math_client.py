# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
import os
import traceback

# 创建服务器参数
server_params = StdioServerParameters(
    command="python",
    # 使用相对路径，确保路径正确
    args=["mcp/basic/server/math_server.py"],
)

def getModel():
    # 检查 DeepSeek API 密钥是否已设置
    os.environ["OPENAI_API_KEY"] = "sk-46e3a3bf58f545beaca174a0eab590a3"
    os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("警告: 未找到 OPENAI_API_KEY 环境变量。请确保已设置该环境变量。")
    else:
        print("已检测到 OPENAI_API_KEY 环境变量")
    model = ChatOpenAI(model="deepseek-chat")
    return model

def analyze_agent_response(agent_response):
    """
    分析代理响应，提取并显示调用的MCP工具和结果
    
    Args:
        agent_response: 代理的响应结果
    """
    messages = agent_response.get("messages", [])
    print("\n===== 详细分析 =====")
    
    # 查找工具调用
    tool_calls = []
    tool_results = []
    final_answer = None
    
    for i, message in enumerate(messages):
        # 检查是否是AI消息并包含工具调用
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print(f"\n步骤 {i+1}: AI决定使用以下工具:")
            for tool_call in message.tool_calls:
                tool_name = tool_call.get('name')
                tool_args = tool_call.get('args')
                print(f"  - 工具: {tool_name}")
                print(f"    参数: {tool_args}")
                tool_calls.append((tool_name, tool_args))
        
        # 检查是否是工具消息（包含工具执行结果）
        elif hasattr(message, 'name') and hasattr(message, 'content'):
            print(f"\n步骤 {i+1}: 工具 '{message.name}' 执行结果:")
            print(f"  - 结果: {message.content}")
            tool_results.append((message.name, message.content))
        
        # 检查是否是最终回答
        elif i == len(messages) - 1 and hasattr(message, 'content'):
            final_answer = message.content
    
    # 打印最终答案
    if final_answer:
        print("\n最终答案:")
        print(f"  {final_answer}")
    
    # 总结
    print("\n===== 总结 =====")
    print("调用的MCP工具:")
    for i, (tool_name, tool_args) in enumerate(tool_calls):
        print(f"  {i+1}. {tool_name}{tool_args} ")
    

async def main():
    try:
        model = getModel()
        print("模型初始化成功")
        
        print("连接到MCP服务器...")
        async with stdio_client(server_params) as (read, write):
            print("MCP服务器连接成功")
            
            try:
                async with ClientSession(read, write) as session:
                    # 初始化连接
                    print("初始化MCP会话...")
                    await session.initialize()
                    print("MCP会话初始化成功")

                    # 获取工具
                    print("加载MCP工具...")
                    try:
                        tools = await load_mcp_tools(session)
                        print(f"成功加载工具: {tools}")
                    except Exception as e:
                        print(f"加载工具时出错: {e}")
                        print("详细错误信息:")
                        traceback.print_exc()
                        return

                    # 创建并运行代理
                    print("创建代理...")
                    
                    try:
                        # 直接创建代理，不使用model_kwargs参数
                        agent = create_react_agent(model, tools)
                        print("代理创建成功，开始调用...")
                        # 使用正确的消息格式
                        agent_response = await agent.ainvoke({
                            "messages": [
                                {"role": "user", "content": "what's (3 + 5) x 12?"}
                            ]
                        })
                        # 分析代理响应
                        analyze_agent_response(agent_response)
                        
                        return agent_response
                    except Exception as e:
                        print(f"创建或调用代理时出错: {e}")
                        print("详细错误信息:")
                        traceback.print_exc()
            except Exception as e:
                print(f"MCP会话出错: {e}")
                print("详细错误信息:")
                traceback.print_exc()
    except Exception as e:
        print(f"发生错误: {e}")
        print("详细错误信息:")
        traceback.print_exc()

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())
