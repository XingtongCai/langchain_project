#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
翻译客户端，使用MCP服务器获取提示词并执行翻译任务。
"""

# 标准库导入
import asyncio
import os
import traceback

# 第三方库导入
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 本地模块导入
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 创建服务器参数
SERVER_PARAMS = StdioServerParameters(
    command="python",
    # 使用相对路径，确保路径正确
    args=["mcp/basic/server/prompt.py"],
)


def get_model():
    """
    初始化并返回语言模型。
    
    Returns:
        ChatOpenAI: 初始化好的语言模型实例
    """
    # 从环境变量获取API密钥，如果没有则使用默认值
    # 注意：在生产环境中应该避免硬编码API密钥
    os.environ.setdefault("OPENAI_API_KEY", "sk-")
    os.environ.setdefault("OPENAI_BASE_URL", "https://api.deepseek.com")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("警告: 未找到 OPENAI_API_KEY 环境变量。请确保已设置该环境变量。")
    else:
        print("已检测到 OPENAI_API_KEY 环境变量")
    
    return ChatOpenAI(model="deepseek-chat")


async def get_prompt(session, language):
    """
    从MCP服务器获取特定语言的翻译提示词。
    
    Args:
        session (ClientSession): MCP客户端会话
        language (str): 目标语言
        
    Returns:
        str: 获取到的提示词，如果出错则返回None
    """
    try:
        print(f"加载{language}翻译提示词...")
        prompt_msg = await session.get_prompt("翻译专家", arguments={"target_language": language})
        prompt = prompt_msg.messages[0].content.text
        print(f"成功加载提示词{prompt}")
        return prompt
    except Exception as e:
        print(f"加载提示词时出错: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return None


async def perform_translation(agent, prompt, text):
    """
    执行翻译任务。
    
    Args:
        agent: 执行翻译的代理
        prompt (str): 翻译提示词
        text (str): 要翻译的文本
        
    Returns:
        dict: 包含原始文本和翻译结果的字典
    """
    translation = await agent.ainvoke({
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}
        ]
    })
    
    return extract_translation(translation)


def extract_translation(translation_result):
    """
    从翻译结果中提取原始输入和最终的翻译文本。
    
    Args:
        translation_result (dict): 包含消息列表的翻译结果字典
        
    Returns:
        dict: 包含原始输入和翻译结果的字典
    """
    result = {
        "original": "",
        "translated": ""
    }
    
    # 检查结果是否包含messages键
    if 'messages' not in translation_result:
        result["translated"] = "无法提取翻译结果：结果格式不正确"
        return result
    
    # 获取消息列表
    messages = translation_result['messages']
    
    # 查找HumanMessage类型的消息（用户输入）
    for message in messages:
        if hasattr(message, 'content') and message.__class__.__name__ == 'HumanMessage':
            result["original"] = message.content
            break
    
    # 查找AIMessage类型的消息（翻译结果）
    for message in reversed(messages):
        if hasattr(message, 'content') and message.__class__.__name__ == 'AIMessage':
            result["translated"] = message.content
            break
    
    # 如果没有找到AIMessage
    if not result["translated"]:
        result["translated"] = "无法提取翻译结果：未找到AI回复"
    
    return result


async def main():
    """主函数，协调整个翻译流程。"""
    try:
        # 初始化模型
        model = get_model()
        print("模型初始化成功")
        
        print("连接到MCP服务器...")
        async with stdio_client(SERVER_PARAMS) as (read, write):
            print("MCP服务器连接成功")
            
            try:
                async with ClientSession(read, write) as session:
                    # 初始化连接
                    print("初始化MCP会话...")
                    await session.initialize()
                    print("MCP会话初始化成功")

                    # 创建代理
                    print("创建代理...")
                    agent = create_react_agent(model, [])
                    print("代理创建成功")

                    # 执行翻译任务
                    print("开始翻译任务...")
                    
                    # 翻译为日语
                    text_to_translate = "Hello, world!"
                    japanese_prompt = await get_prompt(session, 'Japanese')
                    if japanese_prompt:
                        japanese_result = await perform_translation(agent, japanese_prompt, text_to_translate)
                        print(f"原始文本: {japanese_result['original']}")
                        print(f"日语翻译: {japanese_result['translated']}")
                    
                    # 翻译为英语
                    english_prompt = await get_prompt(session, 'English')
                    if english_prompt:
                        english_result = await perform_translation(agent, english_prompt, text_to_translate)
                        print(f"原始文本: {english_result['original']}")
                        print(f"英语翻译: {english_result['translated']}")
                    
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
