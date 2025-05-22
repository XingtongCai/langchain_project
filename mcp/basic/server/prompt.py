from fastmcp import FastMCP

mcp = FastMCP('prompt_and_resources')

@mcp.prompt('翻译专家')
def translate_expert(target_language) -> str:
    return f'你是一个翻译专家，擅长将任何语言翻译成{target_language}。请翻译以下内容：'


if __name__ == '__main__':
    mcp.run(transport='stdio')
