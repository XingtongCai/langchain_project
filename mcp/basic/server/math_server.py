from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a:int,b:int)-> int:
    return a+b

@mcp.tool()
def sub(a:int,b:int)-> int:
    # 返回两个整数相减的结果
    return a-b

@mcp.tool()
def mul(a:int,b:int)-> int:
     return a*b

if __name__ == "__main__":
    mcp.run(transport="stdio")