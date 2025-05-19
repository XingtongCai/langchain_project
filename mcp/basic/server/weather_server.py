from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """获取指定位置的天气信息"""
    if '中国' in location or 'China' in location:
        return "It's always sunny in China"
    else:
        return "It's always snowy in New York"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
