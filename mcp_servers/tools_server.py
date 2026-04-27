# mcp_servers/tools_server.py
import asyncio
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# 1. 创建 MCP Server 实例
app = Server("my-enterprise-tools")


# 2. 注册工具清单（告诉外界我这有啥工具）
@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="get_server_time",
            description="获取企业内网服务器的当前精准时间",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        types.Tool(
            name="get_employee_info",
            description="查询企业员工信息（机密）",
            inputSchema={
                "type": "object",
                "properties": {
                    "emp_id": {
                        "type": "string",
                        "description": "员工工号，如 'E1001'"
                    }
                },
                "required": ["emp_id"]
            }
        )
    ]


# 3. 定义工具的具体执行逻辑
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name == "get_server_time":
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return [types.TextContent(type="text", text=f"服务器时间：{now}")]

    elif name == "get_employee_info":
        emp_id = arguments.get("emp_id")
        # 模拟查数据库
        db = {"E1001": "张三, 研发部", "E1002": "李四, 财务部"}
        info = db.get(emp_id, "查无此人")
        return [types.TextContent(type="text", text=f"员工信息：{info}")]

    else:
        return [types.TextContent(type="text", text=f"未知工具: {name}")]


# 4. 通过标准输入输出流（stdio）启动服务
async def main():
    # stdio_server 意味着它不需要占用网络端口，而是通过进程通信，极其轻量安全
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())