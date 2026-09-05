import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import os
import sys

def search_clickhouse_memory(claim_type: str) -> str:
    """Search the historical database via MCP for how similar video generation flaws were successfully fixed."""
    async def _query():
        server_script = os.path.join(os.path.dirname(__file__), "..", "mcp_server.py")
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script],
            env=os.environ.copy()
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("search_remediation_history", arguments={"claim_type": claim_type})
                    return "\n".join([c.text for c in result.content if c.type == "text"])
        except Exception as e:
            return f"MCP Error: {e}"
            
    return asyncio.run(_query())

def get_axis_priority(scene_summary: str) -> str:
    """Query which creative dimensions historically need to be asked about first."""
    async def _query():
        server_script = os.path.join(os.path.dirname(__file__), "..", "mcp_server.py")
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script],
            env=os.environ.copy()
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("get_axis_priority", arguments={"scene_summary": scene_summary})
                    return "\n".join([c.text for c in result.content if c.type == "text"])
        except Exception as e:
            return f"MCP Error: {e}. Fall back to default order."
            
    return asyncio.run(_query())

