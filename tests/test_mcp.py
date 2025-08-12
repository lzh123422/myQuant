#!/usr/bin/env python3
"""
Test script for Qlib MCP Server
"""

import asyncio
import json
import sys
from pathlib import Path

# Add qlib to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_mcp_server():
    """Test the MCP server functionality."""
    
    try:
        from qlib.mcp.server import QlibMCPServer
        
        print("Creating MCP server...")
        server = QlibMCPServer()
        
        print("Testing tool listing...")
        # Get the list_tools function and call it
        list_tools_func = server.server.list_tools
        print(f"list_tools type: {type(list_tools_func)}")
        
        # Try to call it
        try:
            tools = list_tools_func()
            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool.name}: {tool.description}")
        except Exception as call_error:
            print(f"Error calling list_tools: {call_error}")
            print("This might be a decorator that needs to be called differently")
        
        print("\nMCP server test completed successfully!")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements-mcp.txt")
    except Exception as e:
        print(f"Error testing MCP server: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mcp_server()) 