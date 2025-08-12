#!/usr/bin/env python3
"""
Debug script for Qlib MCP Server
"""

import asyncio
import sys
from pathlib import Path

# Add qlib to path
sys.path.insert(0, str(Path(__file__).parent))

async def debug_mcp_server():
    """Debug the MCP server structure."""
    
    try:
        from qlib.mcp.server import QlibMCPServer
        
        print("Creating MCP server...")
        server = QlibMCPServer()
        
        print("Server object type:", type(server.server))
        print("Server attributes:", dir(server.server))
        
        # Try to find the correct way to access tools
        if hasattr(server.server, 'tools'):
            print("Found tools attribute:", server.server.tools)
        elif hasattr(server.server, '_tools'):
            print("Found _tools attribute:", server.server._tools)
        else:
            print("No tools attribute found")
            
        # Check if there are any methods that might list tools
        methods = [attr for attr in dir(server.server) if callable(getattr(server.server, attr)) and 'tool' in attr.lower()]
        print("Methods with 'tool' in name:", methods)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_mcp_server()) 