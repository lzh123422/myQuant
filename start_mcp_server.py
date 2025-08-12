#!/usr/bin/env python3
"""
Start script for Qlib MCP Server
"""

import asyncio
import sys
from pathlib import Path

# Add qlib to path
sys.path.insert(0, str(Path(__file__).parent))

from qlib.mcp.server import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMCP Server stopped by user")
    except Exception as e:
        print(f"Error starting MCP Server: {e}")
        sys.exit(1) 