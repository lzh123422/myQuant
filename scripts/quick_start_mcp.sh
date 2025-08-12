#!/bin/bash

echo "🚀 启动 Qlib MCP 服务器..."

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ Python未安装或不在PATH中"
    exit 1
fi

# 检查依赖
echo "📦 检查依赖..."
if ! python -c "import mcp" &> /dev/null; then
    echo "⚠️  MCP依赖未安装，正在安装..."
    pip install -r requirements-mcp.txt
fi

# 启动MCP服务器
echo "🔧 启动MCP服务器..."
python start_mcp_server.py 