#!/bin/bash

echo "🚀 启动Jupyter Notebook（无密码模式）..."

# 检查conda环境
if ! conda info --envs | grep -q "qlib-mcp"; then
    echo "❌ 请先激活qlib-mcp环境: conda activate qlib-mcp"
    exit 1
fi

# 激活环境
conda activate qlib-mcp

echo "📊 启动Jupyter Notebook..."
echo "💡 访问地址: http://localhost:8888"
echo "🔓 无需密码，直接访问"
echo "⏹️  按 Ctrl+C 停止服务"

# 启动Jupyter（无密码模式）
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' 