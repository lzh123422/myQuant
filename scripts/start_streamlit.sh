#!/bin/bash

echo "🚀 启动Streamlit Web应用..."

# 检查conda环境
if ! conda info --envs | grep -q "qlib-mcp"; then
    echo "❌ 请先激活qlib-mcp环境: conda activate qlib-mcp"
    exit 1
fi

# 激活环境
conda activate qlib-mcp

echo "📊 启动Streamlit应用..."
echo "💡 访问地址: http://localhost:8501"
echo "🔑 应用会自动在浏览器中打开"
echo "⏹️  按 Ctrl+C 停止服务"

# 启动Streamlit
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 