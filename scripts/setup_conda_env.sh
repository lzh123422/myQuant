#!/bin/bash

echo "🚀 设置 Qlib MCP Conda 环境..."

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ Conda未安装，请先安装Anaconda或Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 检查环境是否已存在
if conda env list | grep -q "qlib-mcp"; then
    echo "⚠️  环境 'qlib-mcp' 已存在"
    read -p "是否要重新创建环境？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  删除现有环境..."
        conda env remove -n qlib-mcp -y
    else
        echo "✅ 使用现有环境"
        echo "请运行: conda activate qlib-mcp"
        exit 0
    fi
fi

echo "📦 创建新的conda环境..."
conda env create -f environment-mcp.yml

if [ $? -eq 0 ]; then
    echo "✅ Conda环境创建成功！"
    echo ""
    echo "🎯 下一步操作："
    echo "1. 激活环境: conda activate qlib-mcp"
    echo "2. 安装Qlib: pip install -e ."
    echo "3. 测试MCP: python test_mcp.py"
    echo "4. 启动服务器: python start_mcp_server.py"
    echo ""
    echo "💡 提示: 使用 'conda activate qlib-mcp' 来激活环境"
else
    echo "❌ Conda环境创建失败"
    exit 1
fi 