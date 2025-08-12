#!/usr/bin/env python3
"""
运行Streamlit应用的主脚本
"""

import sys
import os
import subprocess

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def main():
    """主函数"""
    print("🚀 启动Streamlit量化分析应用...")
    
    # 构建Streamlit应用路径
    app_path = os.path.join(project_root, "src", "core", "streamlit_app.py")
    
    # 启动Streamlit
    cmd = [
        "streamlit", "run", app_path,
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
    except FileNotFoundError:
        print("❌ 未找到streamlit，请先安装: pip install streamlit")

if __name__ == "__main__":
    main() 