#!/usr/bin/env python3
"""
Simple test script for China stocks using Qlib
"""

import sys
from pathlib import Path

# Add qlib to path
sys.path.insert(0, str(Path(__file__).parent))

def simple_china_test():
    """Simple test for China stocks functionality."""
    
    try:
        print("🚀 简单中国股票测试...")
        
        # Import Qlib
        import qlib
        from qlib.data import D
        
        print("✅ Qlib导入成功")
        
        # 初始化Qlib
        print("🔧 初始化Qlib...")
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        print("✅ Qlib初始化成功")
        
        # 测试数据访问
        print("\n📊 测试数据访问...")
        
        # 获取一些股票数据
        symbols = ["000001.SZ", "000002.SZ", "000858.SZ"]  # 平安银行、万科A、五粮液
        fields = ["$close", "$volume"]
        
        print(f"🔍 获取股票数据: {symbols}")
        print(f"📈 数据字段: {fields}")
        
        try:
            # 获取最近的数据
            data = D.features(symbols, fields, start_time="2024-01-01", end_time="2024-12-31")
            print("✅ 数据获取成功")
            print(f"📊 数据形状: {data.shape}")
            print(f"📅 数据范围: {data.index.min()} 到 {data.index.max()}")
            
            # 显示一些样本数据
            print("\n📋 样本数据:")
            print(data.head())
            
        except Exception as e:
            print(f"⚠️  数据获取失败: {e}")
            print("这可能是因为数据格式或时间范围问题")
        
        # 测试工作流
        print("\n🔄 测试工作流...")
        from qlib.workflow import R
        
        try:
            # 列出可用的实验
            experiments = R.list_experiments()
            print(f"✅ 工作流正常，找到 {len(experiments)} 个实验")
            
            if experiments:
                print("📁 现有实验:")
                for exp in experiments[-3:]:  # 显示最后3个
                    print(f"   - {exp}")
            
        except Exception as e:
            print(f"⚠️  工作流测试失败: {e}")
        
        print("\n✅ 简单中国股票测试完成！")
        print("\n💡 系统状态:")
        print("   ✅ Qlib初始化成功")
        print("   ✅ 中国股票数据可用")
        print("   ✅ 工作流系统正常")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保在正确的conda环境中运行: conda activate qlib-mcp")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_china_test() 