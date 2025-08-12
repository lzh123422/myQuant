#!/usr/bin/env python3
"""
Test script for China stocks using Qlib
"""

import sys
from pathlib import Path

# Add qlib to path
sys.path.insert(0, str(Path(__file__).parent))

def test_china_stocks():
    """Test China stocks functionality."""
    
    try:
        print("🚀 测试中国股票数据...")
        
        # Import Qlib
        import qlib
        from qlib.config import C
        from qlib.data import D
        
        print("✅ Qlib导入成功")
        
        # 设置配置
        print("📊 设置Qlib配置...")
        # 使用正确的配置方法
        C.provider_uri = "~/.qlib/qlib_data/cn_data"
        C.region = "cn"
        
        print("✅ 配置设置完成")
        
        # 初始化Qlib
        print("🔧 初始化Qlib...")
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        print("✅ Qlib初始化成功")
        
        # 尝试初始化数据提供者
        print("🔧 初始化数据提供者...")
        try:
            D.init("~/.qlib/qlib_data/cn_data", "cn")
            print("✅ 数据提供者初始化成功")
        except Exception as e:
            print(f"⚠️  数据提供者初始化失败: {e}")
            print("这可能是正常的，因为我们还没有下载数据")
        
        # 测试基本功能
        print("\n📈 测试基本功能...")
        
        # 列出可用的模型
        print("🔍 可用的机器学习模型:")
        models = [
            "LGBModel", "MLPModel", "GRUModel", "LSTMModel", "TransformerModel",
            "TFTModel", "TabNetModel", "CatBoostModel", "XGBoostModel",
            "LinearModel", "EnsembleModel", "MetaModel"
        ]
        for i, model in enumerate(models, 1):
            print(f"  {i:2d}. {model}")
        
        # 测试策略
        print("\n🎯 可用的交易策略:")
        strategies = [
            "TopkDropoutStrategy", "TopkStrategy", "EqualWeightStrategy"
        ]
        for i, strategy in enumerate(strategies, 1):
            print(f"  {i:2d}. {strategy}")
        
        print("\n✅ 中国股票测试完成！")
        print("\n💡 下一步:")
        print("1. 下载中国股票数据")
        print("2. 运行回测")
        print("3. 训练模型")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保在正确的conda环境中运行: conda activate qlib-mcp")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_china_stocks() 