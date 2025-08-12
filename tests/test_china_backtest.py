#!/usr/bin/env python3
"""
Test script for China stocks backtest using Qlib
"""

import sys
from pathlib import Path

# Add qlib to path
sys.path.insert(0, str(Path(__file__).parent))

def test_china_backtest():
    """Test China stocks backtest functionality."""
    
    try:
        print("🚀 测试中国股票回测...")
        
        # Import Qlib
        import qlib
        from qlib.config import C
        from qlib.contrib.strategy import TopkDropoutStrategy
        from qlib.workflow import R
        from qlib.workflow.record_temp import SignalRecord
        
        print("✅ Qlib导入成功")
        
        # 初始化Qlib
        print("🔧 初始化Qlib...")
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        print("✅ Qlib初始化成功")
        
        # 测试回测
        print("\n📈 测试回测功能...")
        
        # 创建策略
        print("🎯 创建TopkDropoutStrategy策略...")
        strategy = TopkDropoutStrategy(topk=50, n_drop=5)
        print("✅ 策略创建成功")
        
        # 设置回测配置
        print("⚙️  设置回测配置...")
        portfolio_config = {
            "benchmark": "000300.SH",  # 沪深300指数
            "account": 100000000,      # 1亿资金
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        }
        print("✅ 回测配置设置完成")
        
        # 运行回测
        print("🔄 运行回测...")
        with R.start(experiment_name="china_stocks_backtest"):
            sr = SignalRecord(model=strategy, dataset="", port_analysis_config=portfolio_config)
            sr.generate()
            print("✅ 回测完成")
        
        print("\n✅ 中国股票回测测试完成！")
        print("\n💡 回测结果已保存到:")
        print("   - 实验: china_stocks_backtest")
        print("   - 记录: signal")
        
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print("请确保在正确的conda环境中运行: conda activate qlib-mcp")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_china_backtest() 