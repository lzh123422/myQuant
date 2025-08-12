#!/usr/bin/env python3
"""
运行回测的主脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.backtest.auto_backtest_system import AutoBacktestSystem
from config.project_config import *

def main():
    """主函数"""
    print(f"🚀 {PROJECT_NAME} v{VERSION} - 全自动回测系统")
    print("=" * 50)
    
    # 创建回测系统
    backtest_system = AutoBacktestSystem()
    
    # 初始化Qlib
    if not backtest_system.init_qlib():
        return
    
    # 加载训练好的模型
    if not backtest_system.load_trained_model():
        return
    
    # 运行回测
    symbols = DATA_CONFIG["default_symbols"][:5]  # 取前5只股票
    start_time = "2020-01-01"
    end_time = "2020-09-25"
    initial_capital = BACKTEST_CONFIG["default_initial_capital"]
    
    print(f"\n📊 回测配置:")
    print(f"   股票池: {symbols}")
    print(f"   回测期间: {start_time} 到 {end_time}")
    print(f"   初始资金: {initial_capital:,} 元")
    
    # 运行自动回测
    if not backtest_system.run_auto_backtest(symbols, start_time, end_time, initial_capital):
        return
    
    # 绘制回测结果
    print("\n🎨 绘制回测结果...")
    backtest_system.plot_backtest_results()
    
    # 保存回测报告
    print("\n📝 保存回测报告...")
    backtest_system.save_backtest_report("results/backtest_report.txt")
    
    print("\n✅ 回测完成！")
    print("💡 查看结果:")
    print("1. 回测结果图表: results/auto_backtest_results.html")
    print("2. 回测报告: results/backtest_report.txt")

if __name__ == "__main__":
    main() 