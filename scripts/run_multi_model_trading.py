#!/usr/bin/env python3
"""
运行多模型模拟交易的主脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.backtest.multi_model_trading import MultiModelTrader

def main():
    """主函数"""
    print("🚀 myQuant 多模型模拟交易系统")
    print("=" * 50)
    
    # 创建交易系统
    trader = MultiModelTrader()
    
    # 初始化Qlib
    if not trader.init_qlib():
        print("❌ Qlib初始化失败，请检查环境配置")
        return
    
    # 加载训练好的模型
    if not trader.load_trained_models():
        print("❌ 模型加载失败")
        return
    
    # 股票池和时间范围
    symbols = [
        "SH600000", "SH600004", "SH600009", "SH600010", "SH600011",
        "SH600016", "SH600019", "SH600025", "SH600027", "SH600028"
    ]
    start_time = "2020-01-01"  # 使用2020年数据进行回测
    end_time = "2020-09-25"
    initial_capital = 1000000  # 100万初始资金
    
    print(f"\n📊 交易配置:")
    print(f"   股票池: {len(symbols)} 只股票")
    print(f"   回测期间: {start_time} 到 {end_time}")
    print(f"   初始资金: {initial_capital:,} 元")
    print(f"   参与模型: {len(trader.models)} 个")
    
    # 运行多模型模拟交易
    print(f"\n🚀 开始多模型模拟交易...")
    if not trader.run_multi_model_trading(symbols, start_time, end_time, initial_capital):
        print("❌ 模拟交易失败")
        return
    
    # 绘制交易结果对比
    print(f"\n🎨 生成交易结果对比图表...")
    trader.plot_trading_comparison()
    
    # 保存交易报告
    print(f"\n📝 保存交易报告...")
    trader.save_trading_report()
    
    # 显示最佳模型
    if trader.trading_results:
        best_model = max(trader.trading_results.keys(), 
                        key=lambda x: trader.trading_results[x]['total_return'])
        best_result = trader.trading_results[best_model]
        
        print(f"\n🏆 交易结果总结:")
        print(f"   最佳模型: {best_model}")
        print(f"   最佳收益率: {best_result['total_return']:.2f}%")
        print(f"   最终价值: {best_result['final_value']:,.0f} 元")
        print(f"   交易次数: {best_result['total_trades']}")
    
    print(f"\n💡 查看详细结果:")
    print("1. 交易对比图表: results/multi_model_trading_comparison.html")
    print("2. 交易报告: results/multi_model_trading_report.txt")
    
    print(f"\n🎉 多模型模拟交易完成！")

if __name__ == "__main__":
    main() 