#!/usr/bin/env python3
"""
量化策略训练脚本
包含策略开发、回测、模型训练等完整流程
"""

import sys
from pathlib import Path
import qlib
from qlib.config import C
from qlib.data import D
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.backtest import backtest, executor
from qlib.contrib.evaluate import risk_analysis
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class QuantStrategyTrainer:
    """量化策略训练器"""
    
    def __init__(self):
        """初始化训练器"""
        self.qlib_initialized = False
        self.strategies = {}
        self.backtest_results = {}
        
    def init_qlib(self):
        """初始化Qlib"""
        try:
            print("🔧 初始化Qlib...")
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
            self.qlib_initialized = True
            print("✅ Qlib初始化成功")
            return True
        except Exception as e:
            print(f"❌ Qlib初始化失败: {e}")
            return False
    
    def get_stock_data(self, symbols, start_time, end_time, fields=None):
        """获取股票数据"""
        if not self.qlib_initialized:
            print("❌ 请先初始化Qlib")
            return None
        
        if fields is None:
            fields = ["$close", "$volume", "$factor"]
        
        try:
            print(f"📊 获取股票数据: {symbols}")
            print(f"📅 时间范围: {start_time} 到 {end_time}")
            
            data = D.features(symbols, fields, start_time=start_time, end_time=end_time)
            print(f"✅ 数据获取成功，形状: {data.shape}")
            return data
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            return None
    
    def create_simple_strategy(self, name, topk=50, n_drop=5):
        """创建简单策略"""
        try:
            print(f"🎯 创建策略: {name}")
            
            # 创建TopkDropout策略
            strategy = TopkDropoutStrategy(topk=topk, n_drop=n_drop)
            
            # 检查策略是否可用
            print(f"   ✅ 策略 '{name}' 创建成功")
            print(f"   参数: topk={topk}, n_drop={n_drop}")
            print(f"   策略类型: {type(strategy).__name__}")
            
            # 测试策略方法
            available_methods = [method for method in dir(strategy) if not method.startswith('_')]
            print(f"   可用方法: {available_methods}")
            
            self.strategies[name] = strategy
            return strategy
        except Exception as e:
            print(f"❌ 策略创建失败: {e}")
            return None
    
    def run_backtest(self, strategy_name, symbols, start_time, end_time, 
                    account=100000000, benchmark="000300.SH"):
        """运行回测"""
        if strategy_name not in self.strategies:
            print(f"❌ 策略 '{strategy_name}' 不存在")
            return None
        
        try:
            print(f"🔄 运行回测: {strategy_name}")
            
            # 获取策略对象
            strategy = self.strategies[strategy_name]
            
            # 回测配置
            backtest_config = {
                "start_time": start_time,
                "end_time": end_time,
                "account": account,
                "benchmark": benchmark,
                "exchange_kwargs": {
                    "freq": "day",
                    "limit_threshold": 0.095,
                    "deal_price": "close",
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5,
                },
            }
            
            # 执行回测
            portfolio_metric_dict, indicator_dict = backtest(
                executor=executor.SimulatorExecutor(),
                strategy=strategy,
                **backtest_config
            )
            
            # 保存回测结果
            self.backtest_results[strategy_name] = {
                'portfolio_metrics': portfolio_metric_dict,
                'indicators': indicator_dict,
                'config': backtest_config
            }
            
            print(f"✅ 回测完成: {strategy_name}")
            return portfolio_metric_dict, indicator_dict
            
        except Exception as e:
            print(f"❌ 回测失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_backtest_results(self, strategy_name):
        """分析回测结果"""
        if strategy_name not in self.backtest_results:
            print(f"❌ 策略 '{strategy_name}' 没有回测结果")
            return None
        
        try:
            print(f"📊 分析回测结果: {strategy_name}")
            
            result = self.backtest_results[strategy_name]
            portfolio_metrics = result['portfolio_metric_dict']
            indicators = result['indicators']
            
            # 获取日频回测结果
            daily_metrics = portfolio_metrics.get('day')
            if daily_metrics is None:
                print("❌ 未找到日频回测结果")
                return None
            
            # 分析收益率
            returns = daily_metrics['return']
            cumulative_returns = (1 + returns).cumprod()
            
            # 计算风险指标
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # 计算最大回撤
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            
            print(f"📈 策略表现分析:")
            print(f"   年化收益率: {annual_return:.4f}")
            print(f"   年化波动率: {annual_volatility:.4f}")
            print(f"   夏普比率: {sharpe_ratio:.4f}")
            print(f"   最大回撤: {max_drawdown:.4f}")
            
            return {
                'returns': returns,
                'cumulative_returns': cumulative_returns,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            print(f"❌ 结果分析失败: {e}")
            return None
    
    def plot_backtest_results(self, strategy_name):
        """绘制回测结果图表"""
        analysis = self.analyze_backtest_results(strategy_name)
        if analysis is None:
            return
        
        try:
            print(f"🎨 绘制回测结果图表: {strategy_name}")
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('累计收益率', '回撤'),
                vertical_spacing=0.1
            )
            
            # 累计收益率
            cumulative_returns = analysis['cumulative_returns']
            fig.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns.values,
                    mode='lines',
                    name='策略收益',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # 回撤
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode='lines',
                    name='回撤',
                    line=dict(color='red', width=2),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            # 更新布局
            fig.update_layout(
                title=f'{strategy_name} 回测结果',
                height=600,
                showlegend=True
            )
            
            # 保存图表
            fig.write_html(f'{strategy_name}_backtest_results.html')
            print(f"✅ 图表已保存为: {strategy_name}_backtest_results.html")
            
            # 显示图表
            fig.show()
            
        except Exception as e:
            print(f"❌ 图表绘制失败: {e}")
    
    def save_strategy_report(self, strategy_name, filename=None):
        """保存策略报告"""
        if filename is None:
            filename = f"{strategy_name}_strategy_report.txt"
        
        try:
            print(f"📝 保存策略报告: {filename}")
            
            analysis = self.analyze_backtest_results(strategy_name)
            if analysis is None:
                return False
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"量化策略报告: {strategy_name}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("📊 策略表现:\n")
                f.write(f"   年化收益率: {analysis['annual_return']:.4f}\n")
                f.write(f"   年化波动率: {analysis['annual_volatility']:.4f}\n")
                f.write(f"   夏普比率: {analysis['sharpe_ratio']:.4f}\n")
                f.write(f"   最大回撤: {analysis['max_drawdown']:.4f}\n\n")
                
                f.write("🎯 策略参数:\n")
                if strategy_name in self.strategies:
                    strategy = self.strategies[strategy_name]
                    f.write(f"   策略类型: {type(strategy).__name__}\n")
                    f.write(f"   参数: {strategy.__dict__}\n\n")
                
                f.write("📅 回测时间:\n")
                if strategy_name in self.backtest_results:
                    config = self.backtest_results[strategy_name]['config']
                    f.write(f"   开始时间: {config['start_time']}\n")
                    f.write(f"   结束时间: {config['end_time']}\n")
                    f.write(f"   初始资金: {config['account']:,}\n")
                    f.write(f"   基准指数: {config['benchmark']}\n")
            
            print(f"✅ 策略报告已保存: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ 报告保存失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 量化策略训练系统启动")
    print("=" * 50)
    
    # 创建训练器
    trainer = QuantStrategyTrainer()
    
    # 初始化Qlib
    if not trainer.init_qlib():
        return
    
    # 定义股票池
    symbols = ["SH600000", "SH600036", "SH600016", "SH600519", "SH600887"]
    start_time = "2020-01-01"
    end_time = "2020-09-25"
    
    print(f"\n📊 股票池: {symbols}")
    print(f"📅 训练时间: {start_time} 到 {end_time}")
    
    # 创建策略
    print("\n🎯 创建量化策略...")
    trainer.create_simple_strategy("基础策略", topk=3, n_drop=1)
    
    # 运行回测
    print("\n🔄 运行策略回测...")
    trainer.run_backtest("基础策略", symbols, start_time, end_time)
    
    # 分析结果
    print("\n📊 分析回测结果...")
    trainer.analyze_backtest_results("基础策略")
    
    # 绘制图表
    print("\n🎨 绘制回测结果...")
    trainer.plot_backtest_results("基础策略")
    
    # 保存报告
    print("\n📝 保存策略报告...")
    trainer.save_strategy_report("基础策略")
    
    print("\n✅ 量化策略训练完成！")
    print("\n💡 下一步:")
    print("1. 查看生成的HTML图表")
    print("2. 阅读策略报告")
    print("3. 调整策略参数重新训练")
    print("4. 尝试不同的策略类型")

if __name__ == "__main__":
    main() 