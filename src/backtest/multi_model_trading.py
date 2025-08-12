#!/usr/bin/env python3
"""
多模型模拟交易系统
让所有训练好的模型同时进行模拟交易，对比表现
"""

import sys
import os
import qlib
from qlib.data import D
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

class MultiModelTrader:
    """多模型模拟交易系统"""
    
    def __init__(self):
        """初始化交易系统"""
        self.qlib_initialized = False
        self.models = {}
        self.trading_results = {}
        self.portfolio_values = {}
        self.trade_records = {}
        
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
    
    def load_trained_models(self):
        """加载训练好的模型"""
        print("🤖 加载训练好的模型...")
        
        try:
            # 导入增强模型训练器
            from src.models.enhanced_ml_training import EnhancedMLTrainer
            
            # 创建训练器并重新训练模型
            trainer = EnhancedMLTrainer()
            if not trainer.init_qlib():
                return False
            
            # 股票池和时间范围
            symbols = [
                "SH600000", "SH600004", "SH600009", "SH600010", "SH600011",
                "SH600016", "SH600019", "SH600025", "SH600027", "SH600028"
            ]
            start_time = "2019-01-01"
            end_time = "2020-09-25"
            
            # 创建增强特征
            features, labels = trainer.create_enhanced_features(symbols, start_time, end_time)
            if features is None:
                print("❌ 特征创建失败")
                return False
            
            # 训练增强模型
            if not trainer.train_enhanced_models(features, labels):
                print("❌ 模型训练失败")
                return False
            
            # 保存训练好的模型
            self.models = trainer.models
            print(f"✅ 成功加载 {len(self.models)} 个模型")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def create_trading_features(self, symbols, start_time, end_time):
        """创建交易特征"""
        print(f"📊 创建交易特征...")
        
        try:
            # 获取基础数据
            fields = ["$close", "$open", "$high", "$low", "$volume", "$factor"]
            data = D.features(symbols, fields, start_time=start_time, end_time=end_time)
            
            if data.empty:
                print("❌ 获取的数据为空")
                return None
            
            print(f"   ✅ 数据获取成功，数据形状: {data.shape}")
            
            all_features = []
            
            for symbol in symbols:
                try:
                    print(f"   处理股票: {symbol}")
                    
                    # 获取单个股票的数据
                    symbol_data = data.loc[symbol]
                    
                    if symbol_data.empty:
                        print(f"   ⚠️  {symbol} 数据为空，跳过")
                        continue
                    
                    # 创建特征DataFrame
                    df = pd.DataFrame(index=symbol_data.index)
                    df['close'] = symbol_data['$close']
                    df['open'] = symbol_data['$open']
                    df['high'] = symbol_data['$high']
                    df['low'] = symbol_data['$low']
                    df['volume'] = symbol_data['$volume']
                    df['factor'] = symbol_data['$factor']
                    df['symbol'] = symbol
                    
                    # 基础价格特征
                    df['price_change'] = df['close'].pct_change()
                    df['price_change_2d'] = df['close'].pct_change(2)
                    df['price_change_5d'] = df['close'].pct_change(5)
                    df['price_change_10d'] = df['close'].pct_change(10)
                    
                    # 移动平均线
                    df['ma_5'] = df['close'].rolling(5).mean()
                    df['ma_10'] = df['close'].rolling(10).mean()
                    df['ma_20'] = df['close'].rolling(20).mean()
                    df['ma_60'] = df['close'].rolling(60).mean()
                    
                    # 价格位置指标
                    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
                    df['price_ma5_ratio'] = df['close'] / df['ma_5']
                    df['price_ma20_ratio'] = df['close'] / df['ma_20']
                    df['price_ma60_ratio'] = df['close'] / df['ma_60']
                    
                    # 成交量特征
                    df['volume_ma5'] = df['volume'].rolling(5).mean()
                    df['volume_ma20'] = df['volume'].rolling(20).mean()
                    df['volume_ratio'] = df['volume'] / df['volume_ma20']
                    df['volume_change'] = df['volume'].pct_change()
                    
                    # 波动率特征
                    df['volatility_5d'] = df['price_change'].rolling(5).std()
                    df['volatility_10d'] = df['price_change'].rolling(10).std()
                    df['volatility_20d'] = df['price_change'].rolling(20).std()
                    
                    # 技术指标
                    df['rsi_14'] = self._calculate_rsi(df['close'], 14)
                    df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
                    df['bollinger_upper'], df['bollinger_lower'] = self._calculate_bollinger_bands(df['close'])
                    
                    # 趋势特征
                    df['trend_5d'] = np.where(df['ma_5'] > df['ma_20'], 1, -1)
                    df['trend_20d'] = np.where(df['ma_20'] > df['ma_60'], 1, -1)
                    df['momentum'] = df['close'] / df['close'].shift(20) - 1
                    
                    # 价格区间特征
                    df['price_range'] = (df['high'] - df['low']) / df['close']
                    df['price_range_5d'] = df['price_range'].rolling(5).mean()
                    
                    # 成交量价格关系
                    df['volume_price_trend'] = df['volume'] * df['price_change']
                    df['volume_price_trend_5d'] = df['volume_price_trend'].rolling(5).sum()
                    
                    # 删除包含NaN的行
                    df = df.dropna()
                    
                    if len(df) > 0:
                        all_features.append(df)
                        print(f"   ✅ {symbol} 特征创建成功，样本数: {len(df)}")
                    
                except Exception as e:
                    print(f"   ❌ 处理 {symbol} 时出错: {e}")
                    continue
            
            if not all_features:
                print("❌ 没有成功创建任何特征")
                return None
            
            # 合并所有特征
            combined_features = pd.concat(all_features, axis=0)
            combined_features = combined_features.sort_index()
            
            print(f"✅ 特征创建完成，最终形状: {combined_features.shape}")
            return combined_features
            
        except Exception as e:
            print(f"❌ 特征创建失败: {e}")
            return None
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """计算布林带"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower
    
    def run_multi_model_trading(self, symbols, start_time, end_time, initial_capital=1000000):
        """运行多模型模拟交易"""
        print(f"\n🚀 开始多模型模拟交易...")
        print(f"   股票池: {len(symbols)} 只股票")
        print(f"   交易期间: {start_time} 到 {end_time}")
        print(f"   初始资金: {initial_capital:,} 元")
        
        # 创建交易特征
        features_df = self.create_trading_features(symbols, start_time, end_time)
        if features_df is None:
            return False
        
        # 为每个模型运行模拟交易
        for model_name, model in self.models.items():
            print(f"\n📊 运行模型: {model_name}")
            
            try:
                # 运行单个模型交易
                result = self._run_single_model_trading(
                    model, features_df, symbols, start_time, end_time, initial_capital, model_name
                )
                
                if result:
                    self.trading_results[model_name] = result
                    print(f"   ✅ {model_name} 交易完成")
                else:
                    print(f"   ❌ {model_name} 交易失败")
                    
            except Exception as e:
                print(f"   ❌ {model_name} 交易出错: {e}")
                continue
        
        print(f"\n✅ 多模型交易完成！成功运行 {len(self.trading_results)} 个模型")
        return True
    
    def _run_single_model_trading(self, model, features_df, symbols, start_time, end_time, initial_capital, model_name):
        """运行单个模型交易"""
        try:
            # 准备特征数据
            feature_columns = [
                'close', 'open', 'high', 'low', 'volume', 'factor',
                'price_change', 'price_change_2d', 'price_change_5d', 'price_change_10d',
                'ma_5', 'ma_10', 'ma_20', 'ma_60',
                'price_position', 'price_ma5_ratio', 'price_ma20_ratio', 'price_ma60_ratio',
                'volume_ma5', 'volume_ma20', 'volume_ratio', 'volume_change',
                'volatility_5d', 'volatility_10d', 'volatility_20d',
                'rsi_14', 'macd', 'macd_signal',
                'bollinger_upper', 'bollinger_lower',
                'trend_5d', 'trend_20d', 'momentum',
                'price_range', 'price_range_5d',
                'volume_price_trend', 'volume_price_trend_5d'
            ]
            
            # 确保所有特征列都存在
            available_features = [col for col in feature_columns if col in features_df.columns]
            features = features_df[available_features].copy()
            
            # 填充缺失值
            features = features.fillna(0)
            
            # 按日期排序
            features = features.sort_index()
            
            # 初始化交易状态
            portfolio_value = initial_capital
            cash = initial_capital
            positions = {symbol: 0 for symbol in symbols}
            portfolio_values = []
            trade_records = []
            
            # 逐日交易
            dates = features.index.unique()
            
            for date in dates:
                try:
                    # 获取当日数据
                    daily_data = features.loc[date]
                    
                    # 按股票分组处理
                    for symbol in symbols:
                        symbol_data = daily_data[daily_data['symbol'] == symbol]
                        
                        if symbol_data.empty:
                            continue
                        
                        # 获取特征
                        symbol_features = symbol_data[available_features].iloc[0]
                        
                        # 模型预测
                        if hasattr(model, 'predict_proba'):
                            prediction = model.predict_proba([symbol_features])[0]
                            buy_probability = prediction[1]  # 上涨概率
                        else:
                            prediction = model.predict([symbol_features])[0]
                            buy_probability = prediction
                        
                        # 交易信号
                        buy_signal = buy_probability > 0.6  # 60%以上概率买入
                        sell_signal = buy_probability < 0.4  # 40%以下概率卖出
                        
                        current_price = symbol_data['close'].iloc[0]
                        
                        # 执行交易
                        if buy_signal and cash > 0 and positions[symbol] == 0:
                            # 买入
                            shares_to_buy = int(cash * 0.1 / current_price)  # 使用10%资金
                            if shares_to_buy > 0:
                                cost = shares_to_buy * current_price
                                cash -= cost
                                positions[symbol] = shares_to_buy
                                
                                trade_records.append({
                                    'date': date,
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'shares': shares_to_buy,
                                    'price': current_price,
                                    'value': cost,
                                    'model': model_name
                                })
                        
                        elif sell_signal and positions[symbol] > 0:
                            # 卖出
                            shares_to_sell = positions[symbol]
                            revenue = shares_to_sell * current_price
                            cash += revenue
                            positions[symbol] = 0
                            
                            trade_records.append({
                                'date': date,
                                'symbol': symbol,
                                'action': 'SELL',
                                'shares': shares_to_sell,
                                'price': current_price,
                                'value': revenue,
                                'model': model_name
                            })
                    
                    # 计算当日组合价值
                    portfolio_value = cash
                    for symbol, shares in positions.items():
                        if shares > 0:
                            symbol_data = daily_data[daily_data['symbol'] == symbol]
                            if not symbol_data.empty:
                                current_price = symbol_data['close'].iloc[0]
                                portfolio_value += shares * current_price
                    
                    portfolio_values.append({
                        'date': date,
                        'value': portfolio_value,
                        'model': model_name
                    })
                    
                except Exception as e:
                    print(f"   ⚠️  处理日期 {date} 时出错: {e}")
                    continue
            
            # 计算交易结果
            final_value = portfolio_values[-1]['value'] if portfolio_values else initial_capital
            total_return = (final_value - initial_capital) / initial_capital * 100
            
            # 统计交易次数
            buy_trades = len([t for t in trade_records if t['action'] == 'BUY'])
            sell_trades = len([t for t in trade_records if t['action'] == 'SELL'])
            total_trades = buy_trades + sell_trades
            
            result = {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'portfolio_values': portfolio_values,
                'trade_records': trade_records,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'total_trades': total_trades,
                'model_name': model_name
            }
            
            print(f"     最终价值: {final_value:,.0f} 元")
            print(f"     总收益率: {total_return:.2f}%")
            print(f"     交易次数: {total_trades} (买入: {buy_trades}, 卖出: {sell_trades})")
            
            return result
            
        except Exception as e:
            print(f"   ❌ 模型 {model_name} 交易失败: {e}")
            return None
    
    def plot_trading_comparison(self):
        """绘制交易结果对比"""
        if not self.trading_results:
            print("❌ 没有交易结果可供绘制")
            return
        
        print("\n🎨 绘制交易结果对比...")
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['组合价值变化对比', '收益率对比', '交易次数对比', '模型表现总结'],
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # 1. 组合价值变化对比
        for model_name, result in self.trading_results.items():
            portfolio_df = pd.DataFrame(result['portfolio_values'])
            if 'date' in portfolio_df.columns:
                portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
                
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_df['date'], 
                        y=portfolio_df['value'],
                        mode='lines',
                        name=f'{model_name}',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
            else:
                # 如果没有日期列，使用索引
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(portfolio_df))), 
                        y=portfolio_df['value'],
                        mode='lines',
                        name=f'{model_name}',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
        
        # 2. 收益率对比
        model_names = list(self.trading_results.keys())
        returns = [result['total_return'] for result in self.trading_results.values()]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=returns,
                name='总收益率 (%)',
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            ),
            row=1, col=2
        )
        
        # 3. 交易次数对比
        buy_trades = [result['buy_trades'] for result in self.trading_results.values()]
        sell_trades = [result['sell_trades'] for result in self.trading_results.values()]
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=buy_trades,
                name='买入次数',
                marker_color='#2ca02c'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=sell_trades,
                name='卖出次数',
                marker_color='#d62728'
            ),
            row=2, col=1
        )
        
        # 4. 模型表现总结表格
        summary_data = []
        for model_name, result in self.trading_results.items():
            summary_data.append([
                model_name,
                f"{result['total_return']:.2f}%",
                f"{result['total_trades']}",
                f"{result['buy_trades']}",
                f"{result['sell_trades']}",
                f"{result['final_value']:,.0f}"
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['模型', '收益率', '总交易', '买入', '卖出', '最终价值'],
                    font=dict(size=12),
                    align="left"
                ),
                cells=dict(
                    values=[[row[i] for row in summary_data] for i in range(6)],
                    font=dict(size=11),
                    align="left"
                )
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title='多模型模拟交易结果对比',
            height=800,
            showlegend=True
        )
        
        # 保存图表
        output_file = 'results/multi_model_trading_comparison.html'
        fig.write_html(output_file)
        print(f"✅ 交易对比图表已保存: {output_file}")
        
        return fig
    
    def save_trading_report(self, filename=None):
        """保存交易报告"""
        if not self.trading_results:
            print("❌ 没有交易结果可供保存")
            return
        
        if filename is None:
            filename = 'results/multi_model_trading_report.txt'
        
        print(f"\n📝 保存交易报告: {filename}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("多模型模拟交易报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 交易结果对比
            f.write("📊 交易结果对比:\n")
            f.write("-" * 30 + "\n")
            
            # 按收益率排序
            sorted_results = sorted(self.trading_results.items(), 
                                  key=lambda x: x[1]['total_return'], reverse=True)
            
            for i, (model_name, result) in enumerate(sorted_results):
                rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                f.write(f"{rank} {model_name}:\n")
                f.write(f"   总收益率: {result['total_return']:.2f}%\n")
                f.write(f"   最终价值: {result['final_value']:,.0f} 元\n")
                f.write(f"   交易次数: {result['total_trades']} (买入: {result['buy_trades']}, 卖出: {result['sell_trades']})\n")
                f.write("-" * 20 + "\n")
            
            # 最佳模型
            best_model = sorted_results[0][0]
            best_result = sorted_results[0][1]
            f.write(f"\n🏆 最佳表现模型: {best_model}\n")
            f.write(f"最佳收益率: {best_result['total_return']:.2f}%\n")
            f.write(f"最终价值: {best_result['final_value']:,.0f} 元\n\n")
            
            # 交易统计
            f.write("📈 交易统计:\n")
            f.write("-" * 30 + "\n")
            f.write(f"参与模型数量: {len(self.trading_results)}\n")
            f.write(f"总交易次数: {sum(r['total_trades'] for r in self.trading_results.values())}\n")
            f.write(f"平均收益率: {np.mean([r['total_return'] for r in self.trading_results.values()]):.2f}%\n")
            f.write(f"收益率标准差: {np.std([r['total_return'] for r in self.trading_results.values()]):.2f}%\n")
        
        print(f"✅ 交易报告已保存: {filename}")

def main():
    """主函数"""
    print("🚀 多模型模拟交易系统")
    print("=" * 50)
    
    # 创建交易系统
    trader = MultiModelTrader()
    
    # 初始化Qlib
    if not trader.init_qlib():
        return
    
    # 加载训练好的模型
    if not trader.load_trained_models():
        return
    
    # 股票池和时间范围
    symbols = [
        "SH600000", "SH600004", "SH600009", "SH600010", "SH600011",
        "SH600016", "SH600019", "SH600025", "SH600027", "SH600028"
    ]
    start_time = "2020-01-01"  # 使用2020年数据进行回测
    end_time = "2020-09-25"
    initial_capital = 1000000  # 100万初始资金
    
    # 运行多模型模拟交易
    if not trader.run_multi_model_trading(symbols, start_time, end_time, initial_capital):
        return
    
    # 绘制交易结果对比
    trader.plot_trading_comparison()
    
    # 保存交易报告
    trader.save_trading_report()
    
    print("\n🎉 多模型模拟交易完成！")
    print("💡 查看结果:")
    print("1. 交易对比图表: results/multi_model_trading_comparison.html")
    print("2. 交易报告: results/multi_model_trading_report.txt")

if __name__ == "__main__":
    main() 