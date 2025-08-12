#!/usr/bin/env python3
"""
全自动模拟投资回测系统
使用训练好的机器学习模型进行历史数据回测
"""

import sys
from pathlib import Path
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

class AutoBacktestSystem:
    """全自动模拟投资回测系统"""
    
    def __init__(self):
        """初始化回测系统"""
        self.qlib_initialized = False
        self.model = None
        self.backtest_results = {}
        self.portfolio_values = []
        self.trade_records = []
        
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
    
    def load_trained_model(self, model_path=None):
        """加载训练好的模型"""
        try:
            print("🤖 加载训练好的模型...")
            
            # 这里我们重新训练一个模型，或者你可以加载之前保存的模型
            # 为了演示，我们重新训练一个简单的模型
            
            # 获取训练数据
            symbols = ["SH600000", "SH600004", "SH600009", "SH600010", "SH600011"]
            start_time = "2019-01-01"
            end_time = "2020-09-25"
            
            # 创建特征
            features = self.create_features(symbols, start_time, end_time)
            if features is None:
                return False
            
            # 训练模型
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            
            # 准备特征和标签
            feature_columns = [col for col in features.columns 
                             if col not in ['symbol', 'label', 'future_return']]
            
            X = features[feature_columns]
            y = features['label']
            
            # 训练模型
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
            print(f"✅ 模型加载成功!")
            print(f"   特征数量: {len(feature_columns)}")
            print(f"   训练样本: {len(X)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def create_features(self, symbols, start_time, end_time):
        """创建特征"""
        try:
            print(f"📊 创建特征...")
            
            # 获取基础数据
            fields = ["$close", "$volume", "$open", "$high", "$low", "$factor"]
            data = D.features(symbols, fields, start_time=start_time, end_time=end_time)
            
            if data.empty:
                print("❌ 获取的数据为空")
                return None
            
            # 创建特征DataFrame
            features_list = []
            
            for symbol in symbols:
                try:
                    symbol_data = data.loc[symbol]
                    
                    # 基础价格特征
                    close = symbol_data['$close']
                    volume = symbol_data['$volume']
                    open_price = symbol_data['$open']
                    high = symbol_data['$high']
                    low = symbol_data['$low']
                    factor = symbol_data['$factor']
                    
                    # 计算技术指标
                    features = pd.DataFrame(index=close.index)
                    
                    # 价格特征
                    features['close'] = close
                    features['volume'] = volume
                    features['open'] = open_price
                    features['high'] = high
                    features['low'] = low
                    features['factor'] = factor
                    
                    # 价格变化特征
                    features['price_change'] = close.pct_change()
                    features['price_change_2d'] = close.pct_change(2)
                    features['price_change_5d'] = close.pct_change(5)
                    
                    # 移动平均特征
                    features['ma_5'] = close.rolling(5).mean()
                    features['ma_10'] = close.rolling(10).mean()
                    features['ma_20'] = close.rolling(20).mean()
                    
                    # 价格位置特征
                    features['price_position'] = (close - low) / (high - low)
                    features['price_ma5_ratio'] = close / features['ma_5']
                    features['price_ma20_ratio'] = close / features['ma_20']
                    
                    # 成交量特征
                    features['volume_ma5'] = volume.rolling(5).mean()
                    features['volume_ratio'] = volume / features['volume_ma5']
                    
                    # 波动率特征
                    features['volatility_5d'] = close.rolling(5).std()
                    features['volatility_10d'] = close.rolling(10).std()
                    
                    # 添加股票标识
                    features['symbol'] = symbol
                    
                    features_list.append(features)
                    
                except Exception as e:
                    print(f"⚠️  处理股票 {symbol} 时出错: {e}")
                    continue
            
            if not features_list:
                print("❌ 没有成功创建任何特征")
                return None
            
            # 合并所有特征
            all_features = pd.concat(features_list, axis=0)
            
            # 清理数据
            all_features = all_features.dropna()
            
            # 创建标签（未来5天收益率）
            all_features = self.create_labels(all_features)
            
            print(f"✅ 特征创建成功，最终形状: {all_features.shape}")
            return all_features
            
        except Exception as e:
            print(f"❌ 特征创建失败: {e}")
            return None
    
    def create_labels(self, features_df, forward_days=5):
        """创建标签"""
        try:
            # 按股票分组计算未来收益率
            labels = []
            
            for symbol in features_df['symbol'].unique():
                symbol_data = features_df[features_df['symbol'] == symbol].copy()
                symbol_data = symbol_data.sort_index()
                
                # 计算未来收益率
                symbol_data['future_return'] = symbol_data['close'].shift(-forward_days) / symbol_data['close'] - 1
                
                # 创建分类标签（1: 上涨, 0: 下跌）
                symbol_data['label'] = (symbol_data['future_return'] > 0).astype(int)
                
                labels.append(symbol_data)
            
            # 合并所有标签
            labeled_data = pd.concat(labels, axis=0)
            
            # 清理数据
            labeled_data = labeled_data.dropna()
            
            return labeled_data
            
        except Exception as e:
            print(f"❌ 标签创建失败: {e}")
            return features_df
    
    def run_auto_backtest(self, symbols, start_time, end_time, initial_capital=1000000):
        """运行自动回测"""
        try:
            print(f"🔄 开始自动回测...")
            print(f"   股票池: {symbols}")
            print(f"   时间范围: {start_time} 到 {end_time}")
            print(f"   初始资金: {initial_capital:,} 元")
            
            # 获取回测数据
            fields = ["$close", "$volume", "$open", "$high", "$low", "$factor"]
            data = D.features(symbols, fields, start_time=start_time, end_time=end_time)
            
            if data.empty:
                print("❌ 回测数据为空")
                return False
            
            # 初始化回测变量
            current_capital = initial_capital
            portfolio_value = initial_capital
            positions = {symbol: 0 for symbol in symbols}
            portfolio_values = []
            trade_records = []
            
            # 获取所有交易日
            all_dates = sorted(data.index.get_level_values('datetime').unique())
            
            print(f"📅 回测期间: {len(all_dates)} 个交易日")
            
            # 按日期进行回测
            for i, current_date in enumerate(all_dates):
                if i < 20:  # 跳过前20天，等待特征计算
                    continue
                
                try:
                    # 获取当前日期的特征
                    current_features = self.get_daily_features(data, symbols, current_date)
                    if current_features is None:
                        continue
                    
                    # 生成投资信号
                    signals = self.generate_signals(current_features)
                    
                    # 执行交易
                    current_capital, positions, trades = self.execute_trades(
                        current_capital, positions, signals, data, current_date
                    )
                    
                    # 计算当前组合价值
                    portfolio_value = self.calculate_portfolio_value(
                        current_capital, positions, data, current_date
                    )
                    
                    # 记录结果
                    portfolio_values.append({
                        'date': current_date,
                        'portfolio_value': portfolio_value,
                        'cash': current_capital,
                        'positions': positions.copy()
                    })
                    
                    trade_records.extend(trades)
                    
                    # 显示进度
                    if i % 50 == 0:
                        print(f"   📊 进度: {i}/{len(all_dates)} 天, 组合价值: {portfolio_value:,.0f} 元")
                
                except Exception as e:
                    print(f"⚠️  处理日期 {current_date} 时出错: {e}")
                    continue
            
            # 保存回测结果
            self.backtest_results = {
                'portfolio_values': portfolio_values,
                'trade_records': trade_records,
                'initial_capital': initial_capital,
                'final_portfolio_value': portfolio_value,
                'total_return': (portfolio_value - initial_capital) / initial_capital,
                'symbols': symbols,
                'start_time': start_time,
                'end_time': end_time
            }
            
            print(f"✅ 自动回测完成!")
            print(f"📊 最终结果:")
            print(f"   初始资金: {initial_capital:,.0f} 元")
            print(f"   最终价值: {portfolio_value:,.0f} 元")
            print(f"   总收益率: {self.backtest_results['total_return']:.2%}")
            
            return True
            
        except Exception as e:
            print(f"❌ 自动回测失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_daily_features(self, data, symbols, current_date):
        """获取每日特征"""
        try:
            features_list = []
            
            for symbol in symbols:
                try:
                    # 获取历史数据（用于计算特征）
                    symbol_data = data.loc[symbol]
                    symbol_data = symbol_data[symbol_data.index <= current_date]
                    
                    if len(symbol_data) < 20:  # 需要足够的历史数据
                        continue
                    
                    # 计算特征（与训练时相同）
                    close = symbol_data['$close']
                    volume = symbol_data['$volume']
                    open_price = symbol_data['$open']
                    high = symbol_data['$high']
                    low = symbol_data['$low']
                    factor = symbol_data['$factor']
                    
                    # 最新特征
                    latest_features = pd.Series()
                    latest_features['close'] = close.iloc[-1]
                    latest_features['volume'] = volume.iloc[-1]
                    latest_features['open'] = open_price.iloc[-1]
                    latest_features['high'] = high.iloc[-1]
                    latest_features['low'] = low.iloc[-1]
                    latest_features['factor'] = factor.iloc[-1]
                    latest_features['price_change'] = close.pct_change().iloc[-1]
                    latest_features['price_change_2d'] = close.pct_change(2).iloc[-1]
                    latest_features['price_change_5d'] = close.pct_change(5).iloc[-1]
                    latest_features['ma_5'] = close.rolling(5).mean().iloc[-1]
                    latest_features['ma_10'] = close.rolling(10).mean().iloc[-1]
                    latest_features['ma_20'] = close.rolling(20).mean().iloc[-1]
                    latest_features['price_position'] = (close.iloc[-1] - low.iloc[-1]) / (high.iloc[-1] - low.iloc[-1])
                    latest_features['price_ma5_ratio'] = close.iloc[-1] / latest_features['ma_5']
                    latest_features['price_ma20_ratio'] = close.iloc[-1] / latest_features['ma_20']
                    latest_features['volume_ma5'] = volume.rolling(5).mean().iloc[-1]
                    latest_features['volume_ratio'] = volume.iloc[-1] / latest_features['volume_ma5']
                    latest_features['volatility_5d'] = close.rolling(5).std().iloc[-1]
                    latest_features['volatility_10d'] = close.rolling(10).std().iloc[-1]
                    latest_features['symbol'] = symbol
                    
                    features_list.append(latest_features)
                    
                except Exception as e:
                    continue
            
            if not features_list:
                return None
            
            # 合并特征
            daily_features = pd.DataFrame(features_list)
            return daily_features
            
        except Exception as e:
            return None
    
    def generate_signals(self, daily_features):
        """生成投资信号"""
        try:
            if self.model is None:
                return {}
            
            signals = {}
            
            for _, row in daily_features.iterrows():
                symbol = row['symbol']
                
                # 准备特征（排除非数值列）
                feature_columns = [col for col in daily_features.columns 
                                 if col not in ['symbol'] and pd.api.types.is_numeric_dtype(daily_features[col])]
                
                X = row[feature_columns].values.reshape(1, -1)
                
                # 预测
                prediction = self.model.predict(X)[0]
                probability = self.model.predict_proba(X)[0]
                
                # 生成信号
                if prediction == 1 and probability[1] > 0.6:  # 强烈买入信号
                    signals[symbol] = 'BUY'
                elif prediction == 0 and probability[0] > 0.6:  # 强烈卖出信号
                    signals[symbol] = 'SELL'
                else:
                    signals[symbol] = 'HOLD'
            
            return signals
            
        except Exception as e:
            print(f"❌ 信号生成失败: {e}")
            return {}
    
    def execute_trades(self, current_capital, positions, signals, data, current_date):
        """执行交易"""
        try:
            trades = []
            new_positions = positions.copy()
            new_capital = current_capital
            
            for symbol, signal in signals.items():
                try:
                    current_price = data.loc[symbol, '$close'].loc[current_date]
                    
                    if signal == 'BUY' and new_positions[symbol] == 0:
                        # 买入逻辑
                        shares_to_buy = int(new_capital * 0.1 / current_price)  # 使用10%资金
                        if shares_to_buy > 0:
                            cost = shares_to_buy * current_price
                            if cost <= new_capital:
                                new_positions[symbol] = shares_to_buy
                                new_capital -= cost
                                trades.append({
                                    'date': current_date,
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'shares': shares_to_buy,
                                    'price': current_price,
                                    'cost': cost
                                })
                    
                    elif signal == 'SELL' and new_positions[symbol] > 0:
                        # 卖出逻辑
                        shares_to_sell = new_positions[symbol]
                        revenue = shares_to_sell * current_price
                        new_positions[symbol] = 0
                        new_capital += revenue
                        trades.append({
                            'date': current_date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'shares': shares_to_sell,
                            'price': current_price,
                            'revenue': revenue
                        })
                
                except Exception as e:
                    continue
            
            return new_capital, new_positions, trades
            
        except Exception as e:
            return current_capital, positions, []
    
    def calculate_portfolio_value(self, cash, positions, data, current_date):
        """计算组合价值"""
        try:
            total_value = cash
            
            for symbol, shares in positions.items():
                if shares > 0:
                    try:
                        current_price = data.loc[symbol, '$close'].loc[current_date]
                        total_value += shares * current_price
                    except:
                        continue
            
            return total_value
            
        except Exception as e:
            return cash
    
    def plot_backtest_results(self):
        """绘制回测结果"""
        if not self.backtest_results:
            print("❌ 没有回测结果")
            return
        
        try:
            print("🎨 绘制回测结果...")
            
            # 准备数据
            portfolio_df = pd.DataFrame(self.backtest_results['portfolio_values'])
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df = portfolio_df.sort_values('date')
            
            # 计算收益率
            initial_capital = self.backtest_results['initial_capital']
            portfolio_df['return_rate'] = (portfolio_df['portfolio_value'] - initial_capital) / initial_capital
            
            # 创建图表
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('组合价值变化', '累计收益率', '交易记录'),
                vertical_spacing=0.1
            )
            
            # 组合价值
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name='组合价值',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # 累计收益率
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df['date'],
                    y=portfolio_df['return_rate'] * 100,
                    mode='lines',
                    name='累计收益率(%)',
                    line=dict(color='green', width=2)
                ),
                row=2, col=1
            )
            
            # 交易记录
            if self.backtest_results['trade_records']:
                trades_df = pd.DataFrame(self.backtest_results['trade_records'])
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                
                # 买入点
                buy_trades = trades_df[trades_df['action'] == 'BUY']
                if not buy_trades.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=buy_trades['date'],
                            y=buy_trades['price'],
                            mode='markers',
                            name='买入信号',
                            marker=dict(color='green', size=8, symbol='triangle-up')
                        ),
                        row=3, col=1
                    )
                
                # 卖出点
                sell_trades = trades_df[trades_df['action'] == 'SELL']
                if not sell_trades.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=sell_trades['date'],
                            y=sell_trades['price'],
                            mode='markers',
                            name='卖出信号',
                            marker=dict(color='red', size=8, symbol='triangle-down')
                        ),
                        row=3, col=1
                    )
            
            # 更新布局
            fig.update_layout(
                title='全自动模拟投资回测结果',
                height=900,
                showlegend=True
            )
            
            # 保存图表
            fig.write_html('auto_backtest_results.html')
            print("✅ 回测结果图表已保存为: auto_backtest_results.html")
            
            # 显示图表
            fig.show()
            
        except Exception as e:
            print(f"❌ 图表绘制失败: {e}")
    
    def save_backtest_report(self, filename=None):
        """保存回测报告"""
        if filename is None:
            filename = "auto_backtest_report.txt"
        
        try:
            print(f"📝 保存回测报告: {filename}")
            
            if not self.backtest_results:
                print("❌ 没有回测结果")
                return False
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("全自动模拟投资回测报告\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("📊 回测概览:\n")
                f.write(f"   股票池: {', '.join(self.backtest_results['symbols'])}\n")
                f.write(f"   回测期间: {self.backtest_results['start_time']} 到 {self.backtest_results['end_time']}\n")
                f.write(f"   初始资金: {self.backtest_results['initial_capital']:,.0f} 元\n")
                f.write(f"   最终价值: {self.backtest_results['final_portfolio_value']:,.0f} 元\n")
                f.write(f"   总收益率: {self.backtest_results['total_return']:.2%}\n\n")
                
                f.write("📈 交易统计:\n")
                f.write(f"   总交易次数: {len(self.backtest_results['trade_records'])}\n")
                
                if self.backtest_results['trade_records']:
                    trades_df = pd.DataFrame(self.backtest_results['trade_records'])
                    buy_trades = trades_df[trades_df['action'] == 'BUY']
                    sell_trades = trades_df[trades_df['action'] == 'SELL']
                    
                    f.write(f"   买入次数: {len(buy_trades)}\n")
                    f.write(f"   卖出次数: {len(sell_trades)}\n")
                
                f.write("\n🎯 策略特点:\n")
                f.write(f"   信号生成: 机器学习模型预测\n")
                f.write(f"   交易频率: 每日评估\n")
                f.write(f"   仓位管理: 单只股票最多10%资金\n")
                f.write(f"   风险控制: 基于模型置信度\n")
            
            print(f"✅ 回测报告已保存: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ 报告保存失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 全自动模拟投资回测系统启动")
    print("=" * 50)
    
    # 创建回测系统
    backtest_system = AutoBacktestSystem()
    
    # 初始化Qlib
    if not backtest_system.init_qlib():
        return
    
    # 加载训练好的模型
    if not backtest_system.load_trained_model():
        return
    
    # 定义回测参数
    symbols = ["SH600000", "SH600004", "SH600009", "SH600010", "SH600011"]
    start_time = "2020-01-01"  # 使用2020年数据进行回测
    end_time = "2020-09-25"
    initial_capital = 1000000  # 100万初始资金
    
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
    backtest_system.save_backtest_report()
    
    print("\n✅ 全自动模拟投资回测完成！")
    print("\n💡 查看结果:")
    print("1. 回测结果图表: auto_backtest_results.html")
    print("2. 回测报告: auto_backtest_report.txt")
    print("3. 浏览器中查看交互式图表")

if __name__ == "__main__":
    main() 