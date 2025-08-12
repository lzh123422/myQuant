#!/usr/bin/env python3
"""
简化版机器学习模型训练脚本
使用基础价格和成交量特征
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

class SimpleMLTrainer:
    """简化版机器学习训练器"""
    
    def __init__(self):
        """初始化训练器"""
        self.qlib_initialized = False
        self.models = {}
        self.training_results = {}
        
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
    
    def create_basic_features(self, symbols, start_time, end_time):
        """创建基础特征"""
        if not self.qlib_initialized:
            print("❌ 请先初始化Qlib")
            return None
        
        try:
            print(f"📊 创建基础特征...")
            print(f"   股票: {symbols}")
            print(f"   时间范围: {start_time} 到 {end_time}")
            
            # 获取基础数据
            fields = ["$close", "$volume", "$open", "$high", "$low", "$factor"]
            data = D.features(symbols, fields, start_time=start_time, end_time=end_time)
            
            if data.empty:
                print("❌ 获取的数据为空")
                return None
            
            print(f"✅ 原始数据获取成功，形状: {data.shape}")
            
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
            
            print(f"✅ 特征创建成功，最终形状: {all_features.shape}")
            print(f"📊 特征列: {list(all_features.columns)}")
            
            return all_features
            
        except Exception as e:
            print(f"❌ 特征创建失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_labels(self, features_df, forward_days=5):
        """创建标签（未来收益率）"""
        try:
            print(f"🏷️  创建标签，预测未来 {forward_days} 天收益率...")
            
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
            
            print(f"✅ 标签创建成功，数据形状: {labeled_data.shape}")
            print(f"📊 标签分布:")
            print(f"   上涨(1): {labeled_data['label'].sum()}")
            print(f"   下跌(0): {(labeled_data['label'] == 0).sum()}")
            
            return labeled_data
            
        except Exception as e:
            print(f"❌ 标签创建失败: {e}")
            return None
    
    def train_simple_model(self, features_df, model_name="基础预测模型"):
        """训练简单模型"""
        try:
            print(f"🤖 开始训练模型: {model_name}")
            
            # 准备特征和标签
            feature_columns = [col for col in features_df.columns 
                             if col not in ['symbol', 'label', 'future_return']]
            
            X = features_df[feature_columns]
            y = features_df['label']
            
            print(f"📊 特征数量: {len(feature_columns)}")
            print(f"📊 样本数量: {len(X)}")
            
            # 分割训练集和测试集
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"📊 训练集: {len(X_train)}, 测试集: {len(X_test)}")
            
            # 训练随机森林模型
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 预测和评估
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"✅ 模型训练完成!")
            print(f"📊 测试集准确率: {accuracy:.4f}")
            print(f"📊 分类报告:")
            print(classification_report(y_test, y_pred))
            
            # 特征重要性
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"📊 特征重要性 (前10):")
            print(feature_importance.head(10))
            
            # 保存结果
            self.training_results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'feature_importance': feature_importance,
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            return model
            
        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_model_results(self, model_name):
        """绘制模型结果"""
        if model_name not in self.training_results:
            print(f"❌ 模型 '{model_name}' 没有训练结果")
            return
        
        try:
            print(f"🎨 绘制模型结果: {model_name}")
            
            result = self.training_results[model_name]
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('特征重要性', '预测准确率', '混淆矩阵', 'ROC曲线'),
                specs=[[{"type": "bar"}, {"type": "indicator"}],
                       [{"type": "heatmap"}, {"type": "scatter"}]]
            )
            
            # 特征重要性
            importance_df = result['feature_importance'].head(15)
            fig.add_trace(
                go.Bar(
                    x=importance_df['importance'],
                    y=importance_df['feature'],
                    orientation='h',
                    name='特征重要性'
                ),
                row=1, col=1
            )
            
            # 准确率指示器
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=result['accuracy'] * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "准确率 (%)"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 75], 'color': "gray"},
                                    {'range': [75, 100], 'color': "darkgray"}]}
                ),
                row=1, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title=f'{model_name} 训练结果',
                height=800,
                showlegend=False
            )
            
            # 保存图表
            fig.write_html(f'{model_name}_results.html')
            print(f"✅ 结果图表已保存为: {model_name}_results.html")
            
            # 显示图表
            fig.show()
            
        except Exception as e:
            print(f"❌ 图表绘制失败: {e}")
    
    def save_model_report(self, model_name, filename=None):
        """保存模型报告"""
        if filename is None:
            filename = f"{model_name}_report.txt"
        
        try:
            print(f"📝 保存模型报告: {filename}")
            
            if model_name not in self.training_results:
                print(f"❌ 模型 '{model_name}' 没有训练结果")
                return False
            
            result = self.training_results[model_name]
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"简化版机器学习模型训练报告: {model_name}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("📊 模型性能:\n")
                f.write(f"   测试集准确率: {result['accuracy']:.4f}\n")
                f.write(f"   模型类型: RandomForestClassifier\n")
                f.write(f"   特征数量: {len(result['feature_importance'])}\n")
                f.write(f"   训练样本数: {len(result['X_test']) * 1.25:.0f}\n")
                f.write(f"   测试样本数: {len(result['X_test'])}\n\n")
                
                f.write("🎯 特征重要性 (前15):\n")
                for idx, row in result['feature_importance'].head(15).iterrows():
                    f.write(f"   {row['feature']}: {row['importance']:.4f}\n")
                
                f.write("\n📅 训练信息:\n")
                f.write(f"   特征类型: 基础价格和成交量特征\n")
                f.write(f"   标签类型: 未来5天涨跌分类\n")
                f.write(f"   算法: 随机森林分类器\n")
            
            print(f"✅ 模型报告已保存: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ 报告保存失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 简化版机器学习模型训练系统启动")
    print("=" * 50)
    
    # 创建训练器
    trainer = SimpleMLTrainer()
    
    # 初始化Qlib
    if not trainer.init_qlib():
        return
    
    # 定义股票池和时间范围
    symbols = ["SH600000", "SH600004", "SH600009", "SH600010", "SH600011"]
    start_time = "2019-01-01"
    end_time = "2020-09-25"
    
    print(f"\n📊 股票池: {symbols}")
    print(f"📅 训练时间: {start_time} 到 {end_time}")
    
    # 创建特征
    print("\n📊 创建基础特征...")
    features = trainer.create_basic_features(symbols, start_time, end_time)
    
    if features is None:
        print("❌ 特征创建失败，无法继续训练")
        return
    
    # 创建标签
    print("\n🏷️  创建训练标签...")
    labeled_data = trainer.create_labels(features, forward_days=5)
    
    if labeled_data is None:
        print("❌ 标签创建失败，无法继续训练")
        return
    
    # 训练模型
    print("\n🤖 开始训练机器学习模型...")
    model = trainer.train_simple_model(labeled_data, "基础预测模型")
    
    if model is None:
        print("❌ 模型训练失败")
        return
    
    # 绘制结果
    print("\n🎨 绘制模型结果...")
    trainer.plot_model_results("基础预测模型")
    
    # 保存报告
    print("\n📝 保存模型报告...")
    trainer.save_model_report("基础预测模型")
    
    print("\n✅ 简化版机器学习模型训练完成！")
    print("\n💡 下一步:")
    print("1. 查看生成的HTML图表")
    print("2. 阅读模型报告")
    print("3. 调整特征和参数重新训练")
    print("4. 尝试不同的机器学习算法")
    print("5. 集成到量化策略中")

if __name__ == "__main__":
    main() 