#!/usr/bin/env python3
"""
增强版机器学习模型训练脚本
使用更先进的算法和特征工程
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

class EnhancedMLTrainer:
    """增强版机器学习模型训练器"""
    
    def __init__(self):
        """初始化训练器"""
        self.qlib_initialized = False
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
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
    
    def create_enhanced_features(self, symbols, start_time, end_time):
        """创建增强特征"""
        print(f"📊 创建增强特征...")
        print(f"   股票池: {symbols}")
        print(f"   时间范围: {start_time} 到 {end_time}")
        
        try:
            # 一次性获取所有股票的基础数据
            fields = ["$close", "$open", "$high", "$low", "$volume", "$factor"]
            data = D.features(symbols, fields, start_time=start_time, end_time=end_time)
            
            if data.empty:
                print("❌ 获取的数据为空")
                return None, None
            
            print(f"   ✅ 数据获取成功，数据形状: {data.shape}")
            
            all_features = []
            all_labels = []
            
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
                    
                    if len(df) < 50:  # 至少需要50个数据点
                        print(f"   ⚠️  {symbol} 有效数据不足，跳过")
                        continue
                    
                    # 创建标签（未来5天涨跌）
                    df['future_return'] = df['close'].shift(-5) / df['close'] - 1
                    df['label'] = np.where(df['future_return'] > 0.02, 1, 0)  # 2%阈值
                    
                    # 选择特征列
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
                    available_features = [col for col in feature_columns if col in df.columns]
                    features = df[available_features].copy()
                    labels = df['label'].copy()
                    
                    # 删除包含NaN的行
                    valid_mask = ~(features.isna().any(axis=1) | labels.isna())
                    features = features[valid_mask]
                    labels = labels[valid_mask]
                    
                    if len(features) > 0:
                        all_features.append(features)
                        all_labels.append(labels)
                        print(f"   ✅ {symbol} 特征创建成功，样本数: {len(features)}")
                    
                except Exception as e:
                    print(f"   ❌ 处理 {symbol} 时出错: {e}")
                    continue
            
            if not all_features:
                print("❌ 没有成功创建任何特征")
                return None, None
            
            # 合并所有股票的特征
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_labels = pd.concat(all_labels, ignore_index=True)
            
            print(f"✅ 特征创建完成")
            print(f"   总样本数: {len(combined_features)}")
            print(f"   特征数量: {len(combined_features.columns)}")
            print(f"   标签分布: {combined_labels.value_counts().to_dict()}")
            
            return combined_features, combined_labels
            
        except Exception as e:
            print(f"❌ 特征创建过程中出错: {e}")
            return None, None
    
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
    
    def train_enhanced_models(self, features_df, labels):
        """训练多个增强模型"""
        print("\n🤖 开始训练增强模型...")
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        from sklearn.preprocessing import StandardScaler
        
        # 数据预处理
        X = features_df.fillna(0)  # 填充缺失值
        y = labels
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 定义模型
        models = {
            'RandomForest_Enhanced': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=8, random_state=42, learning_rate=0.1
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, max_iter=1000, C=1.0
            ),
            'SVM': SVC(
                random_state=42, probability=True, kernel='rbf'
            )
        }
        
        # 训练和评估模型
        for name, model in models.items():
            print(f"\n📊 训练模型: {name}")
            
            try:
                if name in ['LogisticRegression', 'SVM']:
                    # 线性模型使用标准化数据
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
                else:
                    # 树模型使用原始数据
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # 计算准确率
                accuracy = accuracy_score(y_test, y_pred)
                
                # 保存结果
                self.models[name] = model
                self.results[name] = {
                    'accuracy': accuracy,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'y_test': y_test,
                    'feature_names': X.columns.tolist()
                }
                
                # 获取特征重要性（如果可用）
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
                
                print(f"   ✅ 训练完成，准确率: {accuracy:.4f}")
                
                # 详细分类报告
                print(f"   分类报告:")
                print(classification_report(y_test, y_pred))
                
            except Exception as e:
                print(f"   ❌ 训练失败: {e}")
                continue
        
        print(f"\n✅ 所有模型训练完成！")
        return True
    
    def plot_model_comparison(self):
        """绘制模型性能对比"""
        if not self.results:
            print("❌ 没有训练结果可供绘制")
            return
        
        print("\n🎨 绘制模型性能对比...")
        
        # 准备数据
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['模型准确率对比', '特征重要性对比', '预测概率分布', '混淆矩阵'],
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "heatmap"}]]
        )
        
        # 1. 准确率对比
        fig.add_trace(
            go.Bar(x=model_names, y=accuracies, name='准确率', 
                   marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']),
            row=1, col=1
        )
        
        # 2. 特征重要性对比（选择第一个有特征重要性的模型）
        if self.feature_importance:
            first_model = list(self.feature_importance.keys())[0]
            importance_data = self.feature_importance[first_model]
            top_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:15]
            
            feature_names = [item[0] for item in top_features]
            importance_values = [item[1] for item in top_features]
            
            fig.add_trace(
                go.Bar(x=importance_values, y=feature_names, orientation='h', 
                       name='特征重要性', marker_color='#2ca02c'),
                row=1, col=2
            )
        
        # 3. 预测概率分布
        for name in model_names:
            if self.results[name]['y_pred_proba'] is not None:
                fig.add_trace(
                    go.Histogram(x=self.results[name]['y_pred_proba'], name=f'{name}_概率',
                               opacity=0.7, nbinsx=20),
                    row=2, col=1
                )
        
        # 4. 混淆矩阵（选择第一个模型）
        first_model = model_names[0]
        y_test = self.results[first_model]['y_test']
        y_pred = self.results[first_model]['y_pred']
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        
        fig.add_trace(
            go.Heatmap(z=cm, x=['预测下跌', '预测上涨'], y=['实际下跌', '实际上涨'],
                       colorscale='Blues', name='混淆矩阵'),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title='增强模型性能对比分析',
            height=800,
            showlegend=True
        )
        
        # 保存图表
        output_file = 'results/enhanced_models_comparison.html'
        fig.write_html(output_file)
        print(f"✅ 模型对比图表已保存: {output_file}")
        
        return fig
    
    def save_enhanced_report(self, filename=None):
        """保存增强模型报告"""
        if not self.results:
            print("❌ 没有训练结果可供保存")
            return
        
        if filename is None:
            filename = 'results/enhanced_models_report.txt'
        
        print(f"\n📝 保存增强模型报告: {filename}")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("增强版机器学习模型训练报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 模型性能对比
            f.write("📊 模型性能对比:\n")
            f.write("-" * 30 + "\n")
            for name, result in self.results.items():
                f.write(f"模型: {name}\n")
                f.write(f"准确率: {result['accuracy']:.4f}\n")
                f.write("-" * 20 + "\n")
            
            # 最佳模型
            best_model = max(self.results.keys(), key=lambda x: self.results[x]['accuracy'])
            f.write(f"\n🏆 最佳模型: {best_model}\n")
            f.write(f"最佳准确率: {self.results[best_model]['accuracy']:.4f}\n\n")
            
            # 特征重要性
            if self.feature_importance:
                f.write("🎯 特征重要性分析:\n")
                f.write("-" * 30 + "\n")
                for model_name, importance in self.feature_importance.items():
                    f.write(f"\n{model_name} 特征重要性 (前15):\n")
                    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
                    for feature, imp in sorted_importance:
                        f.write(f"   {feature}: {imp:.4f}\n")
            
            # 训练信息
            f.write(f"\n📅 训练信息:\n")
            f.write("-" * 30 + "\n")
            f.write(f"模型数量: {len(self.models)}\n")
            f.write(f"特征数量: {len(self.results[list(self.results.keys())[0]]['feature_names'])}\n")
            f.write(f"算法类型: 集成学习 + 线性模型 + SVM\n")
            f.write(f"特征工程: 技术指标 + 统计特征 + 趋势特征\n")
        
        print(f"✅ 增强模型报告已保存: {filename}")

def main():
    """主函数"""
    print("🚀 增强版机器学习模型训练")
    print("=" * 50)
    
    # 创建训练器
    trainer = EnhancedMLTrainer()
    
    # 初始化Qlib
    if not trainer.init_qlib():
        return
    
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
        return
    
    # 训练增强模型
    if not trainer.train_enhanced_models(features, labels):
        print("❌ 模型训练失败")
        return
    
    # 绘制模型对比
    trainer.plot_model_comparison()
    
    # 保存报告
    trainer.save_enhanced_report()
    
    print("\n🎉 增强模型训练完成！")
    print("💡 查看结果:")
    print("1. 模型对比图表: results/enhanced_models_comparison.html")
    print("2. 训练报告: results/enhanced_models_report.txt")

if __name__ == "__main__":
    main() 