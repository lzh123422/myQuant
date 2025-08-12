#!/usr/bin/env python3
"""
机器学习模型训练脚本
用于训练股票预测模型
"""

import sys
from pathlib import Path
import qlib
from qlib.data import D
from qlib.contrib.model import LGBModel
from qlib.contrib.data import Alpha158
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

class MLModelTrainer:
    """机器学习模型训练器"""
    
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
    
    def create_dataset(self, instruments="csi300", start_time="2018-01-01", end_time="2020-09-25"):
        """创建数据集"""
        if not self.qlib_initialized:
            print("❌ 请先初始化Qlib")
            return None
        
        try:
            print(f"📊 创建数据集...")
            print(f"   股票池: {instruments}")
            print(f"   时间范围: {start_time} 到 {end_time}")
            
            # 创建Alpha158数据集处理器
            handler = Alpha158(
                start_time=start_time,
                end_time=end_time,
                instruments=instruments,
                infer_processors=[
                    {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                learn_processors=[
                    {"class": "DropnaLabel"},
                    {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                ],
                fit_start_time=start_time,
                fit_end_time=end_time,
            )
            
            print("✅ 数据集处理器创建成功")
            return handler
            
        except Exception as e:
            print(f"❌ 数据集创建失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_model(self, model_name, handler, model_config=None):
        """训练模型"""
        try:
            print(f"🤖 开始训练模型: {model_name}")
            
            # 默认模型配置
            if model_config is None:
                model_config = {
                    "loss": "mse",
                    "colsample_bytree": 0.8879,
                    "learning_rate": 0.2,
                    "subsample": 0.8789,
                    "lambda_l1": 205.6999,
                    "lambda_l2": 580.9768,
                    "max_depth": 8,
                    "num_leaves": 210,
                    "num_threads": 20,
                }
            
            # 创建模型
            model = LGBModel(**model_config)
            
            # 训练模型
            with R.start(experiment_name=f"{model_name}_training"):
                sr = SignalRecord(
                    model=model,
                    dataset=handler,
                    port_analysis_config={
                        "benchmark": "000300.SH",
                        "account": 100000000,
                        "exchange_kwargs": {
                            "freq": "day",
                            "limit_threshold": 0.095,
                            "deal_price": "close",
                            "open_cost": 0.0005,
                            "close_cost": 0.0015,
                            "min_cost": 5,
                        },
                    },
                )
                sr.generate()
                
                # 保存训练结果
                self.training_results[model_name] = {
                    'model': model,
                    'handler': handler,
                    'config': model_config,
                    'experiment': R.get_exp(experiment_name=f"{model_name}_training")
                }
                
                print(f"✅ 模型训练完成: {model_name}")
                return model
                
        except Exception as e:
            print(f"❌ 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_model(self, model_name):
        """评估模型性能"""
        if model_name not in self.training_results:
            print(f"❌ 模型 '{model_name}' 没有训练结果")
            return None
        
        try:
            print(f"📊 评估模型性能: {model_name}")
            
            result = self.training_results[model_name]
            experiment = result['experiment']
            
            # 获取评估记录
            recorders = experiment.list_recorders()
            if not recorders:
                print("❌ 未找到评估记录")
                return None
            
            # 获取最新的评估记录
            latest_recorder = recorders[-1]
            
            # 加载评估结果
            report = latest_recorder.load_object("report.pkl")
            if report is None:
                print("❌ 未找到评估报告")
                return None
            
            print(f"📈 模型评估结果:")
            print(f"   信息系数(IC): {report.get('ic', 'N/A')}")
            print(f"   排名IC: {report.get('rank_ic', 'N/A')}")
            print(f"   年化收益率: {report.get('annualized_return', 'N/A')}")
            print(f"   夏普比率: {report.get('sharpe', 'N/A')}")
            print(f"   最大回撤: {report.get('max_drawdown', 'N/A')}")
            
            return report
            
        except Exception as e:
            print(f"❌ 模型评估失败: {e}")
            return None
    
    def plot_model_performance(self, model_name):
        """绘制模型性能图表"""
        report = self.evaluate_model(model_name)
        if report is None:
            return
        
        try:
            print(f"🎨 绘制模型性能图表: {model_name}")
            
            # 这里可以根据实际的report数据结构来绘制图表
            # 由于report结构可能不同，我们先创建一个基础的图表
            
            fig = go.Figure()
            
            # 添加模型性能指标
            metrics = ['IC', 'Rank IC', 'Annualized Return', 'Sharpe', 'Max Drawdown']
            values = [
                report.get('ic', 0),
                report.get('rank_ic', 0),
                report.get('annualized_return', 0),
                report.get('sharpe', 0),
                report.get('max_drawdown', 0)
            ]
            
            fig.add_trace(go.Bar(
                x=metrics,
                y=values,
                name='模型性能',
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title=f'{model_name} 模型性能评估',
                xaxis_title='评估指标',
                yaxis_title='指标值',
                height=500
            )
            
            # 保存图表
            fig.write_html(f'{model_name}_performance.html')
            print(f"✅ 性能图表已保存为: {model_name}_performance.html")
            
            # 显示图表
            fig.show()
            
        except Exception as e:
            print(f"❌ 图表绘制失败: {e}")
    
    def save_model_report(self, model_name, filename=None):
        """保存模型报告"""
        if filename is None:
            filename = f"{model_name}_model_report.txt"
        
        try:
            print(f"📝 保存模型报告: {filename}")
            
            report = self.evaluate_model(model_name)
            if report is None:
                return False
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"机器学习模型训练报告: {model_name}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("📊 模型性能:\n")
                f.write(f"   信息系数(IC): {report.get('ic', 'N/A')}\n")
                f.write(f"   排名IC: {report.get('rank_ic', 'N/A')}\n")
                f.write(f"   年化收益率: {report.get('annualized_return', 'N/A')}\n")
                f.write(f"   夏普比率: {report.get('sharpe', 'N/A')}\n")
                f.write(f"   最大回撤: {report.get('max_drawdown', 'N/A')}\n\n")
                
                f.write("🎯 模型配置:\n")
                if model_name in self.training_results:
                    config = self.training_results[model_name]['config']
                    for key, value in config.items():
                        f.write(f"   {key}: {value}\n")
                
                f.write("\n📅 训练信息:\n")
                f.write(f"   模型类型: LGBModel\n")
                f.write(f"   数据集: Alpha158\n")
                f.write(f"   股票池: CSI300\n")
            
            print(f"✅ 模型报告已保存: {filename}")
            return True
            
        except Exception as e:
            print(f"❌ 报告保存失败: {e}")
            return False

def main():
    """主函数"""
    print("🚀 机器学习模型训练系统启动")
    print("=" * 50)
    
    # 创建训练器
    trainer = MLModelTrainer()
    
    # 初始化Qlib
    if not trainer.init_qlib():
        return
    
    # 创建数据集
    print("\n📊 创建训练数据集...")
    handler = trainer.create_dataset(
        instruments="csi300",
        start_time="2018-01-01",
        end_time="2020-09-25"
    )
    
    if handler is None:
        print("❌ 数据集创建失败，无法继续训练")
        return
    
    # 训练模型
    print("\n🤖 开始训练机器学习模型...")
    model = trainer.train_model("LGB预测模型", handler)
    
    if model is None:
        print("❌ 模型训练失败")
        return
    
    # 评估模型
    print("\n📊 评估模型性能...")
    trainer.evaluate_model("LGB预测模型")
    
    # 绘制性能图表
    print("\n🎨 绘制模型性能图表...")
    trainer.plot_model_performance("LGB预测模型")
    
    # 保存报告
    print("\n📝 保存模型报告...")
    trainer.save_model_report("LGB预测模型")
    
    print("\n✅ 机器学习模型训练完成！")
    print("\n💡 下一步:")
    print("1. 查看生成的HTML图表")
    print("2. 阅读模型报告")
    print("3. 调整模型参数重新训练")
    print("4. 尝试不同的模型类型")
    print("5. 集成到量化策略中")

if __name__ == "__main__":
    main() 