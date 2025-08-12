#!/usr/bin/env python3
"""
运行增强模型训练的主脚本
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.enhanced_ml_training import EnhancedMLTrainer

def main():
    """主函数"""
    print("🚀 myQuant 增强版机器学习模型训练")
    print("=" * 50)
    
    # 创建增强训练器
    trainer = EnhancedMLTrainer()
    
    # 初始化Qlib
    if not trainer.init_qlib():
        print("❌ Qlib初始化失败，请检查环境配置")
        return
    
    # 股票池和时间范围
    symbols = [
        "SH600000", "SH600004", "SH600009", "SH600010", "SH600011",
        "SH600016", "SH600019", "SH600025", "SH600027", "SH600028"
    ]
    start_time = "2019-01-01"
    end_time = "2020-09-25"
    
    print(f"\n📊 训练配置:")
    print(f"   股票池: {len(symbols)} 只股票")
    print(f"   训练期间: {start_time} 到 {end_time}")
    print(f"   特征类型: 增强技术指标 + 统计特征")
    print(f"   模型算法: 随机森林 + 梯度提升 + 逻辑回归 + SVM")
    
    # 创建增强特征
    print(f"\n🔧 开始特征工程...")
    features, labels = trainer.create_enhanced_features(symbols, start_time, end_time)
    
    if features is None:
        print("❌ 特征创建失败")
        return
    
    print(f"\n✅ 特征工程完成!")
    print(f"   特征数量: {len(features.columns)}")
    print(f"   样本数量: {len(features)}")
    print(f"   标签分布: {labels.value_counts().to_dict()}")
    
    # 训练增强模型
    print(f"\n🤖 开始训练增强模型...")
    if not trainer.train_enhanced_models(features, labels):
        print("❌ 模型训练失败")
        return
    
    # 绘制模型对比
    print(f"\n🎨 生成模型对比图表...")
    trainer.plot_model_comparison()
    
    # 保存报告
    print(f"\n📝 保存训练报告...")
    trainer.save_enhanced_report()
    
    # 显示最佳模型
    best_model = max(trainer.results.keys(), key=lambda x: trainer.results[x]['accuracy'])
    best_accuracy = trainer.results[best_model]['accuracy']
    
    print(f"\n🏆 训练结果总结:")
    print(f"   最佳模型: {best_model}")
    print(f"   最佳准确率: {best_accuracy:.4f}")
    print(f"   模型数量: {len(trainer.models)}")
    
    print(f"\n💡 查看详细结果:")
    print(f"1. 模型对比图表: results/enhanced_models_comparison.html")
    print(f"2. 训练报告: results/enhanced_models_report.txt")
    
    print(f"\n🎉 增强模型训练完成！")

if __name__ == "__main__":
    main() 