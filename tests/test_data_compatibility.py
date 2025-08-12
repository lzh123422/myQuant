#!/usr/bin/env python3
"""
测试数据兼容性和可用性
"""

import sys
from pathlib import Path
import qlib
from qlib.data import D
import warnings
warnings.filterwarnings('ignore')

def test_data_compatibility():
    """测试数据兼容性"""
    
    print("🔍 测试数据兼容性...")
    
    try:
        # 初始化Qlib
        print("🔧 初始化Qlib...")
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        print("✅ Qlib初始化成功")
        
        # 测试不同的字段名
        field_tests = [
            "$close",
            "close", 
            "CLOSE",
            "Close",
            "close.day",
            "close_day"
        ]
        
        symbols = ["SH600000"]
        start_time = "2020-01-01"
        end_time = "2020-01-31"
        
        print(f"\n📊 测试股票: {symbols}")
        print(f"📅 时间范围: {start_time} 到 {end_time}")
        
        for field in field_tests:
            print(f"\n🔍 测试字段: {field}")
            try:
                data = D.features(symbols, [field], start_time=start_time, end_time=end_time)
                print(f"   ✅ 成功! 数据形状: {data.shape}")
                if not data.empty:
                    print(f"   📅 实际数据范围: {data.index.min()} 到 {data.index.max()}")
                    print(f"   📋 样本数据:")
                    print(data.head(3))
                else:
                    print("   ⚠️  数据为空")
                break  # 找到可用的字段就停止
                
            except Exception as e:
                print(f"   ❌ 失败: {str(e)[:100]}...")
        
        # 测试可用的字段
        print(f"\n🔍 测试可用字段...")
        try:
            # 尝试获取基本信息
            from qlib.data.dataset import DatasetH
            print("✅ DatasetH模块可用")
            
            # 尝试创建数据集
            from qlib.data.dataset.handler import Alpha158
            print("✅ Alpha158处理器可用")
            
        except Exception as e:
            print(f"❌ 高级功能不可用: {e}")
        
        # 测试股票列表
        print(f"\n📋 测试股票列表...")
        try:
            # 读取股票列表文件
            import os
            csi300_path = os.path.expanduser("~/.qlib/qlib_data/cn_data/instruments/csi300.txt")
            
            if os.path.exists(csi300_path):
                with open(csi300_path, 'r') as f:
                    stocks = [line.split()[0] for line in f.readlines()[:10]]
                print(f"✅ 找到股票列表，前10只: {stocks}")
            else:
                print("❌ 股票列表文件不存在")
                
        except Exception as e:
            print(f"❌ 读取股票列表失败: {e}")
        
        print(f"\n✅ 数据兼容性测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_compatibility() 