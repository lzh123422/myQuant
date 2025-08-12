#!/usr/bin/env python3
"""
Test different time ranges for China stocks data
"""

import sys
from pathlib import Path

# Add qlib to path
sys.path.insert(0, str(Path(__file__).parent))

def test_data_ranges():
    """Test different time ranges for data."""
    
    try:
        print("🚀 测试不同时间范围的中国股票数据...")
        
        # Import Qlib
        import qlib
        from qlib.data import D
        
        print("✅ Qlib导入成功")
        
        # 初始化Qlib
        print("🔧 初始化Qlib...")
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        print("✅ Qlib初始化成功")
        
        # 测试不同的时间范围
        symbols = ["000001.SZ", "000002.SZ"]
        fields = ["$close", "$volume"]
        
        time_ranges = [
            ("2023-01-01", "2023-12-31"),
            ("2022-01-01", "2022-12-31"),
            ("2021-01-01", "2021-12-31"),
            ("2020-01-01", "2020-12-31"),
            ("2019-01-01", "2019-12-31"),
        ]
        
        for start_time, end_time in time_ranges:
            print(f"\n🔍 测试时间范围: {start_time} 到 {end_time}")
            
            try:
                data = D.features(symbols, fields, start_time=start_time, end_time=end_time)
                print(f"   📊 数据形状: {data.shape}")
                
                if not data.empty:
                    print(f"   ✅ 成功获取数据")
                    print(f"   📅 实际数据范围: {data.index.min()} 到 {data.index.max()}")
                    print(f"   📋 样本数据:")
                    print(data.head(3))
                    break
                else:
                    print(f"   ⚠️  数据为空")
                    
            except Exception as e:
                print(f"   ❌ 错误: {e}")
        
        # 尝试获取可用的股票列表
        print("\n📋 尝试获取可用的股票列表...")
        try:
            from qlib.data.dataset import DatasetH
            from qlib.data.dataset.handler import Alpha158
            
            # 创建数据集处理器
            handler = Alpha158(
                start_time="2020-01-01",
                end_time="2020-12-31",
                instruments="csi300",  # 沪深300
            )
            
            # 获取数据
            dataset = DatasetH(handler)
            print("✅ 数据集创建成功")
            
            # 获取特征数据
            features = dataset.prepare("test", col_set="feature")
            print(f"📊 特征数据形状: {features.shape}")
            
            if not features.empty:
                print("✅ 成功获取特征数据")
                print(f"📅 数据范围: {features.index.min()} 到 {features.index.max()}")
                print(f"🔢 特征数量: {len(features.columns)}")
            
        except Exception as e:
            print(f"⚠️  数据集测试失败: {e}")
        
        print("\n✅ 时间范围测试完成！")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_data_ranges() 