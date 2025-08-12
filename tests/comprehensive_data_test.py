#!/usr/bin/env python3
"""
全面的数据测试脚本
"""

import sys
from pathlib import Path
import qlib
from qlib.data import D
import warnings
warnings.filterwarnings('ignore')

def comprehensive_data_test():
    """全面的数据测试"""
    
    print("🔍 全面数据测试...")
    
    try:
        # 初始化Qlib
        print("🔧 初始化Qlib...")
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        print("✅ Qlib初始化成功")
        
        # 测试单个字段
        print(f"\n📊 测试单个字段...")
        symbols = ["SH600000"]
        start_time = "2020-01-01"
        end_time = "2020-01-31"
        
        # 测试基本字段
        basic_fields = ["$close", "$volume", "$factor", "$open", "$high", "$low"]
        
        for field in basic_fields:
            print(f"\n🔍 测试字段: {field}")
            try:
                data = D.features(symbols, [field], start_time=start_time, end_time=end_time)
                print(f"   ✅ 成功! 数据形状: {data.shape}")
                if not data.empty:
                    print(f"   📅 数据范围: {data.index.min()} 到 {data.index.max()}")
                    print(f"   📊 数据统计:")
                    print(f"      最小值: {data[field].min()}")
                    print(f"      最大值: {data[field].max()}")
                    print(f"      平均值: {data[field].mean():.4f}")
                else:
                    print("   ⚠️  数据为空")
                    
            except Exception as e:
                print(f"   ❌ 失败: {str(e)[:100]}...")
        
        # 测试多只股票
        print(f"\n📊 测试多只股票...")
        multiple_symbols = ["SH600000", "SH600004", "SH600009"]
        
        try:
            data = D.features(multiple_symbols, ["$close"], start_time=start_time, end_time=end_time)
            print(f"✅ 多股票数据获取成功! 形状: {data.shape}")
            print(f"📋 股票列表: {list(data.index.get_level_values('instrument').unique())}")
            print(f"📅 时间范围: {data.index.get_level_values('datetime').min()} 到 {data.index.get_level_values('datetime').max()}")
            
        except Exception as e:
            print(f"❌ 多股票数据获取失败: {e}")
        
        # 测试不同时间范围
        print(f"\n📅 测试不同时间范围...")
        time_ranges = [
            ("2020-01-01", "2020-01-31"),
            ("2019-01-01", "2019-12-31"),
            ("2018-01-01", "2018-12-31")
        ]
        
        for start, end in time_ranges:
            print(f"\n🔍 测试时间范围: {start} 到 {end}")
            try:
                data = D.features(symbols, ["$close"], start_time=start, end_time=end)
                print(f"   ✅ 成功! 数据形状: {data.shape}")
                if not data.empty:
                    print(f"   📅 实际数据范围: {data.index.min()} 到 {data.index.max()}")
                else:
                    print("   ⚠️  数据为空")
                    
            except Exception as e:
                print(f"   ❌ 失败: {str(e)[:100]}...")
        
        # 测试策略相关功能
        print(f"\n🎯 测试策略相关功能...")
        try:
            from qlib.contrib.strategy import TopkDropoutStrategy
            print("✅ TopkDropoutStrategy可用")
            
            # 创建简单策略
            strategy = TopkDropoutStrategy(topk=3, n_drop=1)
            print("✅ 策略创建成功")
            
        except Exception as e:
            print(f"❌ 策略功能不可用: {e}")
        
        # 测试回测功能
        print(f"\n🔄 测试回测功能...")
        try:
            from qlib.backtest import backtest, executor
            print("✅ 回测模块可用")
            
        except Exception as e:
            print(f"❌ 回测功能不可用: {e}")
        
        print(f"\n✅ 全面数据测试完成!")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    comprehensive_data_test() 