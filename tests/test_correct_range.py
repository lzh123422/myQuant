#!/usr/bin/env python3
"""
Test China stocks with correct time range
"""

import sys
from pathlib import Path

# Add qlib to path
sys.path.insert(0, str(Path(__file__).parent))

def test_correct_range():
    """Test with correct time range."""
    
    try:
        print("🚀 使用正确时间范围测试中国股票...")
        
        # Import Qlib
        import qlib
        from qlib.data import D
        
        print("✅ Qlib导入成功")
        
        # 初始化Qlib
        print("🔧 初始化Qlib...")
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        print("✅ Qlib初始化成功")
        
        # 使用数据实际可用的时间范围
        symbols = ["SH600000", "SH600004", "SH600009"]  # 使用上海股票代码
        fields = ["$close", "$volume"]
        start_time = "2020-01-01"
        end_time = "2020-09-25"  # 数据实际结束时间
        
        print(f"\n🔍 获取股票数据:")
        print(f"   股票代码: {symbols}")
        print(f"   数据字段: {fields}")
        print(f"   时间范围: {start_time} 到 {end_time}")
        
        try:
            data = D.features(symbols, fields, start_time=start_time, end_time=end_time)
            print(f"✅ 数据获取成功")
            print(f"📊 数据形状: {data.shape}")
            
            if not data.empty:
                print(f"📅 实际数据范围: {data.index.min()} 到 {data.index.max()}")
                print(f"📋 样本数据:")
                print(data.head())
                
                # 显示一些统计信息
                print(f"\n📈 数据统计:")
                print(f"   收盘价范围: {data['$close'].min():.2f} - {data['$close'].max():.2f}")
                print(f"   成交量范围: {data['$volume'].min():.0f} - {data['$volume'].max():.0f}")
            else:
                print("⚠️  数据为空")
                
        except Exception as e:
            print(f"❌ 数据获取失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试获取更多股票
        print(f"\n🔍 测试获取更多股票...")
        try:
            # 获取沪深300的所有股票
            with open("~/.qlib/qlib_data/cn_data/instruments/csi300.txt", "r") as f:
                csi300_stocks = [line.split()[0] for line in f.readlines()[:10]]  # 前10只股票
            
            print(f"📋 测试股票: {csi300_stocks}")
            
            # 获取这些股票的数据
            more_data = D.features(csi300_stocks, ["$close"], start_time="2020-01-01", end_time="2020-01-31")
            print(f"✅ 多股票数据获取成功")
            print(f"📊 数据形状: {more_data.shape}")
            
            if not more_data.empty:
                print(f"📅 数据范围: {more_data.index.min()} 到 {more_data.index.max()}")
                print(f"📋 样本数据:")
                print(more_data.head())
            
        except Exception as e:
            print(f"⚠️  多股票测试失败: {e}")
        
        print("\n✅ 正确时间范围测试完成！")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_correct_range() 