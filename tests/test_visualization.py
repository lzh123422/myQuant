#!/usr/bin/env python3
"""
Test visualization features for China stocks using Qlib
"""

import sys
from pathlib import Path

# Add qlib to path
sys.path.insert(0, str(Path(__file__).parent))

def test_visualization():
    """Test visualization functionality."""
    
    try:
        print("🚀 测试可视化功能...")
        
        # Import Qlib
        import qlib
        from qlib.data import D
        
        print("✅ Qlib导入成功")
        
        # 初始化Qlib
        print("🔧 初始化Qlib...")
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        print("✅ Qlib初始化成功")
        
        # 获取股票数据用于可视化
        print("\n📊 获取股票数据用于可视化...")
        symbols = ["SH600000", "SH600004", "SH600009"]
        fields = ["$close", "$volume"]
        start_time = "2020-01-01"
        end_time = "2020-09-25"
        
        data = D.features(symbols, fields, start_time=start_time, end_time=end_time)
        print(f"✅ 数据获取成功，形状: {data.shape}")
        
        # 测试不同的可视化库
        print("\n🎨 测试可视化库...")
        
        # 1. 测试Matplotlib
        print("📈 测试Matplotlib...")
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图表
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 绘制收盘价
            for symbol in symbols:
                symbol_data = data.loc[symbol, '$close']
                ax1.plot(symbol_data.index, symbol_data.values, label=symbol, linewidth=2)
            
            ax1.set_title('股票收盘价走势', fontsize=14, fontweight='bold')
            ax1.set_ylabel('收盘价 (元)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            
            # 绘制成交量
            for symbol in symbols:
                symbol_data = data.loc[symbol, '$volume']
                ax2.bar(symbol_data.index, symbol_data.values, label=symbol, alpha=0.7)
            
            ax2.set_title('股票成交量', fontsize=14, fontweight='bold')
            ax2.set_ylabel('成交量', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            
            plt.tight_layout()
            
            # 保存图表
            plt.savefig('china_stocks_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Matplotlib图表创建成功，已保存为 'china_stocks_analysis.png'")
            
            # 显示图表
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Matplotlib测试失败: {e}")
        
        # 2. 测试Plotly
        print("\n📊 测试Plotly...")
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import plotly.express as px
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('股票收盘价走势', '股票成交量'),
                vertical_spacing=0.1
            )
            
            # 添加收盘价线图
            for symbol in symbols:
                symbol_data = data.loc[symbol, '$close']
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data.index,
                        y=symbol_data.values,
                        name=symbol,
                        mode='lines',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
            
            # 添加成交量柱状图
            for symbol in symbols:
                symbol_data = data.loc[symbol, '$volume']
                fig.add_trace(
                    go.Bar(
                        x=symbol_data.index,
                        y=symbol_data.values,
                        name=f"{symbol}_volume",
                        opacity=0.7
                    ),
                    row=2, col=1
                )
            
            # 更新布局
            fig.update_layout(
                title='中国股票分析报告',
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            # 保存为HTML文件
            fig.write_html('china_stocks_interactive.html')
            print("✅ Plotly交互式图表创建成功，已保存为 'china_stocks_interactive.html'")
            
            # 显示图表
            fig.show()
            
        except Exception as e:
            print(f"⚠️  Plotly测试失败: {e}")
        
        # 3. 测试Seaborn
        print("\n🎭 测试Seaborn...")
        try:
            import seaborn as sns
            import pandas as pd
            
            # 设置样式
            sns.set_style("whitegrid")
            sns.set_palette("husl")
            
            # 准备数据
            plot_data = data.reset_index()
            plot_data['date'] = pd.to_datetime(plot_data['datetime'])
            
            # 创建图表
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 收盘价分布
            for i, symbol in enumerate(symbols):
                symbol_data = plot_data[plot_data['instrument'] == symbol]['$close']
                sns.histplot(symbol_data, kde=True, ax=axes[0, 0], label=symbol)
            axes[0, 0].set_title('收盘价分布')
            axes[0, 0].legend()
            
            # 成交量分布
            for i, symbol in enumerate(symbols):
                symbol_data = plot_data[plot_data['instrument'] == symbol]['$volume']
                sns.histplot(symbol_data, kde=True, ax=axes[0, 1], label=symbol)
            axes[0, 1].set_title('成交量分布')
            axes[0, 1].legend()
            
            # 收盘价箱线图
            sns.boxplot(data=plot_data, x='instrument', y='$close', ax=axes[1, 0])
            axes[1, 0].set_title('收盘价箱线图')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 成交量箱线图
            sns.boxplot(data=plot_data, x='instrument', y='$volume', ax=axes[1, 1])
            axes[1, 1].set_title('成交量箱线图')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # 保存图表
            plt.savefig('china_stocks_seaborn_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Seaborn图表创建成功，已保存为 'china_stocks_seaborn_analysis.png'")
            
            # 显示图表
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Seaborn测试失败: {e}")
        
        print("\n✅ 可视化功能测试完成！")
        print("\n💡 生成的文件:")
        print("   📊 china_stocks_analysis.png - Matplotlib静态图表")
        print("   🌐 china_stocks_interactive.html - Plotly交互式图表")
        print("   🎭 china_stocks_seaborn_analysis.png - Seaborn统计图表")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_visualization() 