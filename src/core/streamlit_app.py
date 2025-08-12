#!/usr/bin/env python3
"""
Streamlit Web应用 - 中国股票量化分析
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import qlib
from qlib.data import D
from datetime import datetime, timedelta

# 页面配置
st.set_page_config(
    page_title="中国股票量化分析",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题
st.title("🚀 中国股票量化分析平台")
st.markdown("基于Qlib的交互式量化投资分析工具")

# 侧边栏配置
st.sidebar.header("⚙️ 配置参数")

# 股票代码到中文名称的映射
STOCK_NAMES = {
    "SH600000": "平安银行",
    "SH600036": "招商银行", 
    "SH600016": "民生银行",
    "SH600519": "贵州茅台",
    "SH600887": "伊利股份",
    "SH600009": "上海机场",
    "SH600004": "白云机场",
    "SH600010": "包钢股份",
    "SH600011": "华能国际",
    "SH600028": "中国石化",
    "SH600019": "宝钢股份",
    "SH600027": "华电国际",
    "SH600015": "华夏银行",
    "SH600018": "上港集团",
    "SH600025": "华能水电",
    "SH600029": "南方航空",
    "SH600030": "中信证券",
    "SH600031": "三一重工",
    "SH600038": "中直股份",
    "SH600048": "保利发展",
    "SH600050": "中国联通",
    "SH600061": "国投资本"
}

# 预设股票列表（使用中文名称显示）
preset_stocks_display = {
    "银行股": ["平安银行", "招商银行", "民生银行"],
    "科技股": ["贵州茅台", "伊利股份", "上海机场"],
    "消费股": ["白云机场", "包钢股份", "华能国际"],
    "能源股": ["中国石化", "宝钢股份", "华电国际"]
}

# 反向映射：中文名称到股票代码
STOCK_CODES = {v: k for k, v in STOCK_NAMES.items()}

# 初始化Qlib
@st.cache_resource
def init_qlib():
    """初始化Qlib"""
    try:
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")
        return True
    except Exception as e:
        st.error(f"Qlib初始化失败: {e}")
        return False

# 获取股票数据
@st.cache_data
def get_stock_data(symbols, start_time, end_time):
    """获取股票数据"""
    try:
        fields = ["$close", "$volume", "$factor"]
        data = D.features(symbols, fields, start_time=start_time, end_time=end_time)
        return data
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None

# 计算技术指标
def calculate_technical_indicators(data, symbol):
    """计算技术指标"""
    symbol_data = data.loc[symbol, '$close']
    
    # 移动平均线
    ma5 = symbol_data.rolling(window=5).mean()
    ma20 = symbol_data.rolling(window=20).mean()
    
    # RSI指标
    delta = symbol_data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return pd.DataFrame({
        'close': symbol_data,
        'MA5': ma5,
        'MA20': ma20,
        'RSI': rsi
    })

# 主程序
def main():
    # 初始化Qlib
    if not init_qlib():
        st.error("系统初始化失败，请检查配置")
        return
    
    st.success("✅ Qlib初始化成功")
    
    # 侧边栏参数
    st.sidebar.subheader("�� 股票选择")
    
    # 选择股票类别
    stock_category = st.sidebar.selectbox(
        "选择股票类别",
        list(preset_stocks_display.keys())
    )
    
    selected_stocks_display = preset_stocks_display[stock_category]
    
    # 显示选中的股票
    st.sidebar.write("**当前选中的股票:**")
    for stock in selected_stocks_display:
        st.sidebar.write(f"• {stock}")
    
    # 将中文名称转换为股票代码
    selected_stocks = [STOCK_CODES[stock] for stock in selected_stocks_display if stock in STOCK_CODES]
    
    # 自定义股票选择
    st.sidebar.subheader("🔧 自定义股票")
    custom_stocks_input = st.sidebar.text_input(
        "输入股票代码（用逗号分隔）",
        value=",".join(selected_stocks),
        help="例如: SH600000,SH600036,SH600016"
    )
    
    if custom_stocks_input:
        custom_codes = [s.strip() for s in custom_stocks_input.split(",") if s.strip()]
        if custom_codes:
            selected_stocks = custom_codes
            # 显示自定义股票的中文名称
            st.sidebar.write("**自定义股票:**")
            for code in custom_codes:
                name = STOCK_NAMES.get(code, code)
                st.sidebar.write(f"• {name} ({code})")
    
    # 时间范围选择
    st.sidebar.subheader("📅 时间范围")
    
    # 使用数据实际可用的时间范围
    end_date = datetime(2020, 9, 25).date()  # 数据实际结束时间
    start_date = datetime(2019, 1, 1).date()  # 数据实际开始时间（往前推一年）
    
    start_time = st.sidebar.date_input("开始日期", value=start_date, min_value=datetime(2005, 1, 1).date(), max_value=end_date)
    end_time = st.sidebar.date_input("结束日期", value=end_date, min_value=start_time, max_value=end_date)
    
    # 显示数据可用范围提示
    st.sidebar.info(f"📊 数据可用范围: 2005-01-01 到 2020-09-25")
    
    # 转换日期格式
    start_time_str = start_time.strftime("%Y-%m-%d")
    end_time_str = end_time.strftime("%Y-%m-%d")
    
    # 获取数据
    if st.sidebar.button("🔄 获取数据"):
        with st.spinner("正在获取股票数据..."):
            data = get_stock_data(selected_stocks, start_time_str, end_time_str)
            
            if data is not None:
                st.session_state['stock_data'] = data
                st.success(f"✅ 成功获取 {len(selected_stocks)} 只股票的数据")
            else:
                st.error("❌ 数据获取失败")
    
    # 显示数据
    if 'stock_data' in st.session_state:
        data = st.session_state['stock_data']
        
        # 数据概览
        st.subheader("📊 数据概览")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("股票数量", len(selected_stocks))
        with col2:
            st.metric("数据记录数", data.shape[0])
        with col3:
            st.metric("数据字段数", data.shape[1])
        
        # 显示原始数据
        with st.expander("📋 查看原始数据"):
            st.dataframe(data.head(20))
        
        # 价格走势图
        st.subheader("📈 股票价格走势")
        
        fig = make_subplots(
            rows=len(selected_stocks), cols=1,
            subplot_titles=[f"{STOCK_NAMES.get(symbol, symbol)} 价格走势" for symbol in selected_stocks],
            vertical_spacing=0.1
        )
        
        for i, symbol in enumerate(selected_stocks):
            try:
                symbol_data = data.loc[symbol, '$close']
                fig.add_trace(
                    go.Scatter(
                        x=symbol_data.index,
                        y=symbol_data.values,
                        name=STOCK_NAMES.get(symbol, symbol),
                        mode='lines',
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )
            except:
                st.warning(f"无法获取 {STOCK_NAMES.get(symbol, symbol)} 的数据")
        
        fig.update_layout(height=200*len(selected_stocks), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # 技术指标分析
        st.subheader("🔧 技术指标分析")
        
        for symbol in selected_stocks:
            try:
                tech_data = calculate_technical_indicators(data, symbol)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**{STOCK_NAMES.get(symbol, symbol)} 价格与均线**")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['close'], name='收盘价'))
                    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA5'], name='MA5'))
                    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA20'], name='MA20'))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.write(f"**{STOCK_NAMES.get(symbol, symbol)} RSI指标**")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=tech_data.index, y=tech_data['RSI'], name='RSI'))
                    fig.add_hline(y=70, line_dash="dash", line_color="red")
                    fig.add_hline(y=30, line_dash="dash", line_color="green")
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.warning(f"无法计算 {STOCK_NAMES.get(symbol, symbol)} 的技术指标: {e}")
        
        # 统计分析
        st.subheader("📊 统计分析")
        
        stats_data = []
        for symbol in selected_stocks:
            try:
                close_data = data.loc[symbol, '$close']
                volume_data = data.loc[symbol, '$volume']
                returns = close_data.pct_change().dropna()
                
                stats_data.append({
                    '股票代码': symbol,
                    '股票名称': STOCK_NAMES.get(symbol, symbol),
                    '平均价格': f"{close_data.mean():.2f}",
                    '价格标准差': f"{close_data.std():.2f}",
                    '平均成交量': f"{volume_data.mean():.0f}",
                    '平均收益率': f"{returns.mean():.4f}",
                    '收益率标准差': f"{returns.std():.4f}",
                    '最大涨幅': f"{returns.max():.4f}",
                    '最大跌幅': f"{returns.min():.4f}"
                })
            except:
                pass
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)
        
        # 相关性分析
        st.subheader("🔗 相关性分析")
        
        try:
            returns_data = pd.DataFrame()
            for symbol in selected_stocks:
                close_data = data.loc[symbol, '$close']
                returns_data[STOCK_NAMES.get(symbol, symbol)] = close_data.pct_change().dropna()
            
            correlation_matrix = returns_data.corr()
            
            fig = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="股票收益率相关性热力图"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"相关性分析失败: {e}")
    
    # 页脚
    st.markdown("---")
    st.markdown("**💡 使用提示:**")
    st.markdown("- 在侧边栏选择股票类别或输入自定义股票代码")
    st.markdown("- 调整时间范围进行分析")
    st.markdown("- 点击'获取数据'按钮开始分析")
    st.markdown("- 使用MCP工具进行自动化分析")

if __name__ == "__main__":
    main() 