# 🎨 可视化功能使用指南

## 🚀 概述

你的myQuant项目现在具备了完整的可视化功能，不再需要一直看命令行！我们提供了多种可视化界面：

## 📊 可用的可视化工具

### 1. **Jupyter Notebook** - 交互式分析环境
- **启动方式**: `./start_jupyter.sh`
- **访问地址**: http://localhost:8888
- **特点**: 
  - 交互式代码执行
  - 实时图表显示
  - 支持复杂分析
  - 可以保存分析结果

### 2. **Streamlit Web应用** - 现代化Web界面
- **启动方式**: `./start_streamlit.sh`
- **访问地址**: http://localhost:8501
- **特点**:
  - 美观的Web界面
  - 拖拽式参数调整
  - 实时图表更新
  - 无需编程知识

### 3. **静态图表生成** - 高质量图片输出
- **启动方式**: `python test_visualization.py`
- **输出文件**:
  - `china_stocks_analysis.png` - Matplotlib图表
  - `china_stocks_interactive.html` - Plotly交互图表
  - `china_stocks_seaborn_analysis.png` - Seaborn统计图表

## 🎯 功能特性

### 📈 图表类型
- **价格走势图** - 股票收盘价时间序列
- **成交量图** - 交易量柱状图
- **技术指标** - MA、RSI、MACD等
- **相关性热力图** - 股票间相关性分析
- **统计分布图** - 收益率分布、箱线图等

### 🔧 技术指标
- **移动平均线** - MA5、MA20、MA60
- **RSI指标** - 相对强弱指数
- **布林带** - 价格波动区间
- **MACD** - 趋势指标
- **KDJ** - 随机指标

### 📊 统计分析
- **描述性统计** - 均值、标准差、分位数
- **收益率分析** - 日收益率、累计收益率
- **风险指标** - 波动率、最大回撤、夏普比率
- **相关性分析** - 股票间相关性矩阵

## 🚀 快速开始

### 方法1: 启动Jupyter Notebook（推荐）
```bash
# 激活环境
conda activate qlib-mcp

# 启动Jupyter
./start_jupyter.sh
```

然后在浏览器中打开 `china_stocks_analysis.ipynb` 文件，开始交互式分析！

### 方法2: 启动Streamlit Web应用
```bash
# 激活环境
conda activate qlib-mcp

# 启动Streamlit
./start_streamlit.sh
```

在Web界面中：
1. 选择股票类别（银行股、科技股、消费股、能源股）
2. 调整时间范围
3. 点击"获取数据"开始分析

### 方法3: 生成静态图表
```bash
# 激活环境
conda activate qlib-mcp

# 生成图表
python test_visualization.py
```

## 💡 使用技巧

### 🎨 自定义图表
1. **修改股票代码**: 在代码中更改 `symbols` 列表
2. **调整时间范围**: 修改 `start_time` 和 `end_time`
3. **添加技术指标**: 在 `calculate_technical_indicators` 函数中添加新指标
4. **更改图表样式**: 修改Plotly的 `layout` 参数

### 📱 移动端支持
- Streamlit应用支持移动端访问
- Plotly图表支持触摸操作
- 响应式设计，自动适配屏幕尺寸

### 🔄 实时更新
- 使用 `st.cache_data` 缓存数据，提高性能
- 支持实时数据刷新
- 可以设置自动更新间隔

## 🎯 高级功能

### 🤖 AI助手集成
通过MCP工具，AI助手可以：
- 自动生成分析报告
- 执行复杂的量化策略
- 优化投资组合配置
- 提供投资建议

### 📊 数据导出
- 支持CSV、Excel格式导出
- 图表可保存为PNG、SVG、PDF
- 交互式图表可导出为HTML

### 🔗 外部集成
- 支持连接实时数据源
- 可集成其他量化平台
- 支持API接口调用

## 🚨 注意事项

### ⚠️ 系统要求
- 确保conda环境已激活：`conda activate qlib-mcp`
- 检查数据路径：`~/.qlib/qlib_data/cn_data`
- 确保网络连接正常（用于获取数据）

### 🔧 故障排除
1. **图表不显示**: 检查浏览器控制台错误信息
2. **数据加载失败**: 验证Qlib配置和数据路径
3. **性能问题**: 减少同时分析的股票数量
4. **内存不足**: 分批处理大量数据

## 🎉 总结

现在你拥有了完整的可视化量化投资平台：

- ✅ **不再需要命令行** - 所有操作都有图形界面
- ✅ **交互式分析** - 实时调整参数，即时查看结果
- ✅ **专业图表** - 支持多种图表类型和技术指标
- ✅ **移动端支持** - 随时随地进行分析
- ✅ **AI助手集成** - 自动化分析和建议

选择你喜欢的界面开始量化投资之旅吧！🚀📈 