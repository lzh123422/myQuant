# myQuant - 全自动量化投资系统

## 🚀 项目简介

myQuant是一个基于机器学习的全自动量化投资系统，使用Qlib框架进行股票数据分析和策略回测。

## 📊 核心功能

- **🤖 机器学习模型**: 自动生成买卖信号
- **📈 全自动回测**: 历史数据模拟投资
- **🎨 可视化界面**: Streamlit Web应用
- **📊 实时分析**: 股票技术指标分析
- **📋 详细报告**: 回测结果和交易记录

## 🏗️ 项目结构

```
myQuant/
├── 📁 src/                    # 源代码
│   ├── 📁 core/              # 核心功能
│   ├── 📁 strategies/        # 交易策略
│   ├── 📁 models/            # 机器学习模型
│   ├── 📁 backtest/          # 回测系统
│   └── 📁 utils/             # 工具函数
├── 📁 config/                # 配置文件
├── 📁 data/                  # 数据相关
├── 📁 models/                # 训练好的模型
├── 📁 results/               # 回测结果和报告
├── 📁 scripts/               # 运行脚本
├── 📁 docs/                  # 文档
├── 📁 tests/                 # 测试文件
└── 📁 examples/              # 示例代码
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate qlib-mcp

# 或创建新环境
conda env create -f config/environment-mcp.yml
```

### 2. 运行回测

```bash
# 运行全自动回测
python scripts/run_backtest.py
```

### 3. 启动Web应用

```bash
# 启动Streamlit应用
python scripts/run_streamlit.py

# 或直接使用streamlit
streamlit run src/core/streamlit_app.py
```

## 📊 回测结果示例

- **初始资金**: 1,000,000 元
- **最终价值**: 1,235,603 元
- **总收益率**: 23.56%
- **交易次数**: 176次
- **回测期间**: 2020年1月-9月

## 🔧 配置说明

主要配置在 `config/project_config.py` 中：

- **数据配置**: 股票池、时间范围
- **模型配置**: 算法参数、特征选择
- **回测配置**: 资金管理、信号阈值

## 📁 文件说明

### 核心文件
- `src/backtest/auto_backtest_system.py` - 全自动回测系统
- `src/core/streamlit_app.py` - Web可视化应用
- `src/models/simple_ml_training.py` - 机器学习模型训练

### 运行脚本
- `scripts/run_backtest.py` - 回测运行脚本
- `scripts/run_streamlit.py` - Web应用启动脚本

### 配置文件
- `config/project_config.py` - 项目主配置
- `config/environment-mcp.yml` - Conda环境配置

## 🎯 使用场景

- **个人投资者**: 策略回测和优化
- **量化研究员**: 模型开发和测试
- **投资机构**: 策略验证和展示
- **学习研究**: 量化投资入门

## 📈 技术特点

- **完全自动化**: 无需人工干预
- **机器学习驱动**: 基于历史数据训练
- **实时信号**: 每日生成买卖建议
- **风险控制**: 智能仓位管理
- **可视化展示**: 直观的结果展示

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 📞 联系方式

如有问题，请提交Issue或联系开发团队。 