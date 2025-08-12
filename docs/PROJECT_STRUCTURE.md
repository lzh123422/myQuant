# myQuant 项目结构说明

## 🏗️ 整体架构

```
myQuant/
├── 📁 src/                    # 源代码核心
├── 📁 config/                 # 配置文件
├── 📁 data/                   # 数据相关
├── 📁 models/                 # 训练好的模型
├── 📁 results/                # 回测结果和报告
├── 📁 scripts/                # 运行脚本
├── 📁 docs/                   # 文档
├── 📁 tests/                  # 测试文件
├── 📁 examples/               # 示例代码
└── 📁 logs/                   # 日志文件
```

## 📁 目录详细说明

### 1. `src/` - 源代码核心
```
src/
├── 📁 core/                   # 核心功能模块
│   ├── __init__.py
│   └── streamlit_app.py      # Streamlit Web应用
├── 📁 strategies/             # 交易策略模块
│   ├── __init__.py
│   └── quant_strategy_training.py
├── 📁 models/                 # 机器学习模型模块
│   ├── __init__.py
│   ├── ml_model_training.py
│   └── simple_ml_training.py
├── 📁 backtest/               # 回测系统模块
│   ├── __init__.py
│   └── auto_backtest_system.py
└── 📁 utils/                  # 工具函数模块
    └── __init__.py
```

**功能说明**：
- **core**: 核心Web应用和基础功能
- **strategies**: 量化交易策略实现
- **models**: 机器学习模型训练和预测
- **backtest**: 自动回测系统
- **utils**: 通用工具函数

### 2. `config/` - 配置文件
```
config/
├── project_config.py          # 项目主配置
├── environment-mcp.yml        # Conda环境配置
└── environment-mcp-simple.yml # 简化环境配置
```

**功能说明**：
- **project_config.py**: 包含数据配置、模型配置、回测配置等
- **environment-*.yml**: 不同复杂度的conda环境配置

### 3. `scripts/` - 运行脚本
```
scripts/
├── run_backtest.py            # 回测运行脚本
├── run_streamlit.py           # Streamlit启动脚本
├── start_jupyter.sh           # Jupyter启动脚本
├── start_streamlit.sh         # Streamlit启动脚本
└── setup_conda_env.sh         # 环境设置脚本
```

**功能说明**：
- **run_*.py**: Python运行脚本，用于启动不同功能
- **start_*.sh**: Shell启动脚本，用于快速启动服务
- **setup_*.sh**: 环境设置脚本

### 4. `results/` - 结果文件
```
results/
├── auto_backtest_results.html # 回测结果图表
├── auto_backtest_report.txt   # 回测详细报告
├── 基础预测模型_results.html  # 模型训练结果
└── 基础预测模型_report.txt   # 模型训练报告
```

**功能说明**：
- **HTML文件**: 交互式可视化结果
- **TXT文件**: 详细的文本报告

### 5. `docs/` - 文档
```
docs/
├── PROJECT_STRUCTURE.md       # 项目结构说明（本文件）
├── QUANT_TRAINING_GUIDE.md   # 量化训练指南
├── VISUALIZATION_GUIDE.md     # 可视化指南
├── MCP_README.md             # MCP集成说明
└── README.md                 # 项目主文档
```

### 6. `tests/` - 测试文件
```
tests/
├── test_china_stocks.py      # 中国股票测试
├── test_visualization.py     # 可视化测试
├── simple_china_test.py      # 简化中国测试
└── ...                       # 其他测试文件
```

## 🚀 快速使用指南

### 1. 运行回测
```bash
# 激活环境
conda activate qlib-mcp

# 运行回测
python scripts/run_backtest.py
```

### 2. 启动Web应用
```bash
# 启动Streamlit
python scripts/run_streamlit.py

# 或使用shell脚本
./scripts/start_streamlit.sh
```

### 3. 启动Jupyter
```bash
# 启动Jupyter Notebook
./scripts/start_jupyter.sh
```

## 🔧 配置说明

### 主要配置项 (`config/project_config.py`)
- **数据配置**: 股票池、时间范围、数据源
- **模型配置**: 算法参数、特征选择、训练设置
- **回测配置**: 资金管理、信号阈值、风险控制
- **路径配置**: 各模块文件路径
- **日志配置**: 日志级别和格式

## 📊 核心功能模块

### 1. 自动回测系统 (`src/backtest/`)
- 机器学习模型自动生成买卖信号
- 历史数据模拟投资
- 实时计算收益率和风险指标
- 生成详细回测报告和可视化图表

### 2. 机器学习模型 (`src/models/`)
- 特征工程和标签生成
- 模型训练和验证
- 预测信号生成
- 模型性能评估

### 3. Web可视化 (`src/core/`)
- 股票数据实时展示
- 技术指标分析
- 交互式图表
- 策略回测结果展示

## 🎯 项目特点

1. **模块化设计**: 清晰的代码结构，易于维护和扩展
2. **配置驱动**: 通过配置文件控制所有参数
3. **自动化程度高**: 从数据获取到回测结果全自动
4. **可视化丰富**: 多种图表展示方式
5. **易于使用**: 简单的脚本启动，无需复杂配置

## 🔄 开发流程

1. **环境准备**: 使用 `config/environment-*.yml` 创建conda环境
2. **代码开发**: 在 `src/` 目录下开发新功能
3. **测试验证**: 使用 `tests/` 目录下的测试文件验证功能
4. **结果分析**: 在 `results/` 目录查看输出结果
5. **文档更新**: 在 `docs/` 目录更新相关文档

## 📝 注意事项

1. **环境依赖**: 确保使用正确的conda环境 (`qlib-mcp`)
2. **数据路径**: 数据文件路径在配置文件中设置
3. **权限问题**: Shell脚本可能需要执行权限 (`chmod +x`)
4. **端口冲突**: 确保8501端口未被占用
5. **依赖安装**: 首次使用需要安装相关Python包

## 🤝 贡献指南

1. 遵循现有的目录结构
2. 在相应模块下添加新功能
3. 更新相关文档
4. 添加必要的测试用例
5. 保持代码风格一致 