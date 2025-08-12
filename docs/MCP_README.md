# Qlib MCP (Model Context Protocol) 集成

这个项目为Qlib量化投资平台集成了MCP协议，让AI助手能够直接操作量化投资工作流。

## ⚠️ 重要提示

**强烈建议使用conda环境**来运行Qlib和MCP服务器。根据官方README，在非conda环境中可能会缺少必要的头文件，导致某些包安装失败。

## 功能特性

### 可用的MCP工具

1. **initialize_qlib** - 初始化Qlib配置
2. **get_stock_data** - 获取股票数据
3. **run_backtest** - 运行回测
4. **train_model** - 训练机器学习模型
5. **get_portfolio_analysis** - 获取投资组合分析
6. **list_available_models** - 列出可用模型
7. **get_workflow_status** - 获取工作流状态

## 安装步骤

### 方法1: 使用conda环境（推荐）

#### 1. 创建conda环境

```bash
# 创建新的conda环境
conda env create -f environment-mcp.yml

# 激活环境
conda activate qlib-mcp
```

#### 2. 安装Qlib

```bash
# 从源码安装Qlib（推荐用于开发）
pip install -e .

# 或者使用pip安装稳定版
pip install pyqlib
```

#### 3. 验证安装

```bash
python test_mcp.py
```

### 方法2: 手动安装（不推荐）

#### 1. 安装MCP依赖

```bash
pip install -r requirements-mcp.txt
```

#### 2. 安装其他必要依赖

```bash
pip install redis redis-lock
```

## 配置MCP服务器

### 1. 复制配置文件

将 `mcp_config_example.json` 复制到你的MCP客户端配置目录，或者根据你的MCP客户端要求调整配置。

### 2. 启动MCP服务器

```bash
# 在conda环境中
conda activate qlib-mcp
python start_mcp_server.py
```

## 使用示例

### 初始化Qlib

```json
{
  "tool": "initialize_qlib",
  "arguments": {
    "provider_uri": "~/.qlib/qlib_data/cn_data",
    "region": "cn"
  }
}
```

### 获取股票数据

```json
{
  "tool": "get_stock_data",
  "arguments": {
    "symbols": ["000001.SZ", "000002.SZ"],
    "start_time": "2023-01-01",
    "end_time": "2023-12-31",
    "fields": ["$close", "$volume", "$factor"]
  }
}
```

### 运行回测

```json
{
  "tool": "run_backtest",
  "arguments": {
    "strategy": "TopkDropoutStrategy",
    "benchmark": "000300.SH",
    "start_time": "2023-01-01",
    "end_time": "2023-12-31",
    "topk": 50
  }
}
```

### 训练模型

```json
{
  "tool": "train_model",
  "arguments": {
    "model_name": "LGBModel",
    "dataset": "Alpha158",
    "feature_columns": ["$close", "$volume", "$factor"],
    "label_columns": ["Ref($close, -1)/$close - 1"]
  }
}
```

## 配置说明

### MCP配置文件

- `mcp_config.json` - 标准MCP配置
- `mcp_config_example.json` - 配置示例
- `environment-mcp.yml` - conda环境配置

### 环境变量

- `PYTHONPATH` - Python路径设置
- `QLIB_DATA_PATH` - Qlib数据路径

## 故障排除

### 常见问题

1. **导入错误**: 确保在正确的conda环境中运行
2. **路径问题**: 检查PYTHONPATH设置
3. **权限问题**: 确保有足够权限访问数据目录
4. **依赖缺失**: 使用conda环境可以避免大部分依赖问题

### 日志

MCP服务器会输出详细日志，帮助诊断问题。

## 快速启动

使用提供的脚本快速启动：

```bash
# 在conda环境中
conda activate qlib-mcp
./quick_start_mcp.sh
```

## 扩展开发

### 添加新工具

在 `qlib/mcp/server.py` 中的 `_register_tools()` 方法中添加新工具定义。

### 自定义配置

修改 `QlibMCPServer` 类来添加自定义功能和配置。

## 支持

如有问题，请查看：
- Qlib官方文档: https://qlib.readthedocs.io/
- MCP协议文档: https://modelcontextprotocol.io/
- 使用conda环境可以避免大部分安装和依赖问题 