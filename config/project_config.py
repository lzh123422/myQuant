#!/usr/bin/env python3
"""
myQuant 项目配置文件
"""

# 项目基本信息
PROJECT_NAME = "myQuant"
VERSION = "1.0.0"
DESCRIPTION = "全自动量化投资系统"

# 数据配置
DATA_CONFIG = {
    "provider_uri": "~/.qlib/qlib_data/cn_data",
    "region": "cn",
    "default_symbols": [
        "SH600000", "SH600004", "SH600009", "SH600010", "SH600011",
        "SH600016", "SH600019", "SH600025", "SH600027", "SH600028"
    ],
    "default_start_time": "2019-01-01",
    "default_end_time": "2020-09-25"
}

# 模型配置
MODEL_CONFIG = {
    "default_model": "RandomForestClassifier",
    "model_params": {
        "n_estimators": 100,
        "random_state": 42,
        "max_depth": 10
    },
    "feature_columns": [
        "close", "volume", "open", "high", "low", "factor",
        "price_change", "price_change_2d", "price_change_5d",
        "ma_5", "ma_10", "ma_20", "price_position",
        "price_ma5_ratio", "price_ma20_ratio", "volume_ma5",
        "volume_ratio", "volatility_5d", "volatility_10d"
    ]
}

# 回测配置
BACKTEST_CONFIG = {
    "default_initial_capital": 1000000,  # 100万
    "max_position_per_stock": 0.1,      # 单只股票最多10%
    "signal_threshold": 0.6,            # 信号置信度阈值
    "forward_days": 5                   # 预测未来天数
}

# 路径配置
PATHS = {
    "models": "models/",
    "results": "results/",
    "data": "data/",
    "logs": "logs/"
}

# 日志配置
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/myquant.log"
} 