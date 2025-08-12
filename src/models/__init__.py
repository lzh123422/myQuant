"""
模型模块
"""

# 只导入可用的模块
try:
    from .simple_ml_training import *
except ImportError:
    pass

try:
    from .ml_model_training import *
except ImportError:
    pass 