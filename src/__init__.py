"""
myQuant 量化投资系统
"""

__version__ = "1.0.0"
__author__ = "myQuant Team"
__description__ = "全自动量化投资系统"

# 使用try-except避免导入错误
try:
    from .core import *
except ImportError:
    pass

try:
    from .strategies import *
except ImportError:
    pass

try:
    from .models import *
except ImportError:
    pass

try:
    from .backtest import *
except ImportError:
    pass

try:
    from .utils import *
except ImportError:
    pass 