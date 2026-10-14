"""
pytest 的根 conftest —— 唯一作用是让 `from app.xxx import ...` 能解析。

项目代码用的是绝对导入（`from app.import_process...`），
所以测试跑起来时必须把【项目根】放进 sys.path。
放在根目录的 conftest.py 会在 pytest 启动时最早被加载。
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
