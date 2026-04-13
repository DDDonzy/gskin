import maya.OpenMaya as om  # type:ignore
import functools


class MProfiler:
    """
    基于 Maya Python API 1.0 的性能分析工具。
    """

    # 使用类属性,确保在当前 Maya 会话中同一个 Category 只被注册一次
    _category_id = None
    _category_name = "_User_MProfiler"

    @classmethod
    def get_category(cls):
        """
        获取或注册自定义的 Profiler Category
        Return:
            id (int): 分类id
        """
        if cls._category_id is None:
            # 注册分类：MProfiler.addCategory(名称, 描述)
            cls._category_id = om.MProfiler.addCategory(cls._category_name, "MProfiler for custom tools")
        return cls._category_id

    def __init__(self, event_name=None, color=0):
        """
        初始化 Profiler
        Args:
            event_name (str): 显示在 Profiler 窗口中的事件名称
            color (int | MProfiler.kColor): 在 Profiler 中显示的色块颜色 0-18
        """
        self.event_name = event_name
        self.color = color
        self.scope = None

    def __enter__(self):
        """进入 with 语句块时触发"""
        cat_id = self.get_category()
        # 实例化 MProfilingScope，实例化瞬间即开始计时
        self.scope = om.MProfilingScope(cat_id, self.color, self.event_name, "")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 语句块时触发"""
        # 将 scope 置为 None,触发 C++ 底层的析构函数，从而自动结束计时
        self.scope = None

    def __call__(self, func):
        """允许作为函数装饰器使用"""
        display_name = self.event_name if self.event_name else func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cat_id = self.get_category()
            # 直接在这里实例化 scope 更加清晰且安全
            scope = om.MProfilingScope(cat_id, self.color, display_name, "")
            try:
                return func(*args, **kwargs)
            finally:
                # 确保函数执行完 销毁 scope 结束计时
                scope = None  # noqa: F841

        return wrapper
