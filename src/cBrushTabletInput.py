"""
======================================================================================
Module: cTabletInput
======================================================================================
数位板压感拦截器。
======================================================================================
"""

from __future__ import annotations

import time
from PySide2 import QtCore, QtWidgets

class _TabletEventFilter(QtCore.QObject):
    """底层的 Qt 事件过滤器"""

    def __init__(self, tracker: TabletTracker):
        super().__init__()
        self._tracker: TabletTracker = tracker
        self._last_tablet_event_time = 0.0

    def eventFilter(self, obj, event):
        ev_type = event.type()

        # 1. 优先处理并拦截真实的手绘板事件
        if ev_type in (QtCore.QEvent.TabletMove, QtCore.QEvent.TabletPress):
            self._tracker.pressure = event.pressure()
            # 使用 perf_counter 获得最高精度的相对时间
            self._last_tablet_event_time = time.perf_counter() 
            return False

        # 2. 过滤掉手绘板“伪造”的鼠标事件
        if ev_type in (QtCore.QEvent.MouseMove, QtCore.QEvent.MouseButtonPress):
            current_time = time.perf_counter()
            time_delta = current_time - self._last_tablet_event_time

            # 0.1秒 (100ms) 内的鼠标事件视为伪造，直接无视，保护压感
            if time_delta < 0.1:  
                return False

            # 超过 0.1 秒，确认为纯物理鼠标，强制重置压感为 1.0
            self._tracker.pressure = 1.0  
            return False

        # 3. 释放事件清理
        if ev_type in (QtCore.QEvent.TabletRelease, QtCore.QEvent.MouseButtonRelease):
            self._tracker.pressure = 1.0

        # 返回 False 确保 Maya 自身的视图导航（如 Alt+拖拽）不会被阻断
        return False


class TabletTracker:
    """
    数位板压感追踪器 (对外暴露的公共 API)。
    """

    def __init__(self):
        self.pressure: float = 1.0
        self._filter: _TabletEventFilter | None = None

    def start(self):
        """挂载拦截器，开始采集压感"""
        if self._filter is None:
            self._filter = _TabletEventFilter(self)
            app = QtWidgets.QApplication.instance()
            if app:
                app.installEventFilter(self._filter)

    def stop(self):
        """卸载拦截器，释放内存"""
        if self._filter is not None:
            app = QtWidgets.QApplication.instance()
            if app:
                app.removeEventFilter(self._filter)
            self._filter = None  # 释放引用，允许 Python 进行垃圾回收


if __name__ == "__main__":
    test_tracker = TabletTracker()
    test_tracker.start()
