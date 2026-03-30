"""
======================================================================================
Module: cTabletInput
======================================================================================
数位板压感拦截器。
======================================================================================
"""

from __future__ import annotations

from PySide2 import QtCore, QtWidgets


class _TabletEventFilter(QtCore.QObject):
    """底层的 Qt 事件过滤器"""

    def __init__(self, tracker: TabletTracker):
        super().__init__()
        self._tracker: TabletTracker = tracker
        self._pen_in_proximity = False  # 追踪笔尖是否在数位板感应范围内

    def eventFilter(self, obj, event):
        ev_type = event.type()

        # 极简感应：毫无计算压力，耗时 < 0.0001ms
        if ev_type == QtCore.QEvent.TabletEnterProximity:
            self._pen_in_proximity = True
        elif ev_type == QtCore.QEvent.TabletLeaveProximity:
            self._pen_in_proximity = False
            self._tracker.pressure = 1.0  # 恢复满力

        # 极简赋值：只要在板子上，就无脑更新数值，不做任何其他判定
        if self._pen_in_proximity and ev_type in (QtCore.QEvent.TabletMove, QtCore.QEvent.TabletPress):
            self._tracker.pressure = event.pressure()

        return False  # 永远放行，绝对不卡 Maya


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
