from __future__ import annotations
import maya.OpenMaya as OpenMaya  # type: ignore


class DirtyHandler:
    """
    独立的脏数据处理单元 (实例)
    """

    __slots__ = (
        "name",
        "triggers",
        "is_dirty",
        "action_func",
    )

    def __init__(self, name: str, triggers, action_func):
        self.name = name
        if isinstance(triggers, tuple):
            self.triggers = triggers
        elif isinstance(triggers, list):
            self.triggers = tuple(triggers)
        else:
            self.triggers = (triggers,)
        self.is_dirty = True  # 默认初始化为脏, 强制执行一次
        self.action_func = action_func  # 当状态为脏时, 需要执行的回调函数

    def execute(self) -> bool:
        """执行绑定的逻辑, 并自动清理状态"""
        if self.is_dirty:
            self.action_func()  # 执行真正的更新逻辑
            self.is_dirty = False  # 执行完毕, 重置状态
            return True
        return False


class DirtyManager:
    """
    基于 Handler 实例的主动任务调度器
    """

    def __init__(self):
        self._handlers: list[DirtyHandler] = []
        self.global_dirty = True

    def add_handler(self, handler: DirtyHandler):
        """将一个处理器实例注册到管理器中"""
        self._handlers.append(handler)

    def sync_from_plug(self, plug: OpenMaya.MPlug) -> bool:
        """DG 模式:极速遍历所有 handler 的触发器"""
        triggered = False
        for handler in self._handlers:
            if plug in handler.triggers:
                handler.is_dirty = True
                self.global_dirty = True
                triggered = True
        return triggered

    def sync_from_evaluation(self, evaluation_node: OpenMaya.MEvaluationNode) -> bool:
        """Parallel 模式:同步脏状态"""
        for handler in self._handlers:
            if handler.is_dirty:
                continue  # 提前阻断

            for mObj in handler.triggers:
                if evaluation_node.dirtyPlugExists(mObj):
                    handler.is_dirty = True
                    self.global_dirty = True
                    break
        return self.global_dirty

    def execute_all_dirty(self):
        """
        遍历所有 handler, 脏了的就执行它绑定的函数。
        """
        if not self.global_dirty:
            return

        for handler in self._handlers:
            handler.execute()

        self.global_dirty = False
