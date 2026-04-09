from __future__ import annotations
import maya.OpenMaya as OpenMaya  # type: ignore

__all__ = [
    "DirtyEventHandler",
    "DirtyEventManager",
]


class DirtyEventHandler:
    """
    独立的事件处理单元 (观察者)
    """

    __slots__ = (
        "triggers",    # 监听的属性元组
        "action_func", # 触发的回调
        "_is_dirty",   # 脏状态标记
    ) 

    def __init__(self, triggers, action_func):
        # 强制转换为 tuple，利用 Python 底层的快速 C 遍历与 Maya 底层重载
        if isinstance(triggers, (list, tuple, set)):
            self.triggers = tuple(triggers)
        else:
            self.triggers = (triggers,)

        self._is_dirty = True  # 默认初始化为有待办事项, 强制执行一次
        self.action_func = action_func

    @property
    def is_dirty(self) -> bool:
        """获取当前待处理状态 (只读)"""
        return self._is_dirty

    def set_dirty(self, state: bool = True):
        """主动设置事件的待处理状态"""
        self._is_dirty = state

    def execute(self, dataBlock: OpenMaya.MDataBlock) -> bool:
        """
        执行绑定的回调逻辑。
        警告: 必须接收当前计算周期的 dataBlock，绝对禁止在全局缓存 dataBlock！
        """
        if self._is_dirty:
            self.action_func(dataBlock)  # 将当前周期的 dataBlock 安全传给回调
            self._is_dirty = False  # 执行完毕, 消除待办标记
            return True
        return False


class DirtyEventManager:
    """
    全局事件调度中心
    """

    __slots__ = (
        "_handlers",
        "has_dirty_events",
    )

    def __init__(self):
        self._handlers: list[DirtyEventHandler] = []
        self.has_dirty_events = True

    def add_handler(self, handler: DirtyEventHandler):
        """将一个处理器实例注册到管理器中"""
        self._handlers.append(handler)

    def sync_from_plug(self, plug: OpenMaya.MPlug) -> bool:
        """
        DG 模式: 极速遍历
        利用 MPlug::operator==(const mObject&) 的 C++ 底层重载，
        实现零 Python 对象创建的极速匹配。
        """
        triggered = False

        for handler in self._handlers:
            if not handler.is_dirty and plug in handler.triggers:
                handler.set_dirty(True)
                self.has_dirty_events = True
                triggered = True

        return triggered

    def sync_from_evaluation(self, evaluation_node: OpenMaya.MEvaluationNode) -> bool:
        """
        Parallel 模式: 同步评估节点的脏状态
        """
        for handler in self._handlers:
            if handler.is_dirty:
                continue  # 已经处于待办状态了，提前阻断

            for mObj in handler.triggers:
                if evaluation_node.dirtyPlugExists(mObj):
                    handler.set_dirty(True)
                    self.has_dirty_events = True
                    break

        return self.has_dirty_events

    def execute(self, dataBlock: OpenMaya.MDataBlock):
        """
        清算所有待处理的事件。
        在 Deformer.compute() 中真正变形前调用。
        """
        if not self.has_dirty_events:
            return

        for handler in self._handlers:
            handler.execute(dataBlock)

        self.has_dirty_events = False
