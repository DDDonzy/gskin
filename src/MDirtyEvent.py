from __future__ import annotations

import weakref

import maya.OpenMaya as OpenMaya  # type: ignore

__all__ = [
    "DirtyEvent",
]


class DirtyEvent:
    """
    传入 Attribute MObject 和 对应的事件函数
    通过 sync 检测是否为脏, 通过 execute 判断执行回调函数
    避免代码中有太多了脏判断和执行逻辑

    Example:
    ```
    class MySkinDeformer(OpenMayaMPx.MPxDeformerNode):
        aWeights = OpenMaya.MObject()
        aLayer   = OpenMaya.MObject()

        def __init__(self):
            super().__init__()
            # 1. 初始化并绑定监听的属性与对应的执行动作
            self.weights_event = DirtyEvent(
                triggers=(self.aWeights, self.aLayer),
                functions=self._update_weights_cache
            )

        def _update_weights_cache(self, dataBlock):
            # 当 aWeights 或 aLayer 发生变化时，才会执行到这里
            print("数据已脏，正在重新拉取并更新缓存...")

        def setDependentsDirty(self, plug, dirtyPlugArray):
            # 2. DG模式下：让 handler 自己去嗅探 plug 是否触发了事件
            self.weights_event.sync_from_plug(plug)
            return super().setDependentsDirty(plug, dirtyPlugArray)

        def preEvaluation(self, context, evaluationNode):
            # 3. Parallel模式下：让 handler 去嗅探 evaluationNode
            self.weights_event.sync_from_evaluation(evaluationNode)
            return super().preEvaluation(context, evaluationNode)

        def compute(self, plug, dataBlock):
            if plug == self.aOutputGeometry:
                # 4. 在真正开始 Deform 前，清算并执行所有变脏的事件
                self.weights_event.execute(dataBlock)

            return super().compute(plug, dataBlock)
    ```
    """

    __slots__ = (
        "triggers",  # 监听的属性元组
        "_is_dirty",  # 脏状态标记
        "_weak_functions",  # 触发的回调函数元组
    )

    def __init__(self, triggers, functions):
        # 格式化触发器
        if isinstance(triggers, (list, tuple, set)):
            self.triggers = tuple(triggers)
        else:
            self.triggers = (triggers,)

        # 格式化回调函数，支持传入单个函数或函数列表
        if not isinstance(functions, (list, tuple, set)):
            functions = (functions,)

        weak_functions = []
        for func in functions:
            if hasattr(func, "__self__"):
                # 如果是类的绑定方法 (Bound Method, 例如 self.xxxxx)
                # 必须使用 WeakMethod 包装，否则解包时会出错或失效
                weak_functions.append(weakref.WeakMethod(func))
            else:
                # 如果是普通独立函数 (Function)
                weak_functions.append(weakref.ref(func))

        self._weak_functions = tuple(weak_functions)

        self._is_dirty = True  # 默认初始化为有待办事项, 强制执行一次

    @property
    def is_dirty(self) -> bool:
        """获取当前待处理状态 (只读)"""
        return self._is_dirty

    def set_dirty(self, state: bool = True):
        """主动设置事件的待处理状态"""
        self._is_dirty = state

    def sync_from_plug(self, plug: OpenMaya.MPlug) -> bool:
        """DG 模式: 检测 plug 是否触发了当前处理器的脏状态"""
        if not self._is_dirty and plug in self.triggers:
            self._is_dirty = True
            return True
        return False

    def sync_from_evaluation(self, evaluation_node: OpenMaya.MEvaluationNode) -> bool:
        """Parallel 模式: 检测 evaluation_node 是否触发了当前处理器的脏状态"""
        if self._is_dirty:
            return False  # 提前阻断

        for mObj in self.triggers:
            if evaluation_node.dirtyPlugExists(mObj):
                self._is_dirty = True
                return True
        return False

    def execute(self, *args, **kwargs) -> bool:
        """
        执行绑定的所有回调逻辑。
        接受任意位置参数和关键字参数，直接透传给绑定的回调函数。
        """
        if not self._is_dirty:
            return False

        for weak_func in self._weak_functions:
            # 执行前必须先调用括号 () 唤醒真实的函数对象
            func = weak_func()

            # 安全验证：如果 real_func 不为 None，说明宿主节点依然存活
            if func is not None:
                func(*args, **kwargs)  # 参数并执行
            else:
                # 如果走到这里，说明底层的 cSkinDeform 节点已经被 Maya 删除了。
                # 回调函数会静默失效，绝对不会抛出运行时报错。
                pass

        self._is_dirty = False
        return True
