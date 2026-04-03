from __future__ import annotations


import typing
import array
import functools
import threading
import contextlib
from collections import deque


from maya import cmds
import maya.OpenMaya as om1  # type:ignore


from . import apiundo
from . import cBrushCore2Cython as cBrushCoreCython
from ._cRegistry import SkinRegistry
from .cBufferManager import BufferManager


if typing.TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer  # type: ignore


class StrokeParameters:
    __slot__ = (
        "brush_mode",
        "weights_value",
        "influences_indices",
        "pressure",
        "clamp_min",
        "clamp_max",
        "iterations",
    )
    brush_mode: int
    weights_value: array.array
    influences_indices: array.array
    pressure: float
    clamp_min: float
    clamp_max: float
    iterations: int

    def __init__(
        self,
        brush_mode,
        weights_value,
        influences_indices,
        pressure,
        clamp_min=0.0,
        clamp_max=1.0,
        iterations=1,
    ):
        self.brush_mode = brush_mode
        self.weights_value = weights_value
        self.influences_indices = influences_indices
        self.pressure = pressure
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.iterations = iterations


# ==============================================================================
# ⏱️ 任务调度组件 (Composition Component)
# ==============================================================================
class DeferredTaskManager:
    """
    专为 Maya 节点计算周期设计的任务调度器。
    提供线程安全的队列管理和执行防抖。
    """

    def __init__(self, update_callback: typing.Callable[[], None]):
        self._deferred_tasks = deque()
        self._tasks_lock = threading.RLock()
        self._is_dg_updating = False

        # 绑定的回调函数，用于唤醒 Maya 节点 (通常是 WeightsManager.updateDG)
        self._update_callback = update_callback

    def add_task(self, task: typing.Callable):
        """推入队列并尝试唤醒节点"""
        with self._tasks_lock:
            self._deferred_tasks.append(task)
        if self._update_callback:
            self._update_callback()

    def execute_tasks(self):
        """消化队列中的所有任务"""
        if not self._deferred_tasks:
            return
        with self._tasks_lock:
            while self._deferred_tasks:
                task = self._deferred_tasks.popleft()
                task()

    @contextlib.contextmanager
    def update_dg_context(self):
        """防抖锁上下文：确保只有最外层嵌套执行结束后，才触发一次 Maya 刷新"""
        is_top_level = not self._is_dg_updating
        if is_top_level:
            self._is_dg_updating = True
        try:
            yield
        finally:
            if is_top_level:
                self._is_dg_updating = False
                if self._update_callback:
                    self._update_callback()


# ==============================================================================
# 🍬 调度器专属装饰器 (Syntactic Sugar)
# ==============================================================================
def defer_task(func):
    """将被装饰的方法打包为延迟任务，交给实例的 task_manager 处理"""

    @functools.wraps(func)
    def wrapper(self: WeightsManager, *args, **kwargs):
        task = functools.partial(func, self, *args, **kwargs)
        self.task_manager.add_task(task)  # 🌟 直接调用组合组件

    return wrapper


def update_dg(func):
    """为方法包裹防抖锁"""

    @functools.wraps(func)
    def wrapper(self: WeightsManager, *args, **kwargs):
        with self.task_manager.update_dg_context():  # 🌟 调用组合组件的上下文
            return func(self, *args, **kwargs)

    return wrapper


class WeightsHandle:
    """
    权重数据装配器 (MVectorArray 0拷贝黑客版)

    架构方案:
    1. 物理伪装:将 32-bit Float 强行塞入 64-bit MVectorArray 中。
    2. 原地扩容:通过 MVectorArray.setLength() 触发底层 C++ realloc,避免频繁销毁重建 MObject。
    3. 内存布局:
        ```
        ----------------------------------------------------------------------------------
        | vtx_count (int) | inf_count (int) | bone_indices... (int) | Weights... (float) |
        | <------------------ Header -----------------------------> | <--- Payload ----> |
        ----------------------------------------------------------------------------------
        ```
    """

    # fmt:off
    mDataHandle  : om1.MDataHandle  # 变形器计算周期内的临时句柄
    mPlug        : om1.MPlug        # 节点插座,用于数据持久化
    mObject_data : om1.MObject      # 包装了 VectorArray 的 Maya 数据实体
    mVectorArray : om1.MFnVectorArrayData # 指向 MObject 内部缓冲区的引用

    memory        : BufferManager    # 权重数据的底层裸指针映射 (纯 Float 视图)
    weights_memory: BufferManager    # 仅权重部分视图
    max_capacity  : int              # 当前物理内存支持的最大 Float 存储量 (VectorLength * 6)
    length        : int              # 当前逻辑数据的有效 Float 长度  
    # fmt:on

    def __init__(self, mPlug: om1.MPlug, mDataHandle: om1.MDataHandle):
        """
        初始化装配器。
        注:由于 nodeInitializer 设置了默认值,此处假定 mDataHandle 必定包含合法的 VectorArray。
        """
        # fmt:off
        self.mPlug        = mPlug
        self.mDataHandle  = mDataHandle

        self.mObject_data   = None
        self.mVectorArray   = None
        self.memory         = None
        self.weights_memory = None
        self.max_capacity   = -1
        self.length         = -1
        # fmt:on

        self._setup_vector_buffer(self.mDataHandle)

    @classmethod
    def from_attr_string(cls, attr_path: str):
        """工具方法:从字符串路径("node.attr")快速构建"""
        sel = om1.MSelectionList()
        try:
            sel.add(attr_path)
        except RuntimeError as e:
            raise ValueError(f"Attribute path not found: {attr_path}") from e
        plug = om1.MPlug()
        sel.getPlug(0, plug)
        # 注意:外部调用时需自行提供对应的 DataHandle 或是通过 plug.asMDataHandle() 获取
        return cls(plug, plug.asMDataHandle())

    def _setup_vector_buffer(self, mDataHandle: om1.MDataHandle):
        """
        从 MDataHandle 解析 MVectorArray

        Updates:
            - `self.mObject_data`
            - `self.mVectorArray`
            - `self.length`
            - `self.max_capacity`
            - `self.memory`
        """
        self.mObject_data = mDataHandle.data()
        if self.mObject_data.isNull():
            # om1.MGlobal.displayError(f"{self.mPlug.name()}'s dataHandle:{mDataHandle} is Null")
            return

        # 获取 MObject 内部数组的引用 (并非拷贝)
        fn_vector_data = om1.MFnVectorArrayData(self.mObject_data)
        self.mVectorArray = fn_vector_data.array()
        self.max_capacity = self.mVectorArray.length() * 6

        # 刷新物理内存指针映射
        self._remap_memory()

    def _remap_memory(self):
        """
        [物理层映射] 将底层 C++ VectorArray 内存地址映射为 Python 连续视图。
        此函数仅客观圈定物理地盘,不负责数据安全性校验 (校验交由 is_valid 处理)。

        Updates:
            - `self.length`
            - `self.memory`
            - `self.weights_memory` (带有 physical padding,用于 fill 格式化)
        """
        if self.mVectorArray is None:
            self.memory = None
            self.weights_memory = None
            return

        if self.mVectorArray.length() <= 0:
            self.memory = None
            self.weights_memory = None
            return

        _ptr = int(self.mVectorArray[0].this)
        self.memory = BufferManager.from_ptr(_ptr, "f", (self.max_capacity,))

        _int_view = self.memory.view.cast("B").cast("i")
        vtx_count = _int_view[0]
        influence_count = _int_view[1]

        self.length = (2 + influence_count) + (vtx_count * influence_count)

        if influence_count >= 0:
            self.weights_memory = self.memory.slice(start=2 + influence_count)
        else:
            self.weights_memory = None

    def resize(self, vtx_count: int, inf_count: int):
        """
        根据顶点和骨骼数量,原地扩容底层物理内存,并自动格式化为安全状态。

        Updates:
            - `self.length`
            - `self.memory`
            - `self.weights_memory`
        """
        if self.is_null:
            ""
            init_ary = om1.MVectorArray()
            init_obj = om1.MFnVectorArrayData().create(init_ary)
            self.mVectorArray = init_ary
            self.mObject_data = init_obj
            self.mDataHandle.setMObject(init_obj)
            self._setup_vector_buffer(self.mDataHandle)

        required_length = (2 + inf_count) + (vtx_count * inf_count)

        # 1. 如果容量足够,复用内存
        if required_length <= self.max_capacity:
            self.length = required_length

            _ptr = int(self.mVectorArray[0].this)
            _header = BufferManager.from_ptr(_ptr, "i", (2,))
            _header.view[0] = vtx_count
            _header.view[1] = inf_count

            self._remap_memory()

            # 🚀 格式化复用内存的负载区,清除上一次图层遗留的错位旧数据
            if self.weights_memory:
                self.weights_memory.fill(0.0)

            return False

        # 2. 如果容量不足,触发底层物理扩容
        vector_count = (required_length + 5) // 6
        self.mVectorArray.setLength(vector_count)
        self.max_capacity = vector_count * 6

        # 3. 注入合法表头
        _ptr = int(self.mVectorArray[0].this)
        _header = BufferManager.from_ptr(_ptr, "i", (2,))
        _header.view[0] = vtx_count
        _header.view[1] = inf_count

        # 4. 重新映射,切出完美的 weights_memory
        self._remap_memory()

        # 🚀 格式化新分配的物理内存,彻底绞杀底层 realloc 产生的 NaN 垃圾数据
        if self.weights_memory:
            self.weights_memory.fill(0.0)

        return True

    def parse_raw_weights(self, raw_view=None):
        """
        解析符合内存布局协议的连续视图。

        Args:
            raw_view: 如果提供,则解析传入的视图 (用于 Undo 备份还原)
                            如果为 None,则默认解析自身的 self.memory.view (用于内部重映射)
        """
        if raw_view is None:
            if self.memory is None or self.memory.view is None:
                return 0, 0, None, None
            raw_view = self.memory.view

        if len(raw_view) < 2:
            return 0, 0, None, None

        int_view = raw_view.cast("B").cast("i")
        vtx_count = int_view[0]
        influence_count = int_view[1]

        if vtx_count <= 0 or influence_count <= 0:
            return 0, 0, None, None

        required_elements = (2 + influence_count) + (vtx_count * influence_count)

        if required_elements > len(raw_view):
            return 0, 0, None, None

        header_size = 2 + influence_count
        influence_indices_view = int_view[2:header_size]
        weights_view = raw_view[header_size:required_elements]

        return vtx_count, influence_count, influence_indices_view, weights_view

    def clear(self):
        """
        重置数据为默认空壳状态。
        :param shrink: 是否强制释放物理内存。如果为 False,则只做极速的逻辑清零。
        """
        if self.mVectorArray is None:
            return False

        self.mVectorArray.setLength(1)
        self.max_capacity = 6

        self.resize(0, 0)

        return True

    @property
    def is_null(self) -> bool:
        """
        物理级检查:是否完全没有绑定 Maya 数据实体。
        (通常发生在新节点刚创建,或 plug 断开连接时)
        """
        return self.mObject_data is None or self.mObject_data.isNull() or self.mVectorArray is None

    @property
    def is_empty(self) -> bool:
        """
        业务级检查:是否是一个合法的“空图层”。
        (没有顶点,或没有分配任何骨骼,属于正常业务状态)
        """
        if self.is_null:
            return True
        v_cnt = getattr(self, "vtx_count", 0)
        i_cnt = getattr(self, "influence_count", 0)
        return v_cnt == 0 or i_cnt == 0

    @property
    def is_corrupted(self) -> bool:
        """
        安全级检查:物理内存是否被脏数据污染,或存在越界风险。
        """
        if self.is_null:
            return False  # 什么都没有,也就无所谓损坏

        vtx_count = getattr(self, "vtx_count", 0)
        inf_count = getattr(self, "influence_count", 0)

        # 1. 负数拦截 (绝对的 C++ 脏内存现象)
        if vtx_count < 0 or inf_count < 0:
            return True

        # 2. 物理越界拦截 (拦截几十亿的天文数字)
        required_floats = (2 + inf_count) + (vtx_count * inf_count)
        return required_floats > self.max_capacity

    @property
    def is_valid(self) -> bool:
        """
        综合通行证:句柄是否处于绝对安全、可读写的健康状态。
        """
        return not self.is_null and not self.is_corrupted

    @property
    def view(self):
        """
        [语法糖] 安全获取底层的全量物理内存视图 (含 Header 和 Padding)。
        自带判空防御,用于底层数据的整体 Copy 或 Undo 备份。
        """
        return self.memory.view if self.memory is not None else None

    @property
    def weights_view(self):
        """
        [语法糖] 安全获取纯净的权重负载区视图 (去除了 Header)。
        自带判空防御,用于直接格式化 (fill) 或快速读取。
        """
        return self.weights_memory.view if self.weights_memory is not None else None


class WeightsLayerItem:
    def __init__(self, cSkin: CythonSkinDeformer, mDataHandle: om1.MDataHandle, logical_idx: int = -1):
        self.cSkin = cSkin
        self.logical_idx = logical_idx

        _handle_weights = mDataHandle.child(cSkin.aLayerWeights)
        _handle_mask = mDataHandle.child(cSkin.aLayerMask)
        _handle_enabled = mDataHandle.child(cSkin.aLayerEnabled)
        _handle_name = mDataHandle.child(cSkin.aLayerName)

        self.enabled = _handle_enabled.asBool()
        self.weights = WeightsHandle(self.mPlug_weights, _handle_weights)
        self.mask = WeightsHandle(self.mPlug_mask, _handle_mask)
        self.name = _handle_name.asString()

    # ==========================================
    # 安全的属性获取:直接找 cSkin 要 MObject,完全不碰短命的 MDataHandle
    # ==========================================
    @property
    def mPlug_weights(self):
        # 直接使用 self.cSkin.aLayerWeights 这个永恒不变的属性指针
        mPlug = om1.MPlug(self.cSkin.mObject, self.cSkin.aLayerWeights)
        mPlug.selectAncestorLogicalIndex(self.logical_idx, self.cSkin.aLayerCompound)
        return mPlug

    @property
    def mPlug_mask(self):
        mPlug = om1.MPlug(self.cSkin.mObject, self.cSkin.aLayerMask)
        mPlug.selectAncestorLogicalIndex(self.logical_idx, self.cSkin.aLayerCompound)
        return mPlug

    @property
    def mPlug_enabled(self):
        mPlug = om1.MPlug(self.cSkin.mObject, self.cSkin.aLayerEnabled)
        mPlug.selectAncestorLogicalIndex(self.logical_idx, self.cSkin.aLayerCompound)
        return mPlug

    @property
    def mPlug_name(self):
        mPlug = om1.MPlug(self.cSkin.mObject, self.cSkin.aLayerName)
        mPlug.selectAncestorLogicalIndex(self.logical_idx, self.cSkin.aLayerCompound)
        return mPlug

    def set_name(self, new_name: str):
        self.mPlug_name.setString(new_name)
        self.name = new_name


class WeightsManager:
    def __init__(self, cSkin: CythonSkinDeformer):
        super().__init__()

        self.cSkin = cSkin
        self.mObject_node = cSkin.mObject
        self.mFnDepend_node = cSkin.mFnDep

        self.weights: WeightsHandle = None
        self.layers: dict[int, WeightsLayerItem] = {}

        self.plug_refresh: om1.MPlug = cSkin.plug_refresh
        self.plug_weights: om1.MPlug = om1.MPlug(cSkin.mObject, cSkin.aWeights)

        # =========================================================
        # 笔刷引擎的共享物理内存池 (Object Pool)
        # 寿命与节点等长，避免鼠标按下时疯狂申请内存引发 GC 卡顿
        # =========================================================
        self._pool_vtx_idx: BufferManager = None
        self._pool_vtx_bool: BufferManager = None
        self._pool_inf_locks: BufferManager = None
        self._pool_undo_buffer: BufferManager = None
        #
        self.task_manager = DeferredTaskManager(self.updateDG)

    @property
    def layer_indices(self) -> list[int]:
        return list(self.layers.keys())

    @classmethod
    def from_node(cls, node_name: str):
        cSkin: CythonSkinDeformer = SkinRegistry.from_instance_by_string(node_name)
        return cSkin.weights_manager

    def sync_layer_cache(self, mDataBlock: om1.MDataBlock):
        """
        [状态同步器]
        一次性扫描 Maya 节点,刷新所有 Plug 缓存与底层内存池。
        当你在 UI 层面添加、删除了图层,或改变了节点连接后,手动调用此函数。
        """
        # weights
        weights_dataHandle = mDataBlock.outputValue(self.cSkin.aWeights)
        self.weights = WeightsHandle(self.plug_weights, weights_dataHandle)

        # layer
        self.layers.clear()
        mArrayDataHandle: om1.MArrayDataHandle = mDataBlock.outputArrayValue(self.cSkin.aLayerCompound)
        for idx in range(mArrayDataHandle.elementCount()):
            mArrayDataHandle.jumpToArrayElement(idx)
            logical_idx = mArrayDataHandle.elementIndex()
            element_handle: om1.MDataHandle = mArrayDataHandle.outputValue()

            self.layers[logical_idx] = WeightsLayerItem(self.cSkin, element_handle, logical_idx)

    def updateDG(self):
        self.cSkin.setDirty()

    def get_layer(self, index: int, logicalIndex: bool = True) -> WeightsLayerItem:
        """
        获取图层实例,支持通过逻辑索引或物理索引进行查询。

        Args:
            index (int): 图层的索引值
            logicalIndex (bool): True 表示按逻辑索引查询,False 表示按物理顺序查询。
        Returns:
            WeightsLayerItem: 找到的图层实例,如果越界或找不到则返回 None。
        """
        if logicalIndex:
            return self.layers.get(index, None)
        try:
            return list(self.layers.values())[index]
        except IndexError:
            return None

    def get_handle(self, layer_logical_idx, is_mask: bool = False) -> WeightsHandle:
        """
        严格根据逻辑索引寻找对应的 Handle
        """
        # 基础权重
        if layer_logical_idx == -1:
            if is_mask:
                return None
            return self.weights
        # layer 权重
        if layer_logical_idx in self.layers:
            layer_item = self.layers[layer_logical_idx]
            if is_mask:
                return layer_item.mask
            return layer_item.weights

        return None

    def get_weights(
        self,
        layer_idx: int,
        is_mask: bool,
        vtx_indices=None,
        bone_local_indices=None,
    ):
        """
        [提取/复制权重] (统一高级接口)
        提取指定范围的权重数据,并返回完整的上下文。
        - 全量提取时:直接底层内存拷贝,实现极致零延迟。
        - 局部提取时:调度 Cython 引擎进行极速精细抠取。
        """
        # 1. 直接获取句柄并进行绝对安全拦截
        handle = self.get_handle(layer_idx, is_mask)
        if handle is None or not handle.is_valid:
            return 0, 0, array.array("i"), array.array("i"), array.array("i"), array.array("f")

        # 2. 🚀 让句柄自己解析,直接拿到完美的 1D 权重视图
        v_count, i_count, g_bones_view, w_1d_view = handle.parse_raw_weights()

        # 拦截空图层
        if v_count <= 0 or i_count <= 0 or w_1d_view is None:
            return 0, 0, array.array("i"), array.array("i"), array.array("i"), array.array("f")

        # 提取全局骨骼 ID 备份
        global_bone_ids = array.array("i", g_bones_view) if g_bones_view else array.array("i")

        # ---------------------------------------------------------------------
        # 🚀 性能分支:全量提取 vs 局部提取
        # ---------------------------------------------------------------------
        if vtx_indices is None and bone_local_indices is None:
            # 【分支 A:全量提取】(如 Undo 备份)
            # 既然 handle 已经切出了完美的 w_1d_view,直接 C 级拷贝,跳过引擎启动
            weights_1d = array.array("f", w_1d_view)
            out_vtx = array.array("i", range(v_count))
            out_bone = array.array("i", range(i_count))
        else:
            # 【分支 B:局部提取】(如笔刷吸色 / 局部修改备份)
            # 装配引擎,让 C++ 去做跳跃式的内存抓取
            processor = self._create_processor(layer_idx, is_mask)
            if not processor:
                return v_count, i_count, global_bone_ids, array.array("i"), array.array("i"), array.array("f")

            v_view = BufferManager.auto(vtx_indices, "i").view if vtx_indices is not None else None
            b_view = BufferManager.auto(bone_local_indices, "i").view if bone_local_indices is not None else None

            # Cython 极速返回局部 1D 纯净数据
            weights_1d = processor.get_custom_array(v_view, b_view)

            # 补全上下文索引
            out_vtx = array.array("i", range(v_count)) if vtx_indices is None else array.array("i", v_view)
            out_bone = array.array("i", range(i_count)) if bone_local_indices is None else array.array("i", b_view)

        return v_count, i_count, global_bone_ids, out_vtx, out_bone, weights_1d

    def _create_processor(self, layer_idx: int, is_mask: bool):
        """
        创建一个 Cython 笔刷引擎实例,并传入完美复用的共享内存池视图。
        用户可以根据这个实例进行高性能的局部权重提取或修改。

        Example:
        ```python
            with self._processor_session(layer_idx, is_mask) as processor:
                if processor:
                    processor.get_custom_array(vertex_indices=..., channel_indices=...)
                    processor.set_custom_array( source_values=..., blend_mode=..., vertex_indices=..., channel_indices=..., alpha=..., falloff_weights=... )
                    processor.normalize_weights(vertex_indices=..., priority_influence=...)
        ```
        """
        handle = self.get_handle(layer_idx, is_mask)

        if handle is None or not handle.is_valid:
            return None

        vtx_count, inf_count, _, weights_1d = handle.parse_raw_weights()

        if vtx_count <= 0 or inf_count <= 0 or weights_1d is None:
            return None

        weights_2d = weights_1d.cast("B").cast("f", (vtx_count, inf_count))

        # =========================================================
        # 只在第一次，或者顶点数/骨骼数变大时才去申请新内存！
        # =========================================================
        # 1. 顶点索引池 (vtx_count)
        if not self._pool_vtx_idx or self._pool_vtx_idx.shape[0] < vtx_count:
            self._pool_vtx_idx = BufferManager.allocate("i", (vtx_count,))

        # 2. 顶点布尔防重录池 (vtx_count)
        if not self._pool_vtx_bool or self._pool_vtx_bool.shape[0] < vtx_count:
            self._pool_vtx_bool = BufferManager.allocate("B", (vtx_count,))

        # 3. 骨骼锁定池 (inf_count)
        if not self._pool_inf_locks or self._pool_inf_locks.shape[0] < inf_count:
            self._pool_inf_locks = BufferManager.allocate("B", (inf_count,))

        # 4. Undo 2D 矩阵池 (vtx_count * inf_count)
        req_undo_elements = vtx_count * inf_count
        # 计算当前物理池的真实总容量
        curr_capacity = 0
        if self._pool_undo_buffer:
            if len(self._pool_undo_buffer.shape) == 2:  # noqa: SIM108
                curr_capacity = self._pool_undo_buffer.shape[0] * self._pool_undo_buffer.shape[1]
            else:
                curr_capacity = self._pool_undo_buffer.shape[0]

        # 保证底层物理池足够大 (统一用 1D 存储以保留最大容量)
        if curr_capacity < req_undo_elements:
            self._pool_undo_buffer = BufferManager.allocate("f", (req_undo_elements,))

        # 🌟 核心修复：从物理池中切出精确的尺寸，然后再转 2D
        # 注意：这里我们产生一个局部变量 _undo_2d_view 传给引擎，绝不覆盖 self._pool_undo_buffer
        _flat_view = self._pool_undo_buffer.view.cast("B").cast("f")  # 强制统一为 1D 浮点视图
        _exact_slice = _flat_view[:req_undo_elements]  # 精确切取本次所需的数据量
        _undo_2d_view = _exact_slice.cast("B").cast("f", shape=(vtx_count, inf_count))  # 完美 2D 塑形

        # =========================================================

        # 组装引擎，传入复用的内存池视图
        return cBrushCoreCython.SkinWeightProcessor(
            self.cSkin.brush_engine,
            weights_2d,
            self._pool_vtx_idx.view,
            self._pool_vtx_bool.view,
            self._pool_inf_locks.view,
            _undo_2d_view,  # 🌟 传入精确塑形后的 2D 视图
        )

    @contextlib.contextmanager
    def processor_session(self, layer_idx: int, is_mask: bool, backup: bool = True):
        """
        [引擎上下文管理器]
        统管 Cython 算力引擎的获取、快照录制与撤销栈注册。
        """
        # 1. 尝试获取算力引擎
        processor = self._create_processor(layer_idx, is_mask)

        # 拦截空图层或坏句柄
        if not processor:
            yield None
            return

        # 2. 开启录制 (Setup)
        if backup:
            processor.begin_stroke()

        try:
            # 3. 将引擎实例递交给业务代码使用
            yield processor

        finally:
            # 4. 收尾工作 (Teardown):确保快照闭合与闭包注册
            if backup:
                undo_data = processor.end_stroke()
                if undo_data:
                    mod_vtx, mod_ch, old_sparse, new_sparse = undo_data

                    def redo():
                        self.set_sparse_weights(layer_idx, is_mask, mod_vtx, mod_ch, new_sparse)

                    def undo():
                        self.set_sparse_weights(layer_idx, is_mask, mod_vtx, mod_ch, old_sparse)

                    # 注册进 Maya 的原生撤销栈
                    apiundo.commit(redo, undo, execute=False)

                handle = self.get_handle(layer_idx, is_mask)
                if handle and handle.is_valid:
                    pass

    def paint_stroke_coroutine(self, layer_idx: int, is_mask: bool, backup: bool = True) -> typing.Generator[bool, StrokeParameters, None]:
        """
        [笔刷涂抹协程]
        这是一个专为笔刷涂抹设计的协程函数，提供了一个高性能的上下文环境，让用户可以在其中安全地调用 Cython 引擎进行实时权重修改。
        使用时,在 BrushManager 的涂抹循环中调用 .send(kwargs) 将当前帧的涂抹参数传入协程,在协程内部调用 processor.set_custom_array(...) 进行权重修改。
        Example:
        ```python
            #  =========================================================
            #  在 BrushManager 的涂抹循环中使用这个协程
            #  =========================================================
            stroke_coroutine = weights_manager.paint_stroke_coroutine(layer_idx, is_mask)
            next(stroke_coroutine)  # 启动协程

            # ===========================================================
            # 涂抹循环中与这个协程进行交互
            # ===========================================================
            while True:
                params = StrokeParameters(
                    brush_mode         = self.settings.mode,
                    weights_value      = array.array("f", [self.settings.strength]),
                    influences_indices = array.array("i", [self._active_influence_idx]),
                    pressure           = pressure,
                    iterations         = self.settings.iter,
                )  # 构造当前帧的参数
                stroke_coroutine.send(params)  # 将参数传入协程进行处理

            # ============================================================
            # 涂抹结束时调用 .close() 来触发协程的安全收尾与撤销注册
            # ============================================================
            stroke_coroutine.close()  # 结束涂抹,触发收尾与撤销注册
        """
        # 1. 负责绘制当前图层的 Processor
        processor = self._create_processor(layer_idx, is_mask)
        if not processor:
            yield False
            return

        # 前把 Layer -1 (渲染幕布) 的引擎实例化好，
        # 因为只做缓存覆盖，所以它完全不需要开启 begin_stroke 录制。
        base_processor = None
        if layer_idx >= 0:
            base_processor = self._create_processor(-1, is_mask=False)

        # 开启当前图层的 Undo 录制
        if backup:
            processor.begin_stroke()

        try:
            while True:
                # 在此挂起协程，等待 BrushManager 发来这一帧的涂抹参数
                # 直接接收外部 (Brush) 投喂的 C 级上下文
                ctx: cBrushCoreCython.BrushStrokeContext = yield True

                if ctx:
                    # 将参数解包传给 C 引擎进行真实涂抹与归一化
                    hit_count, hit_indices, _ = processor.process_stroke(ctx, normalize=not is_mask)
                    if hit_count > 0:
                        dirty_vtx_view = hit_indices[:hit_count]
                        if base_processor:
                            self.update_composite(dirty_vtx_view, out_processor=base_processor)

        except GeneratorExit:
            # 4. 外部调用 .close() 时，会触发此异常（相当于拦截了 doRelease）
            # 我们在这里安全收尾，并生成 Maya 撤销栈
            if backup:
                undo_data = processor.end_stroke()
                if undo_data:
                    mod_vtx, mod_ch, old_sparse, new_sparse = undo_data

                    def redo():
                        self.set_sparse_weights(layer_idx, is_mask, mod_vtx, mod_ch, new_sparse)
                        self.update_composite(mod_vtx)
                        self.updateDG()

                    def undo():
                        self.set_sparse_weights(layer_idx, is_mask, mod_vtx, mod_ch, old_sparse)
                        self.update_composite(mod_vtx)
                        self.updateDG()

                    apiundo.commit(redo, undo, execute=False)

    @defer_task
    def allocate_and_set_weights(
        self,
        layer_idx,
        is_mask,
        vtx_count,
        inf_count,
        influence_indices,
        weights_1d,
        normalize=False,
        backup=True,
    ):
        """
        [全量重建/覆盖图层]
        Python 负责重建 Maya 的底层物理内存和结构,然后移交 Cython 引擎进行纯数据的光速覆写。
        """
        handle = self.get_handle(layer_idx, is_mask)
        if handle is None:
            return False

        # --- [1. 记录图层全量结构快照] ---
        if backup:
            old_v_cnt, old_i_cnt, old_g_bones, _, _, old_w_1d = self.get_weights(layer_idx, is_mask)  # get weights 是拷贝,无需再次拷贝

            # 传入进来的数据可能是引用,这个数据要用来处理 redo, 一定要拷贝数据
            safe_new_idx = array.array("i", BufferManager.auto(influence_indices, "i").view)
            safe_new_w = array.array("f", BufferManager.auto(weights_1d, "f").view)

            def redo():
                self.allocate_and_set_weights(layer_idx, is_mask, vtx_count, inf_count, safe_new_idx, safe_new_w, backup=False)

            def undo():
                self.allocate_and_set_weights(layer_idx, is_mask, old_v_cnt, old_i_cnt, old_g_bones, old_w_1d, backup=False)

            apiundo.commit(redo, undo, execute=False)

        if vtx_count == 0:
            handle.clear()
            return True

        # --- [2. Python 负责重建底层物理内存与骨骼 Header (搭地基)] ---
        handle.resize(vtx_count, len(influence_indices))

        if influence_indices:
            _view = handle.memory.view
            _int = _view.cast("B").cast("i")
            _int[2 : 2 + len(influence_indices)] = BufferManager.auto(influence_indices, "i").view

        # 🚀 修复 2:召唤 Cython 引擎,执行全量底层 Replace 覆写
        if weights_1d is not None:
            processor = self._create_processor(layer_idx, is_mask)
            if processor:
                processor.set_custom_array(
                    source_values=BufferManager.auto(weights_1d, "f").view,
                    blend_mode=2,
                    vertex_indices=None,
                    channel_indices=None,
                )
                if normalize:
                    processor.normalize_weights(vertex_indices=None, priority_influence=-1)

        return True

    @update_dg
    def set_sparse_weights(
        self,
        layer_idx: int,
        is_mask: bool,
        vtx_indices,
        channel_indices,
        sparse_values,
    ):
        """
        设置稀疏权重
        专供 Undo / Redo 调用。
        直接使用 C 级覆盖能力还原快照
        """
        processor = self._create_processor(layer_idx, is_mask)
        if not processor:
            return

        # 撤销时不需要记录新的 Undo 快照,也不需要复杂的模式,直接暴力 Replace (blend_mode=2)
        processor.set_custom_array(
            source_values=sparse_values,
            blend_mode=2,
            vertex_indices=vtx_indices,
            channel_indices=channel_indices,
        )

    @defer_task
    def blend_weights(
        self,
        layer_idx: int,
        is_mask: bool,
        weights_1d,
        vtx_indices,
        influence_indices=None,
        blend_mode: int = 2,
        alpha: float = 1.0,
        falloff_weights=None,
        normalize=True,
        backup: bool = True,
    ):
        """
        [混合/覆写权重] (统一高级接口)
        支持加减乘除、透明度混合、蒙版衰减、自动归一化与稀疏撤销。
        """
        # 开启引擎会话
        with self.processor_session(layer_idx, is_mask, backup) as processor:
            # 如果没拿到引擎,直接退出
            if not processor:
                return False

            # --- 纯粹的数据转换与计算 ---
            # fmt:off
            v_view   = BufferManager.auto(vtx_indices              , "i").view    if vtx_indices              is not None else None
            b_view   = BufferManager.auto(influence_indices        , "i").view    if influence_indices        is not None else None
            src_view = BufferManager.auto(weights_1d               , "f").view
            fal_view = BufferManager.auto(falloff_weights          , "f").view    if falloff_weights          is not None else None
            # fmt:on

            # Cython 执行带混合模式的覆写
            processor.set_custom_array( source_values   = src_view   ,
                                        blend_mode      = blend_mode ,
                                        vertex_indices  = v_view     ,
                                        channel_indices = b_view     ,
                                        alpha           = alpha      ,
                                        falloff_weights = fal_view   )  # fmt:skip

            # 权重归一化
            if not is_mask and normalize:
                priority = b_view[0] if (b_view is not None and len(b_view) > 0) else -1
                processor.normalize_weights(v_view, priority)

        return True

    def add_layer(self, name: str = "NewLayer", weights_value=0.0, mask_weights_value=1.0) -> int:
        """
        添加一个新图层，自动分配并初始化底层物理内存。
        - 蒙版 (Mask) 初始化为全 1.0 (完全无遮挡)
        - 权重 (Weights) 初始化为全 0.0 (空权重)
        """
        node_name = self.mFnDepend_node.name()
        compound_plug = om1.MPlug(self.mObject_node, self.cSkin.aLayerCompound)

        # 1. 寻找可用 Logical Index
        max_idx = -1
        for i in range(compound_plug.numElements()):
            elem = compound_plug.elementByPhysicalIndex(i)
            if elem.logicalIndex() > max_idx:
                max_idx = elem.logicalIndex()
        new_idx = max_idx + 1

        plug_base = f"{node_name}.layers[{new_idx}]"

        # 2. 访问创建并设置基础属性 (利用 cmds 完美融入 Maya 原生撤销栈)
        cmds.setAttr(f"{plug_base}.layerName", name, type="string")
        cmds.setAttr(f"{plug_base}.layerEnabled", True)

        # 3. 获取 Base 层拓扑信息，用于精确开辟物理内存
        base_v_cnt, base_i_cnt, base_infs, _, _, _ = self.get_weights(-1, False)

        if base_v_cnt > 0:
            # --- 初始化 Mask (全 1.0) ---
            mask_weights = array.array("f", [mask_weights_value] * base_v_cnt)

            # 由于 allocate_and_set_weights 带有 @_defer_task 装饰器，
            # 这里并不会立刻执行，而是压入队列。当下方的 updateDG() 唤醒节点后，
            # sync_layer_cache 会首先执行识别到新图层，紧接着安全消费这里的修改任务！
            self.allocate_and_set_weights(layer_idx=new_idx, is_mask=True, vtx_count=base_v_cnt, inf_count=1, influence_indices=array.array("i", [0]), weights_1d=mask_weights, backup=True)

            # --- 初始化 Weights (全 0.0) ---
            layer_weights = array.array("f", [weights_value] * (base_v_cnt * base_i_cnt))

            self.allocate_and_set_weights(layer_idx=new_idx, is_mask=False, vtx_count=base_v_cnt, inf_count=base_i_cnt, influence_indices=base_infs, weights_1d=layer_weights, backup=True)

        # 4. 脏标记：唤醒节点求值，让 Maya 处理上述延迟队列！
        self.updateDG()

        return new_idx

    def delete_layer(self, layer_idx: int) -> bool:
        """
        删除指定图层并释放关联资源。
        """
        # 基础层 (Base Layer) 拦截，防止误删
        if layer_idx == -1:
            return False

        node_name = self.mFnDepend_node.name()
        plug_base = f"{node_name}.layers[{layer_idx}]"

        if cmds.objExists(plug_base):
            # 1. 移除多实例属性 (原生撤销栈支持)
            cmds.removeMultiInstance(plug_base, b=True)

            # 2. 清理 Python 层的字典路由映射
            if layer_idx in self.layers:
                del self.layers[layer_idx]

            # 3. 唤醒 DG，下一次 compute 将通过 sync_layer_cache 重新梳理图层列表
            self.updateDG()
            return True

        return False

    @defer_task
    def update_influences(self, new_influence_indices: list[int]):
        """
        [专属骨骼重组器] - 极致 O(I) 优化版
        前提：模型的顶点数绝对不变！
        支持：骨骼的新增、删除、打乱顺序。
        """
        if not new_influence_indices:
            return False

        new_i_cnt = len(new_influence_indices)
        layer_indices_to_update = [-1, *self.layer_indices]

        for layer_idx in layer_indices_to_update:
            # 1. 提取当前图层的旧数据
            v_cnt, old_i_cnt, old_g_bones, _, _, old_w_1d = self.get_weights(layer_idx, is_mask=False)

            if v_cnt <= 0 or old_i_cnt <= 0 or not old_w_1d:
                new_w_1d = array.array("f", [0.0] * (v_cnt * new_i_cnt))
                self.allocate_and_set_weights(
                    layer_idx,
                    False,
                    v_cnt,
                    new_i_cnt,
                    new_influence_indices,
                    new_w_1d,
                    normalize=True,
                    backup=True,
                )
                continue

            # 2. 建立新旧骨骼的“映射字典”
            bone_mapping = []
            old_bones_list = list(old_g_bones)
            for new_j, bone_id in enumerate(new_influence_indices):
                if bone_id in old_bones_list:
                    bone_mapping.append((old_bones_list.index(bone_id), new_j))

            # 3. 创建全新容器，并获取 1D 内存视图
            new_w_1d = array.array("f", [0.0] * (v_cnt * new_i_cnt))
            old_view = memoryview(old_w_1d).cast("B").cast("f")
            new_view = memoryview(new_w_1d).cast("B").cast("f")

            # =========================================================
            # 🚀 4. 终极性能魔法：按步长 (Stride) 在 C 层面整列拷贝！
            # 我们将 O(V) 的 10万次循环，降维成了 O(I) 的几十次循环！
            # =========================================================
            for old_j, new_j in bone_mapping:
                # 语法：view[start :: step]
                # 从旧视图提取第 old_j 根骨骼的所有顶点数据
                # 直接赋值给新视图第 new_j 根骨骼的所有顶点位置
                new_view[new_j::new_i_cnt] = old_view[old_j::old_i_cnt]

            # 5. 提交底层物理内存重组
            self.allocate_and_set_weights(
                layer_idx=layer_idx,
                is_mask=False,
                vtx_count=v_cnt,
                inf_count=new_i_cnt,
                influence_indices=array.array("i", new_influence_indices),
                weights_1d=new_w_1d,
                normalize=True,
                backup=True,
            )

        return True

    def update_composite(self, vtx_indices=None, out_processor=None):
        """
        图层混合器
        """
        is_sparse = vtx_indices is not None and len(vtx_indices) > 0
        v_view = BufferManager.auto(vtx_indices, "i").view if is_sparse else None

        # 提取激活的图层
        active_layers = [idx for idx in sorted(self.layer_indices) if self.get_layer(idx).enabled]
        if not active_layers:
            return

        def _do_blend(processor_inst: cBrushCoreCython.SkinWeightProcessor):
            # 1. 瞬间清空幕布 (直接调用底层 C 算子，0 毫秒)
            processor_inst.clear_buffer_sparse(vertex_indices=v_view)

            # 2. 遍历图层，扔给 C 引擎进行极速加法
            for layer_idx in active_layers:
                handle_w = self.get_handle(layer_idx, False)
                handle_m = self.get_handle(layer_idx, True)
                if not handle_w or not handle_m:
                    continue
                    
                _, _, _, w_in_1d = handle_w.parse_raw_weights()
                _, _, _, m_in_1d = handle_m.parse_raw_weights()
                if not w_in_1d or not m_in_1d:
                    continue
                    
                w_in_view = memoryview(w_in_1d).cast("B").cast("f")
                m_in_view = memoryview(m_in_1d).cast("B").cast("f")
                
                # 🚀 绝杀：将底层指针直接抛给 C 引擎执行循环，全过程纯 C 级运算！
                processor_inst.add_layer_weights(
                    layer_weights=w_in_view, 
                    layer_mask=m_in_view, 
                    vertex_indices=v_view
                )

            # 3. 最后在 C 引擎中执行强制归一化
            processor_inst.normalize_weights(vertex_indices=v_view, priority_influence=-1)

        # ---------------------------------------------------------
        # 引擎会话调度分流
        # ---------------------------------------------------------
        if out_processor:
            _do_blend(out_processor)
        else:
            with self.processor_session(-1, is_mask=False, backup=False) as temp_processor:
                if temp_processor:
                    _do_blend(temp_processor)