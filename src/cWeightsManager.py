from __future__ import annotations


import typing
import array
import functools
import threading
import contextlib
from collections import deque


import maya.OpenMaya as om1  # type:ignore

from . import apiundo
from . import cBrushCoreCython
from ._cRegistry import SkinRegistry
from .cBufferManager import BufferManager


if typing.TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer  # type: ignore


class WeightsHandle:
    """
    权重数据装配器 (MVectorArray 0拷贝黑客版)

    架构方案：
    1. 物理伪装：将 32-bit Float 强行塞入 64-bit MVectorArray 中。
    2. 原地扩容：通过 MVectorArray.setLength() 触发底层 C++ realloc，避免频繁销毁重建 MObject。
    3. 内存布局：
        ```
        ----------------------------------------------------------------------------------
        | vtx_count (int) | inf_count (int) | bone_indices... (int) | Weights... (float) |
        | <------------------ Header -----------------------------> | <--- Payload ----> |
        ----------------------------------------------------------------------------------
        ```
    """

    # fmt:off
    mDataHandle  : om1.MDataHandle = None # 变形器计算周期内的临时句柄
    mPlug        : om1.MPlug       = None # 节点插座，用于数据持久化
    mObject_data : om1.MObject     = None # 包装了 VectorArray 的 Maya 数据实体
    mVectorArray : om1.MVectorArray= None # 指向 MObject 内部缓冲区的引用

    memory        : BufferManager   = None # 权重数据的底层裸指针映射 (纯 Float 视图)
    weights_memory: BufferManager   = None # 仅权重部分视图
    max_capacity  : int             = -1   # 当前物理内存支持的最大 Float 存储量 (VectorLength * 6)
    length        : int             = -1   # 当前逻辑数据的有效 Float 长度
    # fmt:on

    def __init__(self, mPlug: om1.MPlug, mDataHandle: om1.MDataHandle):
        """
        初始化装配器。
        注：由于 nodeInitializer 设置了默认值，此处假定 mDataHandle 必定包含合法的 VectorArray。
        """
        self.mPlug = mPlug
        self.mDataHandle = mDataHandle
        self._setup_vector_buffer(self.mDataHandle)

    @classmethod
    def from_attr_string(cls, attr_path: str):
        """工具方法：从字符串路径("node.attr")快速构建"""
        sel = om1.MSelectionList()
        try:
            sel.add(attr_path)
        except RuntimeError:
            raise ValueError(f"Attribute path not found: {attr_path}")
        plug = om1.MPlug()
        sel.getPlug(0, plug)
        # 注意：外部调用时需自行提供对应的 DataHandle 或是通过 plug.asMDataHandle() 获取
        return cls(plug, plug.asMDataHandle())

    def _setup_vector_buffer(self, mDataHandle: om1.MDataHandle):
        """
        从 MDataHandle 解析 MVectorArray

        Updates:
            - `self.mObject_data`
            - `self.mVectorArray`
            - `self.vtx_count`
            - `self.influence_count`
            - `self.length`
            - `self.max_capacity`
            - `self.memory`
        """
        self.mObject_data = mDataHandle.data()
        if self.mObject_data.isNull():
            om1.MGlobal.displayError(f"{self.mPlug.name()}'s dataHandle:{mDataHandle} is Null")

        # 获取 MObject 内部数组的引用 (并非拷贝)
        fn_vector_data = om1.MFnVectorArrayData(self.mObject_data)
        self.mVectorArray = fn_vector_data.array()
        self.max_capacity = self.mVectorArray.length() * 6

        # 刷新物理内存指针映射
        self._remap_memory()

    def _remap_memory(self):
        """
        [物理层映射] 将底层 C++ VectorArray 内存地址映射为 Python 连续视图。
        此函数仅客观圈定物理地盘，不负责数据安全性校验 (校验交由 is_valid 处理)。

        Updates:
            - `self.vtx_count`
            - `self.influence_count`
            - `self.length`
            - `self.memory`
            - `self.weights_memory` (带有 physical padding，用于 fill 格式化)
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
        self.vtx_count = _int_view[0]
        self.influence_count = _int_view[1]

        self.length = (2 + self.influence_count) + (self.vtx_count * self.influence_count)

        if self.influence_count >= 0:
            self.weights_memory = self.memory.slice(start=2 + self.influence_count)
        else:
            self.weights_memory = None

    def resize(self, vtx_count: int, inf_count: int):
        """
        根据顶点和骨骼数量，原地扩容底层物理内存，并自动格式化为安全状态。

        Updates:
            - `self.vtx_count`
            - `self.influence_count`
            - `self.length`
            - `self.memory`
            - `self.weights_memory`
        """
        required_length = (2 + inf_count) + (vtx_count * inf_count)

        # 1. 如果容量足够，复用内存
        if required_length <= self.max_capacity:
            self.length = required_length
            self.vtx_count = vtx_count
            self.influence_count = inf_count

            _ptr = int(self.mVectorArray[0].this)
            _header = BufferManager.from_ptr(_ptr, "i", (2,))
            _header.view[0] = vtx_count
            _header.view[1] = inf_count

            self._remap_memory()

            # 🚀 格式化复用内存的负载区，清除上一次图层遗留的错位旧数据
            if self.weights_memory:
                self.weights_memory.fill(0.0)

            return False

        # 2. 如果容量不足，触发底层物理扩容
        vector_count = (required_length + 5) // 6
        self.mVectorArray.setLength(vector_count)
        self.max_capacity = vector_count * 6

        # 3. 注入合法表头
        _ptr = int(self.mVectorArray[0].this)
        _header = BufferManager.from_ptr(_ptr, "i", (2,))
        _header.view[0] = vtx_count
        _header.view[1] = inf_count

        # 4. 重新映射，切出完美的 weights_memory
        self._remap_memory()

        # 🚀 格式化新分配的物理内存，彻底绞杀底层 realloc 产生的 NaN 垃圾数据！
        if self.weights_memory:
            self.weights_memory.fill(0.0)

        return True

    def parse_raw_weights(self, raw_view=None):
        """
        解析符合内存布局协议的连续视图。

        Args:
            raw_view: 如果提供，则解析传入的视图 (用于 Undo 备份还原)；
                            如果为 None，则默认解析自身的 self.memory.view (用于内部重映射)。
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
        :param shrink: 是否强制释放物理内存。如果为 False，则只做极速的逻辑清零。
        """
        if self.mVectorArray is None:
            return False

        self.mVectorArray.setLength(1)
        self.max_capacity = 6

        self.resize(0, 0)

        return True

    def commit(self):
        """
        [数据固化]
        将当前内存里的 MObject 实体正式同步回节点的 MPlug。
        这会触发 Maya 的存盘标记 (Scene Dirty) 并将数据存入 Internal Storage。
        """
        pass
        # if (not self.mPlug 
        #     or not self.mObject_data 
        #     or self.mObject_data.isNull()):  # fmt:skip
        #     return
        # self.mPlug.setMObject(self.mObject_data)
        # print(f"commit weights handle by mPlug: {self.mPlug.name()}")

    @property
    def is_null(self) -> bool:
        """
        物理级检查：是否完全没有绑定 Maya 数据实体。
        (通常发生在新节点刚创建，或 plug 断开连接时)
        """
        return self.mObject_data is None or self.mObject_data.isNull() or self.mVectorArray is None

    @property
    def is_empty(self) -> bool:
        """
        业务级检查：是否是一个合法的“空图层”。
        (没有顶点，或没有分配任何骨骼，属于正常业务状态)
        """
        if self.is_null:
            return True
        v_cnt = getattr(self, "vtx_count", 0)
        i_cnt = getattr(self, "influence_count", 0)
        return v_cnt == 0 or i_cnt == 0

    @property
    def is_corrupted(self) -> bool:
        """
        安全级检查：物理内存是否被脏数据污染，或存在越界风险。
        """
        if self.is_null:
            return False  # 什么都没有，也就无所谓损坏

        vtx_count = getattr(self, "vtx_count", 0)
        inf_count = getattr(self, "influence_count", 0)

        # 1. 负数拦截 (绝对的 C++ 脏内存现象)
        if vtx_count < 0 or inf_count < 0:
            return True

        # 2. 物理越界拦截 (拦截几十亿的天文数字)
        required_floats = (2 + inf_count) + (vtx_count * inf_count)
        if required_floats > self.max_capacity:
            return True

        return False

    @property
    def is_valid(self) -> bool:
        """
        综合通行证：句柄是否处于绝对安全、可读写的健康状态。
        """
        return not self.is_null and not self.is_corrupted

    @property
    def view(self):
        """
        [语法糖] 安全获取底层的全量物理内存视图 (含 Header 和 Padding)。
        自带判空防御，用于底层数据的整体 Copy 或 Undo 备份。
        """
        return self.memory.view if self.memory is not None else None

    @property
    def weights_view(self):
        """
        [语法糖] 安全获取纯净的权重负载区视图 (去除了 Header)。
        自带判空防御，用于直接格式化 (fill) 或快速读取。
        """
        return self.weights_memory.view if self.weights_memory is not None else None


class WeightsLayerItem:
    def __init__(self, cSkin: CythonSkinDeformer, mDataHandle: om1.MDataHandle, logical_idx: int = -1):
        self.cSkin = cSkin
        self.logical_idx = logical_idx

        _handle_weights = mDataHandle.child(cSkin.aLayerWeights)
        _handle_mask = mDataHandle.child(cSkin.aLayerMask)
        _handle_enabled = mDataHandle.child(cSkin.aLayerEnabled)

        self.enabled = _handle_enabled.asBool()
        self.weights = WeightsHandle(self.mPlug_weights, _handle_weights)
        self.mask = WeightsHandle(self.mPlug_mask, _handle_mask)

    # ==========================================
    # 安全的属性获取：直接找 cSkin 要 MObject，完全不碰短命的 MDataHandle
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


class _DeferredTaskMixin:
    """
    专为 Maya 节点计算周期设计的任务调度器与刷新锁。
    提供队列管理、并发锁以及 updateDG 的执行防抖。
    """

    def __init__(self):
        # 全部声明为保护属性，向子类隐藏实现细节
        self._deferred_tasks = deque()
        self._tasks_lock = threading.Lock()
        self._is_dg_updating = False

    @staticmethod
    def _update_dg(func):
        """防抖锁：确保唯一次视口刷新"""

        @functools.wraps(func)
        def wrapper(self: WeightsManager, *args, **kwargs):
            is_top_level = not getattr(self, "_is_dg_updating", False)
            if is_top_level:
                self._is_dg_updating = True
            try:
                return func(self, *args, **kwargs)
            finally:
                if is_top_level:
                    self._is_dg_updating = False
                    if hasattr(self, "updateDG") and callable(getattr(self, "updateDG")):
                        self.updateDG()

        return wrapper

    @staticmethod
    def _defer_task(func):
        """延迟执行：推入队列并唤醒节点"""

        @functools.wraps(func)
        def wrapper(self: WeightsManager, *args, **kwargs):
            task = functools.partial(func, self, *args, **kwargs)
            with self._tasks_lock:
                self._deferred_tasks.append(task)

            if hasattr(self, "updateDG") and callable(getattr(self, "updateDG")):
                self.updateDG()

        return wrapper

    def execute_deferred_tasks(self):
        """[高内聚] 暴露给子类，用于消化队列"""
        if not self._deferred_tasks:
            return

        with self._tasks_lock:
            while self._deferred_tasks:
                task = self._deferred_tasks.popleft()
                task()


class WeightsManager(_DeferredTaskMixin):
    weights: WeightsHandle = None
    layers: list[WeightsLayerItem] = None

    def __init__(self, cSkin: CythonSkinDeformer):
        super().__init__()

        self.cSkin = cSkin
        self.mObj_node = cSkin.mObject
        self.mFnDep_node = cSkin.mFnDep

        self.plug_refresh: om1.MPlug = cSkin.plug_refresh
        self.plug_weights: om1.MPlug = om1.MPlug(cSkin.mObject, cSkin.aWeights)

        self.layers: dict[int, WeightsLayerItem] = {}

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
        一次性扫描 Maya 节点，刷新所有 Plug 缓存与底层内存池。
        当你在 UI 层面添加、删除了图层，或改变了节点连接后，手动调用此函数。
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

    def get_layer(self, index: int, logicalIndex: bool = True) -> "WeightsLayerItem":
        """
        获取图层实例，支持通过逻辑索引或物理索引进行查询。

        Args:
            index (int): 图层的索引值
            logicalIndex (bool): True 表示按逻辑索引查询，False 表示按物理顺序查询。
        Returns:
            WeightsLayerItem: 找到的图层实例，如果越界或找不到则返回 None。
        """
        if logicalIndex:
            return self.layers.get(index, None)
        else:
            try:
                return list(self.layers.values())[index]
            except IndexError:
                return None

    def get_handle(self, layer_logical_idx, isMask: bool = False) -> WeightsHandle:
        """
        严格根据逻辑索引寻找对应的 Handle
        """
        # 基础权重
        if layer_logical_idx == -1:
            return self.weights
        # layer 权重
        if layer_logical_idx in self.layers:
            layer_item = self.layers[layer_logical_idx]
            if isMask:
                return layer_item.mask
            else:
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
        提取指定范围的权重数据，并返回完整的上下文。
        - 全量提取时：直接底层内存拷贝，实现极致零延迟。
        - 局部提取时：调度 Cython 引擎进行极速精细抠取。
        """
        # 1. 直接获取句柄并进行绝对安全拦截
        handle = self.get_handle(layer_idx, is_mask)
        if handle is None or not handle.is_valid:
            return 0, 0, array.array("i"), array.array("i"), array.array("i"), array.array("f")

        # 2. 🚀 让句柄自己解析，直接拿到完美的 1D 权重视图
        v_count, i_count, g_bones_view, w_1d_view = handle.parse_raw_weights()

        # 拦截空图层
        if v_count <= 0 or i_count <= 0 or w_1d_view is None:
            return 0, 0, array.array("i"), array.array("i"), array.array("i"), array.array("f")

        # 提取全局骨骼 ID 备份
        global_bone_ids = array.array("i", g_bones_view) if g_bones_view else array.array("i")

        # ---------------------------------------------------------------------
        # 🚀 性能分支：全量提取 vs 局部提取
        # ---------------------------------------------------------------------
        if vtx_indices is None and bone_local_indices is None:
            # 【分支 A：全量提取】(如 Undo 备份)
            # 既然 handle 已经切出了完美的 w_1d_view，直接 C 级拷贝，跳过引擎启动！
            weights_1d = array.array("f", w_1d_view)
            out_vtx = array.array("i", range(v_count))
            out_bone = array.array("i", range(i_count))
        else:
            # 【分支 B：局部提取】(如笔刷吸色 / 局部修改备份)
            # 装配引擎，让 C++ 去做跳跃式的内存抓取
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

    @contextlib.contextmanager
    def _processor_session(self, layer_idx: int, is_mask: bool, backup: bool = True):
        """
        [引擎上下文管理器]
        统管 Cython 算力引擎的获取、快照录制与撤销栈注册。
        基于 RAII 思想，确保底层资源绝对安全。
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
            # 4. 收尾工作 (Teardown)：确保快照闭合与闭包注册
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
                    handle.commit()

    @_DeferredTaskMixin._defer_task
    def set_weights(
        self,
        layer_idx: int,
        is_mask: bool,
        vtx_indices,
        weights_1d,
        influence_indices=None,
        blend_mode: int = 2,
        alpha: float = 1.0,
        falloff_weights=None,
        normalize=True,
        backup: bool = True,
    ):
        """
        将所有输入丢给 Cython 无头引擎，
        支持加减乘除、透明度混合、蒙版衰减、自动归一化与稀疏撤销。
        """
        # 开启引擎会话
        with self._processor_session(layer_idx, is_mask, backup) as processor:
            # 如果没拿到引擎，直接退出
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
                processor._normalize_weights(v_view, priority)

        return True

    @_DeferredTaskMixin._defer_task
    def init_handle_data(
        self,
        layer_idx,
        is_mask,
        vtx_count,
        inf_count,
        influence_indices,
        weights_1d,
        backup=True,
    ):
        """
        [全量重建/覆盖图层]
        Python 负责重建 Maya 的底层物理内存和结构，然后移交 Cython 引擎进行纯数据的光速覆写。
        """
        handle = self.get_handle(layer_idx, is_mask)
        if handle is None:
            return False

        # --- [1. 记录图层全量结构快照] ---
        if backup:
            old_v_cnt, old_i_cnt, old_g_bones, _, _, old_w_1d = self.get_weights(layer_idx, is_mask)  # get weights 是拷贝，无需再次拷贝

            # 传入进来的数据可能是引用，这个数据要用来处理 redo, 一定要拷贝数据
            safe_new_idx = array.array("i", BufferManager.auto(influence_indices, "i").view)
            safe_new_w = array.array("f", BufferManager.auto(weights_1d, "f").view)

            def redo():
                self.init_handle_data(layer_idx, is_mask, vtx_count, inf_count, safe_new_idx, safe_new_w, backup=False)

            def undo():
                self.init_handle_data(layer_idx, is_mask, old_v_cnt, old_i_cnt, old_g_bones, old_w_1d, backup=False)

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

        # 🚀 修复 2：召唤 Cython 引擎，执行全量底层 Replace 覆写
        if weights_1d is not None:
            processor = self._create_processor(layer_idx, is_mask)
            if processor:
                processor.set_custom_array(
                    source_values=BufferManager.auto(weights_1d, "f").view,
                    blend_mode=2,
                    vertex_indices=None,
                    channel_indices=None,
                )
        return True

    @_DeferredTaskMixin._update_dg
    def set_sparse_weights(
        self,
        layer_idx: int,
        is_mask: bool,
        vtx_indices,
        channel_indices,
        sparse_values,
    ):
        """
        专供 Undo / Redo 闭包调用。
        直接使用 C 级覆盖能力还原快照，彻底告别 Python 循环！
        """
        processor = self._create_processor(layer_idx, is_mask)
        if not processor:
            return

        # 撤销时不需要记录新的 Undo 快照，也不需要复杂的模式，直接暴力 Replace (blend_mode=2)
        processor.set_custom_array(
            source_values=sparse_values,
            blend_mode=2,
            vertex_indices=vtx_indices,
            channel_indices=channel_indices,
        )

    def _create_processor(self, layer_idx: int, is_mask: bool):
        """
        - 提取指定图层的物理内存，并为创建 `SkinWeightProcessor`实例，可以根据`return`的实例进行操作。
        - `SkinWeightProcessor` 本质是笔刷处理器，初始化权重笔刷，将权重数据与笔刷引擎托管给父类进行通用运算。
        - 在这里可以用来快速设置权重，并且自动注册undo和redo快照(参考`set_sparse_data`函数)。
        """
        handle = self.get_handle(layer_idx, is_mask)

        if handle is None or not handle.is_valid:
            return None

        vtx_count, inf_count, _, weights_1d = handle.parse_raw_weights()

        if vtx_count <= 0 or inf_count <= 0 or weights_1d is None:
            return None

        weights_2d = weights_1d.cast("B").cast("f", (vtx_count, inf_count))

        tmp_idx = BufferManager.allocate("i", (vtx_count,))
        tmp_bool = BufferManager.allocate("B", (vtx_count,))
        tmp_locks = BufferManager.allocate("B", (inf_count,))

        _undo_buffer = BufferManager.allocate("f", (vtx_count, inf_count))

        processor = cBrushCoreCython.SkinWeightProcessor(
            weights_2d,
            tmp_idx.view,  # 直接传入 view
            tmp_bool.view,  # 直接传入 view
            tmp_locks.view,  # 直接传入 view
            _undo_buffer.view,  # 直接传入完美的 2D view！
        )
        return processor
