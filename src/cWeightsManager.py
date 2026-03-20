from __future__ import annotations


import typing
import array
import functools
import threading
from collections import deque


import maya.OpenMaya as om1  # type:ignore

from . import apiundo
from . import cBrushCoreCython
from ._cRegistry import SkinRegistry
from .cBufferManager import BufferManager


if typing.TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer  # type: ignore


def updateDG(func):
    """
    触发唯一次 Maya 视口刷新。
    """

    @functools.wraps(func)
    def wrapper(self: WeightsManager, *args, **kwargs):
        # 检查当前是否已经处于某一个 updateDG 的执行周期内
        is_top_level = not getattr(self, "_is_dg_updating", False)

        # 如果是最外层调用，把门锁死
        if is_top_level:
            self._is_dg_updating = True

        try:
            return func(self, *args, **kwargs)
        finally:
            # 只有最外层函数运行完毕，才允许刷新视口
            if is_top_level:
                self._is_dg_updating = False  # 先解锁

                # 触发你原本的强制刷新逻辑
                if hasattr(self, "updateDG") and callable(getattr(self, "updateDG")):
                    self.updateDG()

    return wrapper


def async_queued_task(func):
    """
    [魔法装饰器]
    拦截函数的直接执行，将其打包为闭包任务推入队列，并通知 Maya 刷新。
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 1. 使用 partial 将函数本体、self 以及所有参数“冻结”成一个待执行任务包
        task = functools.partial(func, self, *args, **kwargs)

        # 2. 塞进队列
        with self.queue_lock:
            self.stroke_queue.append(task)

        # 3. 踢醒 Maya 的 deform
        self.updateDG()

    return wrapper


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

    memory       : BufferManager   = None # 权重数据的底层裸指针映射 (纯 Float 视图)
    max_capacity : int             = -1   # 当前物理内存支持的最大 Float 存储量 (VectorLength * 6)
    length       : int             = -1   # 当前逻辑数据的有效 Float 长度
    # fmt:on

    def __init__(self, mPlug: om1.MPlug, mDataHandle: om1.MDataHandle):
        """
        初始化装配器。
        注：由于 nodeInitializer 设置了默认值，此处假定 mDataHandle 必定包含合法的 VectorArray。
        """
        self.mPlug = mPlug
        self.mDataHandle = mDataHandle
        self._setup_vector_buffer(self.mDataHandle)

    @property
    def is_valid(self) -> bool:
        """检查内存映射是否有效且逻辑数据是否存在"""
        # 1. 基础对象与内存引用检查
        if self.mObject_data is None or self.mObject_data.isNull():
            return False
        if self.mVectorArray is None or self.memory is None:
            return False
        if self.memory.view is None:
            return False

        # 2. 获取表头数据
        v_count = getattr(self, "vtx_count", 0)
        i_count = getattr(self, "influence_count", 0)
        
        # 🚨 修正点 1：0 是合法的空图层状态！负数才是内存乱码
        if v_count < 0 or i_count < 0:
            return False
            
        # 3. 计算所需的 Float 容量
        # 哪怕 v_count 和 i_count 都是 0，Header 依然存在 (占用 2 个坑位)
        required_floats = 2 + i_count + (v_count * i_count)
        
        # 物理分配的总容量 (1 个 MVector = 3 个 double = 6 个 float)
        actual_floats = self.mVectorArray.length() * 6 

        # 🚨 修正点 2：如果表头读出了巨大的乱码，required_floats 会远超 actual_floats
        if required_floats > actual_floats:
            return False

        return True

    def _setup_vector_buffer(self, mDataHandle: om1.MDataHandle):
        """
        从 MDataHandle 解析并确立对 MVectorArray 内存块的绑定。
        """
        self.mObject_data = mDataHandle.data()
        if self.mObject_data.isNull():
            return

        # 获取 MObject 内部数组的引用 (并非拷贝)
        fn_vector_data = om1.MFnVectorArrayData(self.mObject_data)
        self.mVectorArray = fn_vector_data.array()

        # 刷新物理内存指针映射
        self._remap_memory()

    def _remap_memory(self):
        """
        核心方法：重新映射物理指针。
        由于 setLength 会触发 C++ 内存重新分配，每次长度改变后必须刷新指针。
        """
        if self.mVectorArray is None or self.mVectorArray.length() <= 0:
            return

        # 1. 提取第 0 个元素的裸指针 (SWIG 代理地址)
        _ptr = int(self.mVectorArray[0].this)
        self.max_capacity = self.mVectorArray.length() * 6

        # 2. 映射表头 (Header): 解析前两个 int32 (vtx_count, inf_count)
        _header_memory = BufferManager.from_ptr(_ptr, "i", (2,))
        self.vtx_count = _header_memory.view[0]
        print("self.vtx_count",self.vtx_count)
        self.influence_count = _header_memory.view[1]
        print("self.influence_count",self.influence_count)

        # 3. 计算并锁定逻辑数据视图长度
        self.length = (2 + self.influence_count) + (self.vtx_count * self.influence_count)
        print("self.length",self.length)

        # 4. 生成 Float32 视图供 Cython 使用
        self.memory = BufferManager.from_ptr(_ptr, "f", (self.length,))

    def resize(self, length: int):
        """
        原地扩容底层物理内存。
        通过修改 MVectorArray 长度，实现对 MObject 缓冲区的直接重塑。
        """
        # 如果当前容量足够，只需更新逻辑长度
        if length <= self.max_capacity:
            self.length = length
            return False

        # 1. 换算所需的 Vector 数量 (向上取整)
        vector_count = (length + 5) // 6

        # 2. 触发 C++ 原地扩容 (原地 realloc)
        self.mVectorArray.setLength(vector_count)


        return True

    def commit(self):
        """
        [数据固化]
        将当前内存里的 MObject 实体正式同步回节点的 MPlug。
        这会触发 Maya 的存盘标记 (Scene Dirty) 并将数据存入 Internal Storage。
        """
        if self.mPlug and self.mObject_data and not self.mObject_data.isNull():
            self.mPlug.setMObject(self.mObject_data)

    def clear(self):
        """
        重置数据为默认状态 (1个元素的空壳)
        """
        empty_array = om1.MVectorArray()
        empty_array.setLength(1)  # 保留1个元素防止 [0].this 崩溃

        fn_data = om1.MFnVectorArrayData()
        self.mObject_data = fn_data.create(empty_array)
        self.mVectorArray = empty_array

        if self.mDataHandle is not None:
            self.mDataHandle.setMObject(self.mObject_data)

        self._remap_memory()
        return True

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


class WeightsLayerItem:
    def __init__(self, cSkin: CythonSkinDeformer, mDataHandle: om1.MDataHandle, logical_idx: int = -1):
        self.cSkin = cSkin
        self.logical_idx = logical_idx

        _handle_weights = mDataHandle.child(cSkin.aLayerWeights)
        _handle_mask = mDataHandle.child(cSkin.aLayerMask)
        _handle_enabled = mDataHandle.child(cSkin.aLayerEnabled)

        self.enabled = _handle_enabled.asBool()
        self.weights = WeightsHandle(self.mPlug_mask, _handle_weights)
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


class WeightsManager:
    weights: WeightsHandle = None
    layers: list[WeightsLayerItem] = None

    def __init__(self, cSkin: CythonSkinDeformer):
        self.cSkin = cSkin
        self.mObj_node = cSkin.mObject
        self.mFnDep_node = cSkin.mFnDep

        self.plug_refresh: om1.MPlug = cSkin.plug_refresh
        self.plug_weights: om1.MPlug = om1.MPlug(cSkin.mObject, cSkin.aWeights)

        self.layers: dict[int, WeightsLayerItem] = {}

        # 🚀 新增：把指令队列和锁封装在 Manager 内部
        self.stroke_queue = deque()
        self.queue_lock = threading.Lock()

    def process_queued_strokes(self):
        """
        [内部专用] 消化队列里的任务。在 deform 周期内调用。
        """
        if not self.stroke_queue:
            return

        with self.queue_lock:
            while self.stroke_queue:
                # 弹出任务包
                task = self.stroke_queue.popleft()
                # 💥 闭眼直接调用它！它会自动把之前保存的参数全部传进去执行
                task()

    @classmethod
    def get_manager_from_cSkin(cls, cSkinNodeName: str):
        cSkin: CythonSkinDeformer = SkinRegistry.from_instance_by_string(cSkinNodeName)
        return cSkin.weights_manager

    def update_data(self, mDataBlock: om1.MDataBlock):
        """
        [状态同步器]
        一次性扫描 Maya 节点，刷新所有 Plug 缓存与底层内存池。
        当你在 UI 层面添加、删除了图层，或改变了节点连接后，手动调用此函数。
        """
        # weights
        weights_dataHandle = mDataBlock.inputValue(self.cSkin.aWeights)
        plug_weights = om1.MPlug(self.cSkin.mObject, self.cSkin.aWeights)
        self.weights = WeightsHandle(plug_weights, weights_dataHandle)

        # layer
        self.layers.clear()
        mArrayDataHandle: om1.MArrayDataHandle = mDataBlock.inputArrayValue(self.cSkin.aLayerCompound)
        _count = mArrayDataHandle.elementCount()
        for idx in range(_count):
            mArrayDataHandle.jumpToArrayElement(idx)
            logical_idx = mArrayDataHandle.elementIndex()
            element_handle: om1.MDataHandle = mArrayDataHandle.inputValue()
            self.layers[logical_idx] = WeightsLayerItem(self.cSkin, element_handle, logical_idx)

    @property
    def layer_indices(self) -> list[int]:
        return list(self.layers.keys())

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

    @staticmethod
    def parse_raw_weights(raw_view):
        """
        将连续内存拆解为逻辑视图。
        """
        if raw_view is None or len(raw_view) < 2:
            return 0, 0, None, None

        int_view = raw_view.cast("B").cast("i")

        vtx_count = int_view[0]
        influence_count = int_view[1]
        header_size = 2 + influence_count

        influence_indices_view = int_view[2:header_size]
        weights_view = raw_view[header_size:]

        return vtx_count, influence_count, influence_indices_view, weights_view

    def get_raw_weights(self, layer_idx: int, is_mask: bool):
        """
        直接索要底层的物理内存地址视图，不做任何解析。
        """
        handle = self.get_handle(layer_idx, is_mask)
        if handle is None or not handle.is_valid:
            return None
        return handle.memory.view

    def _create_processor(self, layer_idx: int, is_mask: bool):
        """
        - 提取指定图层的物理内存，并为创建 `SkinWeightProcessor`实例，可以根据`return`的实例进行操作。
        - `SkinWeightProcessor` 本质是笔刷处理器，初始化权重笔刷，将权重数据与笔刷引擎托管给父类进行通用运算。
        - 在这里可以用来快速设置权重，并且自动注册undo和redo快照(参考`set_sparse_data`函数)。
        """
        _raw_view = self.get_raw_weights(layer_idx, is_mask)
        if _raw_view is None:
            return None

        vtx_count, inf_count, _, weights_1d = self.parse_raw_weights(_raw_view)
        print(vtx_count, inf_count)

        weights_2d = weights_1d.cast("B").cast("f", (vtx_count, inf_count))

        tmp_idx = BufferManager.allocate("i", (vtx_count,))
        tmp_bool = BufferManager.allocate("B", (vtx_count,))
        tmp_locks = BufferManager.allocate("B", (inf_count,))

        # 分配 Undo 内存 (先拉平为 1D，再利用 memoryview 强转为 2D)
        _undo_buffer = BufferManager.allocate("f", (vtx_count, inf_count))

        processor = cBrushCoreCython.SkinWeightProcessor(
            weights_2d,
            tmp_idx.view,  # 直接传入 view
            tmp_bool.view,  # 直接传入 view
            tmp_locks.view,  # 直接传入 view
            _undo_buffer.view,  # 直接传入完美的 2D view！
        )
        return processor

    @updateDG
    def _set_sparse_data(self, layer_idx: int, is_mask: bool, vtx_indices, channel_indices, sparse_values):
        """
        专供 Undo / Redo 闭包调用。
        直接使用 C 级覆盖能力还原快照，彻底告别 Python 循环！
        """
        processor = self._create_processor(layer_idx, is_mask)
        if not processor:
            return

        # 撤销时不需要记录新的 Undo 快照，也不需要复杂的模式，直接暴力 Replace (blend_mode=2)
        processor.set_custom_array(source_values=sparse_values, blend_mode=2, vertex_indices=vtx_indices, channel_indices=channel_indices)

    @async_queued_task
    def set_weights(
        self,
        layer_idx: int,
        is_mask: bool,
        vtx_indices,
        weights_1d,
        locked_influence_indices=None,
        blend_mode: int = 2,  # 0:Add, 1:Sub, 2:Replace, 3:Multiply
        alpha: float = 1.0,
        falloff_weights=None,
        normalize=True,
        backup: bool = True,
    ):
        """
        将所有输入丢给 Cython 无头引擎，
        支持加减乘除、透明度混合、蒙版衰减、自动归一化与稀疏撤销。
        """
        processor = self._create_processor(layer_idx, is_mask)
        if not processor:
            return False

        # 转为 Cython 认识的 memoryview
        # fmt:off
        v_view   = BufferManager.auto(vtx_indices              , "i").view    if vtx_indices              is not None else None
        b_view   = BufferManager.auto(locked_influence_indices , "i").view    if locked_influence_indices is not None else None
        src_view = BufferManager.auto(weights_1d               , "f").view
        fal_view = BufferManager.auto(falloff_weights          , "f").view    if falloff_weights          is not None else None
        # fmt:on

        # 开启快照录制
        if backup:
            processor.begin_stroke()

        # Cython 执行带混合模式的覆写
        processor.set_custom_array( source_values   = src_view   ,
                                    blend_mode      = blend_mode ,
                                    vertex_indices  = v_view     ,
                                    channel_indices = b_view     ,
                                    alpha           = alpha      ,
                                    falloff_weights = fal_view   )  # fmt:skip

        # 如果不是mask，覆写后必须进行权重归一化！
        if not is_mask and normalize:
            priority = b_view[0] if (b_view is not None and len(b_view) > 0) else -1
            processor._normalize_weights(v_view, priority)

        #  从快照 提取Undo/Redo 数据
        if backup:
            undo_data = processor.end_stroke()
            if undo_data:
                mod_vtx, mod_ch, old_sparse, new_sparse = undo_data

                def redo():
                    self._set_sparse_data(layer_idx, is_mask, mod_vtx, mod_ch, new_sparse)

                def undo():
                    self._set_sparse_data(layer_idx, is_mask, mod_vtx, mod_ch, old_sparse)

                apiundo.commit(redo, undo, execute=False)

        return True

    @async_queued_task
    def rebuild_layer(self, layer_idx, is_mask, vtx_count, inf_count, influence_indices, weights_1d, backup=True):
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
                self.rebuild_layer(layer_idx, is_mask, vtx_count, inf_count, safe_new_idx, safe_new_w, backup=False)

            def undo():
                self.rebuild_layer(layer_idx, is_mask, old_v_cnt, old_i_cnt, old_g_bones, old_w_1d, backup=False)

            apiundo.commit(redo, undo, execute=False)

        if vtx_count == 0:
            handle.clear()
            return True

        # --- [2. Python 负责重建底层物理内存与骨骼 Header (搭地基)] ---
        total_size = (2 + len(influence_indices)) + (vtx_count * len(influence_indices))
        handle.resize(total_size)
        if not handle.is_valid:
            return False

        _view = handle.memory.view
        _int = _view.cast("B").cast("i")
        _int[0], _int[1] = vtx_count, len(influence_indices)
        if influence_indices:
            _int[2 : 2 + len(influence_indices)] = BufferManager.auto(influence_indices, "i").view

        processor = self._create_processor(layer_idx, is_mask)
        if processor:
            processor.set_custom_array(
                source_values=BufferManager.auto(weights_1d, "f").view,
                blend_mode=2,
                vertex_indices=None,
                channel_indices=None,
            )

        return True

    def get_weights(self, layer_idx: int, is_mask: bool, vtx_indices=None, bone_local_indices=None):
        """
        [提取/复制权重] (统一高级接口)
        利用底层 Cython 引擎极速抠出指定范围的权重数据，并返回完整的上下文。
        如果不传索引，默认提取全量数据。

        Returns:
            tuple: (
                vtx_count: int,                # 1. 图层总顶点数
                inf_count: int,                # 2. 图层总骨骼数
                global_bone_ids: array.array,  # 3. 图层的全局骨骼ID (Header数据)
                out_vtx_indices: array.array,  # 4. 本次提取对应的顶点局部索引
                out_bone_indices: array.array, # 5. 本次提取对应的骨骼局部索引
                weights_1d: array.array        # 6. 本次提取的 1D 纯净权重数据
            )
        """
        # 1. 解析基础 Header 结构信息 (顺便获取全局骨骼ID，供 set_weights_all 备份使用)
        raw_view = self.get_raw_weights(layer_idx, is_mask)
        if not raw_view:
            return 0, 0, array.array("i"), array.array("i"), array.array("i"), array.array("f")

        v_count, i_count, g_bones_view, _ = self.parse_raw_weights(raw_view)
        global_bone_ids = array.array("i", g_bones_view) if g_bones_view else array.array("i")

        # 2. 装配无头引擎，利用 C 语言极速提取 1D 权重数据
        processor = self._create_processor(layer_idx, is_mask)
        if not processor:
            return v_count, i_count, global_bone_ids, array.array("i"), array.array("i"), array.array("f")

        v_view = BufferManager.auto(vtx_indices, "i").view if vtx_indices is not None else None
        b_view = BufferManager.auto(bone_local_indices, "i").view if bone_local_indices is not None else None

        # Cython 极速返回 1D 纯净数据
        weights_1d = processor.get_custom_array(v_view, b_view)

        # 3. 智能补全提取范围的上下文索引 (这是最贴心的一步)
        # 如果你传了 None (要求全量提取)，系统自动帮你生成完整的物理索引数组
        if vtx_indices is None:
            out_vtx = array.array("i", range(v_count))
        else:
            out_vtx = array.array("i", v_view)

        if bone_local_indices is None:
            out_bone = array.array("i", range(i_count))
        else:
            out_bone = array.array("i", b_view)

        return v_count, i_count, global_bone_ids, out_vtx, out_bone, weights_1d

    def updateDG(self):
        self.cSkin.setDirty()
