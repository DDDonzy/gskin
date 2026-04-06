from __future__ import annotations


import typing
import array
import contextlib


import maya.OpenMaya as om1  # type:ignore


from . import apiundo
from ._cRegistry import SkinRegistry
from .cBufferManager import BufferManager
if typing.TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer  # type: ignore




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
    mVectorArray : om1.MVectorArray # 指向 MObject 内部缓冲区的引用

    memory        : BufferManager    # 权重数据的底层裸指针映射 (纯 Float 视图)
    weights_memory: BufferManager    # 仅权重部分视图
    max_capacity  : int              # 当前物理内存支持的最大 Float 存储量 (VectorLength * 6)
    length        : int              # 当前逻辑数据的有效 Float 长度  
    # fmt:on

    def __init__(
        self,
        cSkin: CythonSkinDeformer,
        mPlug: om1.MPlug,
        mDataHandle: om1.MDataHandle,
    ):
        """
        初始化装配器。
        注:由于 nodeInitializer 设置了默认值,此处假定 mDataHandle 必定包含合法的 VectorArray。
        """
        # fmt:off
        self.cSkin        = cSkin
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
        """
        工具方法:从字符串路径("node.attr")快速构建。
        自动反查注册表，绑定 cSkin 节点实例。
        """
        sel = om1.MSelectionList()
        try:
            sel.add(attr_path)
        except RuntimeError as e:
            raise ValueError(f"Attribute path not found: {attr_path}") from e

        plug = om1.MPlug()
        sel.getPlug(0, plug)

        # 1. 向上反查：通过 Plug 拿到所属节点的 MObject
        node_obj = plug.node()
        if node_obj.isNull():
            raise RuntimeError(f"Cannot find node for attribute: {attr_path}")
        cSkin = SkinRegistry.get_instance_by_api1(node_obj)
        if not cSkin:
            raise RuntimeError(f"Node '{node_obj.name()}' is not a registered CythonSkinDeformer.")

        return cls(cSkin, plug, plug.asMDataHandle())

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
            # 如果当前没有合法的 MVectorArray,先创建一个空的,再走正常扩容流程
            init_ary = om1.MVectorArray()
            init_obj = om1.MFnVectorArrayData().create(init_ary)
            self.mVectorArray = init_ary
            self.mObject_data = init_obj
            self.mDataHandle.setMObject(init_obj)
            self._setup_vector_buffer(self.mDataHandle)

        required_length = (2 + inf_count) + (vtx_count * inf_count)

        # 如果容量足够,复用内存
        if required_length <= self.max_capacity:
            self.length = required_length

            _ptr = int(self.mVectorArray[0].this)
            _header = BufferManager.from_ptr(_ptr, "i", (2,))
            _header.view[0] = vtx_count
            _header.view[1] = inf_count

            self._remap_memory()

            # 格式化现有的物理内存,避免 realloc 产生的 NaN 垃圾数据
            if self.weights_memory:
                self.weights_memory.fill(0.0)

            return False

        # 如果容量不足,触发底层物理扩容
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

        # 格式化新分配的物理内存,避免 realloc 产生的 NaN 垃圾数据
        if self.weights_memory:
            self.weights_memory.fill(0.0)

        return True

    def parse_raw_weights(self, raw_view=None):
        """
        解析符合内存布局协议的连续视图。

        Args:
            raw_view: 如果提供,则解析传入的视图 (用于 Undo 备份还原)如果为 None,则默认解析自身的 self.memory.view (用于内部重映射)
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
            return False

        vtx_count = getattr(self, "vtx_count", 0)
        inf_count = getattr(self, "influence_count", 0)

        if vtx_count < 0 or inf_count < 0:
            return True

        required_floats = (2 + inf_count) + (vtx_count * inf_count)
        return required_floats > self.max_capacity

    @property
    def is_valid(self) -> bool:
        """综合通行证:句柄是否处于绝对安全、可读写的健康状态。"""
        return not self.is_null and not self.is_corrupted

    @property
    def view(self):
        """[语法糖] 安全获取底层的全量物理内存视图 (含 Header 和 Padding)。"""
        return self.memory.view if self.memory is not None else None

    @property
    def weights_view(self):
        """[语法糖] 安全获取纯净的权重负载区视图 (去除了 Header)。"""
        return self.weights_memory.view if self.weights_memory is not None else None

    @contextlib.contextmanager
    def processor_session(self, backup: bool = True):
        """
        [引擎上下文管理器]
        Handle 自己向 cSkin 节点申请即时算力引擎，并统管快照录制与撤销栈注册。
        """
        v_count, stride_count, _, raw_1d = self.parse_raw_weights()
        if v_count <= 0 or stride_count <= 0 or not raw_1d:
            yield None
            return

        weights_2d = memoryview(raw_1d).cast("B").cast("f", (v_count, stride_count))
        processor = self.cSkin.get_processor(weights_2d)

        if backup:
            processor.begin_stroke()

        try:
            yield processor
        finally:
            if backup:
                undo_data = processor.end_stroke()
                if undo_data:
                    mod_vtx, mod_ch, old_sparse, new_sparse = undo_data

                    def redo():
                        self.set_sparse_weights(mod_vtx, mod_ch, new_sparse)
                        self.cSkin.forceRefresh()

                    def undo():
                        self.set_sparse_weights(mod_vtx, mod_ch, old_sparse)
                        self.cSkin.forceRefresh()

                    apiundo.commit(redo, undo, execute=False)

    def get_weights(self, channel_logical_ids=None, vtx_indices=None):
        """
        [纯净版 API - 零拷贝] 提取当前句柄的物理数据视图。

        Returns:
            tuple: (out_vtx, out_logical_ids, weights_1d)
                - out_vtx: 局部提取时返回 memoryview；全量提取时返回 None (代表全量)。
                - out_logical_ids: 通道逻辑 ID 的 memoryview (零拷贝底层 Header)。
                - weights_1d: 权重数据的 memoryview (零拷贝)。
        """
        v_count, stride_count, indices_view, raw_1d = self.parse_raw_weights()
        if v_count <= 0 or stride_count <= 0 or not raw_1d:
            return None, None, None

        out_logical_ids = channel_logical_ids if channel_logical_ids is not None else indices_view

        physical_cols = None
        if channel_logical_ids is not None:
            cols = []
            for logic_id in channel_logical_ids:
                try:
                    cols.append(list(indices_view).index(logic_id))
                except ValueError:
                    pass
            if not cols:
                return None, None, None
            physical_cols = cols

        out_vtx = vtx_indices if vtx_indices is not None else None

        if vtx_indices is None and physical_cols is None:
            weights_1d = memoryview(raw_1d).cast("B").cast("f")

        elif vtx_indices is None and physical_cols is not None and len(physical_cols) == 1:
            col_idx = physical_cols[0]
            weights_1d = memoryview(raw_1d).cast("B").cast("f")[col_idx::stride_count]

        else:
            v_view = BufferManager.auto(vtx_indices, "i").view if vtx_indices is not None else None
            b_view = BufferManager.auto(physical_cols, "i").view if physical_cols is not None else None

            weights_2d = memoryview(raw_1d).cast("B").cast("f", (v_count, stride_count))
            processor = self.cSkin.get_processor(weights_2d)
            weights_1d = processor.get_custom_array(v_view, b_view)

        return out_vtx, out_logical_ids, weights_1d

    def set_sparse_weights(self, vtx_indices, channel_indices, sparse_values):
        """
        设置稀疏权重
        专供 Undo / Redo 调用。
        直接使用 C 级覆盖能力还原快照
        """
        v_count, stride_count, _, raw_1d = self.parse_raw_weights()
        if v_count <= 0:
            return

        weights_2d = memoryview(raw_1d).cast("B").cast("f", (v_count, stride_count))
        processor = self.cSkin.get_processor(weights_2d)

        # 撤销时不需要记录新的 Undo 快照,也不需要复杂的模式,直接暴力 Replace (blend_mode=2)
        processor.set_custom_array(
            source_values=sparse_values,
            blend_mode=2,
            vertex_indices=vtx_indices,
            channel_indices=channel_indices,
        )

    def blend_weights(self, weights_1d, channel_logical_ids=None, vtx_indices=None, blend_mode=2, alpha=1.0, falloff_weights=None, normalize=False, backup=True):
        """
        [混合/覆写权重] (统一高级接口)
        支持加减乘除、透明度混合、蒙版衰减、自动归一化与稀疏撤销。
        """
        v_count, stride_count, indices_view, _ = self.parse_raw_weights()
        if v_count <= 0:
            return False

        b_view = None
        if channel_logical_ids is not None:
            cols = []
            for logic_id in channel_logical_ids:
                try:
                    cols.append(list(indices_view).index(logic_id))
                except ValueError:
                    pass
            if not cols:
                return False
            b_view = BufferManager.auto(cols, "i").view

        # fmt:off
        v_view   = BufferManager.auto(vtx_indices              , "i").view    if vtx_indices              is not None else None
        src_view = BufferManager.auto(weights_1d               , "f").view
        fal_view = BufferManager.auto(falloff_weights          , "f").view    if falloff_weights          is not None else None
        # fmt:on

        with self.processor_session(backup=backup) as processor:
            if not processor:
                return False

            processor.set_custom_array( source_values   = src_view   ,
                                        blend_mode      = blend_mode ,
                                        vertex_indices  = v_view     ,
                                        channel_indices = b_view     ,
                                        alpha           = alpha      ,
                                        falloff_weights = fal_view   )  # fmt:skip

            if normalize:
                priority = b_view[0] if (b_view is not None and len(b_view) > 0) else -1
                processor.normalize_weights(v_view, priority)

        return True

    def allocate_and_set_weights(self, vtx_count: int, influence_indices: list[int], weights_1d=None, normalize=False, backup=True):
        """
        [全量重建/覆盖图层]
        Python 负责重建 Maya 的底层物理内存和结构,然后移交 Cython 引擎进行纯数据的光速覆写。
        """
        safe_influence_indices = influence_indices if influence_indices is not None else []
        inf_count = len(safe_influence_indices)

        if backup:
            bake_vtx_count, bake_channel_indices, bake_weights_1d = self.get_weights()

            if bake_vtx_count is None or bake_vtx_count <= 0 or bake_channel_indices is None or len(bake_channel_indices) == 0:
                bake_vtx_count = 0
                bake_channel_indices = []
                bake_weights_1d = []

            safe_new_idx = list(safe_influence_indices)
            safe_old_idx = list(bake_channel_indices)

            # 加入安全判断，只有真的有权重数据时才创建深拷贝 array
            safe_new_w = array.array("f", weights_1d) if (weights_1d is not None and len(weights_1d) > 0) else None
            safe_old_w = array.array("f", bake_weights_1d) if (bake_weights_1d is not None and len(bake_weights_1d) > 0) else None

            def redo():
                self.allocate_and_set_weights(vtx_count, safe_new_idx, safe_new_w, normalize, backup=False)

            def undo():
                self.allocate_and_set_weights(bake_vtx_count, safe_old_idx, safe_old_w, backup=False)

            apiundo.commit(redo, undo, execute=False)

        if vtx_count <= 0 or inf_count == 0:
            self.clear()
            return True

        self.resize(vtx_count, inf_count)

        if safe_influence_indices:
            _view = self.memory.view
            _int = _view.cast("B").cast("i")

            _int[2 : 2 + inf_count] = memoryview(array.array("i", safe_influence_indices))

        if weights_1d is not None:
            self.blend_weights(weights_1d, blend_mode=2, normalize=normalize, backup=False)

        return True

    def rebuild_influences(self, new_influence_indices: list[int]):
        """
        支持：骨骼的新增、删除、打乱顺序。
        """
        if not new_influence_indices:
            return False

        v_cnt, old_i_cnt, old_g_bones, old_w_1d = self.parse_raw_weights()
        new_i_cnt = len(new_influence_indices)

        if v_cnt <= 0 or old_i_cnt <= 0 or not old_w_1d:
            new_w_1d = array.array("f", [0.0] * (v_cnt * new_i_cnt))
            return self.allocate_and_set_weights(v_cnt, new_influence_indices, new_w_1d, normalize=True, backup=True)

        bone_mapping = []
        old_bones_list = list(old_g_bones)
        for new_j, bone_id in enumerate(new_influence_indices):
            if bone_id in old_bones_list:
                bone_mapping.append((old_bones_list.index(bone_id), new_j))

        new_w_1d = array.array("f", [0.0] * (v_cnt * new_i_cnt))
        old_view = memoryview(old_w_1d).cast("B").cast("f")
        new_view = memoryview(new_w_1d).cast("B").cast("f")


        for old_j, new_j in bone_mapping:
            new_view[new_j::new_i_cnt] = old_view[old_j::old_i_cnt]

        return self.allocate_and_set_weights(v_cnt, new_influence_indices, new_w_1d, normalize=True, backup=True)

