from __future__ import annotations


import math
import typing
from dataclasses import dataclass, field

import maya.OpenMaya as om1

from . import cWeightsCoreCython
from .cMemoryView import CMemoryManager
from ._cRegistry import SkinRegistry


if typing.TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer  # type: ignore


class WeightsHandle:
    """
    [底层核心] 权重数据装配器
    采用 Header (int32) + Payload (float32) 混合内存布局。
    严格贯彻显式赋值，所有隐式属性计算均通过 return 暴露给调用者。
    """

    def __init__(self):
        self.plug: om1.MPlug = None
        self.data_handle: om1.MDataHandle = None

        # 核心数据：由外部显式装配
        self.mObj_mesh: om1.MObject = None
        self.fn_mesh: om1.MFnMesh = None
        self._is_plug_mode = False
        self.max_capacity = 0
        """当前寄生 `MFnMesh` 的最大储存数据的长度(也可以理解为预分配数组大小)"""
        self.length = 0
        """
        - 当前有效数据长度，对应 `self.memory:CMemoryManager` 的长度
        - 数据结构 [vtx_count:int, influence_count:int, [influences_indices...], [weights...]]
        """
        self.memory: CMemoryManager = None
        """权重数据的底层`CMemoryManager`对象"""

    @property
    def is_valid(self) -> bool:
        """检查当前 Handle 是否持有有效的底层内存"""
        return self.memory is not None

    # -------------------------------------------------------------------------
    # 显式工厂方法
    # -------------------------------------------------------------------------
    @classmethod
    def from_plug(cls, plug: om1.MPlug):
        instance = cls()
        instance.plug = plug
        instance._is_plug_mode = True
        try:
            instance.mObj_mesh = plug.asMObject()
        except RuntimeError:
            instance.mObj_mesh = om1.MObject()

        # 💥 显式装配：将加工厂计算出的属性挂载到实例上
        (
            instance.mObj_mesh,
            instance.fn_mesh,
            instance.max_capacity,
            instance.length,
            instance.memory,
        ) = instance._setup_mesh()
        return instance

    @classmethod
    def from_data_handle(cls, data_handle: om1.MDataHandle):
        instance = cls()
        instance.data_handle = data_handle
        instance._is_plug_mode = False
        try:
            instance.mObj_mesh = data_handle.asMesh()
        except RuntimeError:
            instance.mObj_mesh = om1.MObject()

        # 💥 显式装配
        (
            instance.mObj_mesh,
            instance.fn_mesh,
            instance.max_capacity,
            instance.length,
            instance.memory,
        ) = instance._setup_mesh()
        return instance

    @classmethod
    def from_attr_string(cls, attr_path: str):
        """通过字符串路径直接获取 Handle，严格遵循显式装配规范"""
        sel = om1.MSelectionList()
        try:
            sel.add(attr_path)
        except RuntimeError:
            raise ValueError(f"找不到指定的属性路径: {attr_path}")

        plug = om1.MPlug()
        sel.getPlug(0, plug)

        # 显式实例化与底层变量初始化
        instance = cls()
        instance.plug = plug
        instance._is_plug_mode = True
        try:
            instance.mObj_mesh = plug.asMObject()
        except RuntimeError:
            instance.mObj_mesh = om1.MObject()

        # 💥 显式装配：将加工厂计算出的属性挂载到实例上
        (
            instance.mObj_mesh,
            instance.fn_mesh,
            instance.max_capacity,
            instance.length,
            instance.memory,
        ) = instance._setup_mesh()

        return instance

    # -------------------------------------------------------------------------
    # 数据加工厂 (纯函数模式，绝不修改 self)
    # -------------------------------------------------------------------------
    def _setup_mesh(self):
        """初始化 mFnMesh，并显式返回所有计算好的核心数据"""
        mObj_mesh = self.mObj_mesh
        fn_mesh = self.fn_mesh

        if (fn_mesh is None) and (mObj_mesh) and (not mObj_mesh.isNull()) and (mObj_mesh.hasFn(om1.MFn.kMesh)):
            fn_mesh = om1.MFnMesh(mObj_mesh)

        # 显式传递参数给下级加工厂，并接收返回
        (
            max_capacity,
            length,
            memory,
        ) = self._init_lengths(fn_mesh)

        # 统一打包返回，绝不在此函数内隐式赋值
        return mObj_mesh, fn_mesh, max_capacity, length, memory

    def _setup_mesh_buffer(self, mDataHandle: om1.MDataHandle):
        """
        从MDataHandle解析权重数据

        Side Effects :
            - self.mObject_mesh
            - self.mFnMesh
            - self.max_capacity
            - self.vtx_count
            - self.influence_count
            - self.length
            - self.memory
        """
        self.mObject_mesh: om1.MObject = mDataHandle.asMesh()
        self.mFnMesh = om1.MFnMesh(self.mObject_mesh)
        _vtx_count = self.mFnMesh.numVertices()
        self.max_capacity = _vtx_count * 3  # 预分配数组大小
        _ptr = self.mFnMesh.getRawPoints()
        _full_memory = CMemoryManager.from_ptr(_ptr, "f", (self.max_capacity,))
        _int_view = _full_memory.cast("B").cast("i")

        self.vtx_count = _int_view[0]  # 内存地址第一个数据是 权重vertex_count:int
        self.influence_count = _int_view[1]  # 内存地址第二个数据是 权重influence_count:int
        self.length = (2 + self.influence_count) + (self.vtx_count * self.influence_count)  # 内存有效长度
        self.memory = CMemoryManager.from_ptr(_ptr, "f", (self.length,))  # 生成 cMemoryManager

    def _init_lengths(self, fn_mesh: om1.MFnMesh):
        """计算并严格返回 (max_capacity, length, memory)"""
        if fn_mesh is None or fn_mesh.numVertices() == 0:
            return 0, 0, None

        max_capacity = fn_mesh.numVertices() * 3
        if max_capacity == 0:
            return 0, 0, None

        ptr_addr = int(fn_mesh.getRawPoints())
        full_memory = CMemoryManager.from_ptr(ptr_addr, "f", (max_capacity,))
        float_view = full_memory.view

        # 强转指针读取 Header(vtx_count, influence_count)
        int_view = float_view.cast("B").cast("i")
        vtx_count = int_view[0]
        influence_count = int_view[1]

        length = (2 + influence_count) + (vtx_count * influence_count)

        # 正常生成并返回
        memory = CMemoryManager.from_ptr(ptr_addr, "f", (length,))
        return max_capacity, length, memory

    def _build_mesh_buffer(self, vtx_count: int):
        """负责构建全新的 Mesh 对象，显式返回给调度者"""
        vtx_count = max(3, vtx_count)
        v_count = om1.MIntArray()
        v_list = om1.MIntArray()
        v_count.append(3)
        v_list.append(0)
        v_list.append(1)
        v_list.append(2)

        base_pts = om1.MFloatPointArray()
        base_pts.setLength(vtx_count)

        mesh_data_obj = om1.MFnMeshData().create()
        new_mesh_fn = om1.MFnMesh()
        new_mesh_fn.create(vtx_count, 1, base_pts, v_count, v_list, mesh_data_obj)

        max_capacity = vtx_count * 3
        return mesh_data_obj, new_mesh_fn, max_capacity

    def resize(self, required_length: int):
        """
        自动扩容底层物理内存空间。

        Side Effects :
            - self.mObj_mesh
            - self.fn_mesh
            - self.max_capacity
            - self.length
            - self.memory
        """

        if required_length > self.max_capacity:
            dummy_vtx_count = int(math.ceil(required_length / 3.0))
            # 显式接收重新构建的 Mesh 数据
            (
                self.mObj_mesh,
                self.fn_mesh,
                self.max_capacity,
            ) = self._build_mesh_buffer(dummy_vtx_count)

            self.length = required_length

        if self.fn_mesh:
            ptr_addr = int(self.fn_mesh.getRawPoints())
            self.memory = CMemoryManager.from_ptr(ptr_addr, "f", (self.length,))

    def commit(self):
        """强制提交到 Maya，仅作为外部状态刷新接口"""
        if self.mObj_mesh:
            if self._is_plug_mode:
                print("[commit]: update object --- plug Mode")
                self.plug.setMObject(self.mObj_mesh)
            else:
                print("[commit]: update object --- dataHandle Mode")
                self.data_handle.setMObject(self.mObj_mesh)

    # -------------------------------------------------------------------------
    # 核心读写接口 (全量 / 稀疏)
    # -------------------------------------------------------------------------

    def set_weights(self, vtx_count: int, influence_indices: tuple, raw_weights_1d):
        """[全量写入] 传入顶点总数、骨骼映射表、完整的一维权重数组"""
        influence_count = len(influence_indices)
        header_size = 2 + influence_count
        total_size = header_size + (vtx_count * influence_count)
        # 重构数组大小(resize 会自行判断是否需要重构)
        self.resize(total_size)

        float_view = self.memory.view
        int_view = float_view.cast("B").cast("i")

        int_view[0] = vtx_count
        int_view[1] = influence_count
        for i, bone_id in enumerate(influence_indices):
            int_view[2 + i] = bone_id

        if isinstance(raw_weights_1d, memoryview):
            src_view = raw_weights_1d.cast("B").cast("f")
        else:
            temp_mgr = CMemoryManager.from_list(list(raw_weights_1d), "f")
            src_view = temp_mgr.view

        float_view[header_size:total_size] = src_view

    def get_weights(self):
        """极速解析底层连续内存，分离 Header (元数据) 与 Payload (纯权重)。

        底层的物理内存是一块连续的 32 位数据块，通过 C 指针强转进行零拷贝读取。
        其物理内存布局如下：
        ```
        ----------------------------------------------------------------------------------
        | vtx_count (int) | inf_count (int) | bone_indices... (int) | Weights... (float) |
        | <------------------ Header -----------------------------> | <--- Payload ----> |
        ----------------------------------------------------------------------------------
        ```

        Returns:
            tuple: 若当前句柄无效或内存数据为空，则返回 (None, 0, 0, ()), 否则返回包含以下元素的元组:
                - pure_weights_view (memoryview): 去除 Header 后的纯权重 1D 数据视图 (float32)。
                - vtx_count (int): 当前图层记录的网格顶点总数。
                - influence_count (int): 当前图层包含的有效骨骼列数。
                - influence_indices (tuple[int, ...]): 当前图层包含的骨骼逻辑索引列表 (只读的 Bone IDs)。
        """
        if not self.is_valid or self.length == 0:
            return None, 0, 0, ()

        float_view = self.memory.view
        int_view = float_view.cast("B").cast("i")

        vtx_count = int_view[0]
        influence_count = int_view[1]
        header_size = 2 + influence_count

        influence_indices = tuple(int_view[2:header_size])
        pure_weights_view = float_view[header_size : self.length]

        return pure_weights_view, vtx_count, influence_count, influence_indices

    def set_sparse_weights(self, vtx_indices, bone_local_indices, sparse_weights_1d):
        """[稀疏写入] 传入被修改的顶点ID列表、局部骨骼列索引列表、稀疏权重数组 (专供画刷 Undo/Redo)"""
        if not self.is_valid or self.length == 0 or not vtx_indices or not bone_local_indices:
            return

        float_view = self.memory.view
        int_view = float_view.cast("B").cast("i")

        influence_count = int_view[1]
        header_size = 2 + influence_count

        if isinstance(sparse_weights_1d, memoryview):
            src_view = sparse_weights_1d.cast("B").cast("f")
        else:
            src_view = memoryview(sparse_weights_1d).cast("B").cast("f")

        num_modified_bones = len(bone_local_indices)

        for i, vtx_id in enumerate(vtx_indices):
            row_start = header_size + vtx_id * influence_count
            for j, local_col_id in enumerate(bone_local_indices):
                dest_idx = row_start + local_col_id
                src_idx = i * num_modified_bones + j
                float_view[dest_idx] = src_view[src_idx]

        self.commit()


@dataclass
class WeightsLayerItem:
    logical_idx: int = -1

    weights: WeightsHandle = None
    mask: WeightsHandle = None
    enabled: bool = None

    mHandle_weights: om1.MDataHandle = field(default=None, repr=False)
    mHandle_mask: om1.MDataHandle = field(default=None, repr=False)
    mHandle_enabled: om1.MDataHandle = field(default=None, repr=False)

    cSkin: CythonSkinDeformer = field(default=None, repr=False)

    def __init__(self, cSkin: CythonSkinDeformer, mDataHandle: om1.MDataHandle, logical_idx=-1) -> None:
        self.cSkin = cSkin
        self.logical_idx = logical_idx
        self.mHandle_mask = mDataHandle.child(cSkin.aLayerMask)
        self.mHandle_weights = mDataHandle.child(cSkin.aLayerWeights)
        self.mHandle_enabled = mDataHandle.child(cSkin.aLayerEnabled)

        self.weights = WeightsHandle.from_data_handle(self.mHandle_weights)
        self.mask = WeightsHandle.from_data_handle(self.mHandle_mask)
        self.enabled = self.mHandle_enabled.asBool()

    @property
    def mPlug_weights(self):
        mPlug = om1.MPlug(self.cSkin.mObject, self.mHandle_weights.attribute())
        mPlug.selectAncestorLogicalIndex(self.logical_idx, self.cSkin.aLayerCompound)
        return mPlug

    @property
    def mPlug_mask(self):
        mPlug = om1.MPlug(self.cSkin.mObject, self.mHandle_mask.attribute())
        mPlug.selectAncestorLogicalIndex(self.logical_idx, self.cSkin.aLayerCompound)
        return mPlug

    @property
    def mPlug_enabled(self):
        mPlug = om1.MPlug(self.cSkin.mObject, self.mHandle_enabled.attribute())
        mPlug.selectAncestorLogicalIndex(self.logical_idx, self.cSkin.aLayerCompound)
        return mPlug


@dataclass
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

    @classmethod
    def get_manager_from_cSkin(cls, cSkinNodeName: str):
        cSkin: CythonSkinDeformer = SkinRegistry.from_instance_by_string(cSkinNodeName)
        return cSkin.weights_manager

    def update_data(self, dataBlock: om1.MDataBlock):
        """
        [状态同步器]
        一次性扫描 Maya 节点，刷新所有 Plug 缓存与底层内存池。
        当你在 UI 层面添加、删除了图层，或改变了节点连接后，手动调用此函数。
        """
        # weights
        weights_dataHandle = dataBlock.inputValue(self.cSkin.aWeights)
        self.handle = weights_dataHandle
        self.weights = WeightsHandle.from_data_handle(weights_dataHandle)

        # layer
        self.layers.clear()
        mArrayDataHandle: om1.MArrayDataHandle = dataBlock.inputArrayValue(self.cSkin.aLayerCompound)
        _count = mArrayDataHandle.elementCount()
        for idx in range(_count):
            mArrayDataHandle.jumpToArrayElement(idx)
            logical_idx = mArrayDataHandle.elementIndex()
            element_handle = mArrayDataHandle.inputValue()
            self.layers[logical_idx] = WeightsLayerItem(self.cSkin, element_handle, logical_idx)

    @property
    def layer_indices(self) -> list[int]:
        return list(self.layers.keys())

    def get_layer(self, logical_index: int = None, physical_index: int = None) -> "WeightsLayerItem":
        """
        获取图层实例，支持通过逻辑索引或物理索引进行查询。
        注意：每次调用请只传入其中一个参数。

        Args:
            logical_index (int, optional): 节点上的真实稀疏索引 (如 cWeightsLayers[5] 里的 5)。O(1) 极速查询。
            physical_index (int, optional): 当前内存数组/UI列表中的实际排列顺序 (第 N 个图层)。O(N) 顺序查询。

        Returns:
            WeightsLayerItem: 找到的图层实例，如果越界或找不到则返回 None。
        """
        if logical_index is not None:
            return self.layers.get(logical_index, None)
        if physical_index is not None:
            if physical_index < 0 or physical_index >= len(self.layers):
                return None
            return list(self.layers.values())[physical_index]
        return None

    # -------------------------------------------------------------------------
    # 核心：纯享版的零 API 烘焙热循环
    # -------------------------------------------------------------------------
    def bake_to_final_weights(self, vtx_indices: typing.Sequence[int] = None) -> None:
        """
        [终极调度引擎] 全封闭的图层合并与渲染核心。

        彻底断绝与 Maya API 的实时交互（避开 findPlug 与字符串寻址的高昂开销）。
        完全依赖初始化时建立的 CMemoryManager 内存池，配合 Cython 进行极其暴力的原地物理内存覆写。

        工作流 (Workflow):
            1. 骨骼并集提取 (Union): 极速遍历缓存的可用图层，提取出所有被影响的骨骼并集 (Influence Union)。
            2. 画布重塑 (Resize Canvas): 触发输出层 (cWeights) 底层的动态扩容，确保矩阵宽度足以容纳所有骨骼。
            3. 底漆重置 (Zeroing): 利用 C 级别的 memset 瞬间将输出层的纯数据区 (Payload) 归零，彻底消灭旧数据幽灵。
            4. 黑盒混合 (Cython Blending): 将缓存的源图层内存指针、遮罩指针直接投喂给 Cython。
               基于数学公式 `Output = Output * (1 - Mask*Alpha) + Layer * Mask*Alpha` 原地累加。

        Args:
            vtx_indices (Sequence[int], optional): 稀疏更新的顶点 ID 列表 (局部重算)。
                - 传入有效列表时: 仅重算并覆写指定的顶点，专供画刷实时涂抹或 Undo/Redo 时实现极限帧率。
                - 传入 None 或空列表时: 执行 100% 全量顶点的遍历与烘焙 (向下合并图层或图层开关切换时使用)。

        Returns:
            None: 无返回值。计算结果将通过 C 指针直接强行写入 Maya 节点的 `cWeights` 物理内存中。
        """

        active_layers: list[WeightsLayerItem] = []
        combined_bone_set: set[int] = set()
        vtx_count: int = 0

        # 骨骼集
        for idx, item in self.layers.items():
            if item.enabled:
                w_handle: WeightsHandle = item.weights()
                if w_handle and w_handle.is_valid:
                    active_layers.append(item)

                    _, vtx_count, _, inf_indices = w_handle.get_weights()
                    if vtx_count > 0:
                        vtx_count = vtx_count
                        combined_bone_set.update(inf_indices)

        if not active_layers or vtx_count == 0:
            return

        out_influence_indices = tuple(sorted(list(combined_bone_set)))
        out_inf_count = len(out_influence_indices)

        # 2. 原地重塑 Output Canvas
        header_size = 2 + out_inf_count
        weights_size = vtx_count * out_inf_count
        required_length = header_size + weights_size

        (
            self.weights.mObj_mesh,
            self.weights.fn_mesh,
            self.weights.max_capacity,
            self.weights.length,
            self.weights.memory,
        ) = self.weights.resize(required_length)

        int_view = self.weights.memory.view.cast("B").cast("i")
        int_view[0] = vtx_count
        int_view[1] = out_inf_count

        for i, bone_id in enumerate(out_influence_indices):
            int_view[2 + i] = bone_id

        # 极速清理幽灵数据
        if weights_size > 0:
            header_byte_offset = header_size * 4
            payload_ptr = self.weights.memory.ptr + header_byte_offset
            CMemoryManager.from_ptr(payload_ptr, "f", (weights_size,)).fill(0.0)

        output_view = self.weights.memory.view

        # 3. 准备稀疏顶点视图
        vtx_indices_view = None
        if vtx_indices is not None and len(vtx_indices) > 0:
            vtx_indices_view = CMemoryManager.from_list(list(vtx_indices), "i").view

        # 4. Cython 完美接棒
        for item in active_layers:
            layer_view = item.weights.memory.view

            m_handle = item.mask
            mask_view = None
            if m_handle and m_handle.is_valid and m_handle.length > 0:
                mask_view = m_handle.memory.view

            cWeightsCoreCython.blend_layer_raw_view(
                output_view,
                layer_view,
                mask_view,
                1.0,
                vtx_indices_view,
            )

    def update(self):
        self.cSkin.setDirty()
