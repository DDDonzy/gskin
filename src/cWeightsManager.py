from __future__ import annotations


import typing
from dataclasses import dataclass, field

import maya.OpenMaya as om1  # type:ignore

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

    # fmt:off
    mDataHandle  : om1.MDataHandle = None
    mPlug        : om1.MPlug       = None
    mObject_mesh : om1.MObject     = None
    mFnMesh      : om1.MFnMesh     = None

    memory       : CMemoryManager  = None
    """权重数据的底层`CMemoryManager`对象"""

    max_capacity : int             = -1
    """当前寄生 `MFnMesh` 的最大储存数据的长度(也可以理解为预分配数组大小)"""

    length       : int             = -1
    """
    - 当前有效数据长度，对应 `self.memory:CMemoryManager` 的长度
    - 数据结构 [vtx_count:int, influence_count:int, [influences_indices...], [weights...]]
    """
    # fmt:on

    def __init__(self, source: om1.MDataHandle | om1.MPlug):
        if isinstance(source, om1.MPlug):
            self.mPlug = source
            self.mDataHandle = self.mPlug.asMDataHandle()
        elif isinstance(source, om1.MDataHandle):
            self.mPlug = None
            self.mDataHandle = source
        else:
            raise TypeError("Input not OpenMaya.MDataHandle or OpenMaya.MPlug")

        self._setup_mesh_buffer(self.mDataHandle)

    @property
    def is_valid(self) -> bool:
        """检查当前 Handle 是否持有有效"""
        if self.mDataHandle is None:
            return False
        if self.mObject_mesh is None or self.mObject_mesh.isNull():
            return False
        if self.mFnMesh is None:
            return False
        if self.memory is None:
            return False
        if self.memory.view is None:
            return False
        if self.length is None or self.length <= 0:
            return
        return True

    # -------------------------------------------------------------------------
    # 显式工厂方法
    # -------------------------------------------------------------------------

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
        instance = cls(plug)
        return instance

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
        if self.mObject_mesh.isNull():
            return

        self.mFnMesh = om1.MFnMesh(self.mObject_mesh)
        _vtx_count = self.mFnMesh.numVertices()
        if _vtx_count <= 0:
            return None

        self.max_capacity = _vtx_count * 3  # 预分配数组大小
        _ptr = int(self.mFnMesh.getRawPoints())
        _full_memory = CMemoryManager.from_ptr(_ptr, "f", (self.max_capacity,))
        _int_view = _full_memory.view.cast("B").cast("i")

        self.vtx_count = _int_view[0]  # 内存地址第一个数据是 权重vertex_count:int
        self.influence_count = _int_view[1]  # 内存地址第二个数据是 权重influence_count:int

        self.length = (2 + self.influence_count) + (self.vtx_count * self.influence_count)  # 内存有效长度
        self.memory = CMemoryManager.from_ptr(_ptr, "f", (self.length,))  # 生成 cMemoryManager

    def _build_mesh_buffer(self, vtx_count: int):
        """
        负责构建全新的 Mesh 对象，显式返回给调度者

        Return:
            - MObject (om1.MObject): New Meshes's MObject.
            - MFnMesh (om1.MFnMesh): New MFnMesh instance.
            - max_capacity (int): New MFnMesh's buffer max capacity.
        """
        vtx_count = max(3, vtx_count)
        v_count = om1.MIntArray()
        v_list = om1.MIntArray()
        v_count.append(3)
        v_list.append(0)
        v_list.append(1)
        v_list.append(2)

        base_pts = om1.MFloatPointArray()
        base_pts.setLength(vtx_count)

        mObject = om1.MFnMeshData().create()
        mFnMesh = om1.MFnMesh()
        mFnMesh.create(vtx_count, 1, base_pts, v_count, v_list, mObject)

        max_capacity = vtx_count * 3
        return mObject, mFnMesh, max_capacity

    def resize(self, length: int):
        """
        自动扩容底层物理内存空间。

        Side Effects :
            - self.mObject_mesh
            - self.mFnMesh
            - self.max_capacity
            - self.length
            - self.memory
        """
        resized = False

        if length > self.max_capacity:
            (self.mObject_mesh,
             self.mFnMesh     ,
             self.max_capacity) = self._build_mesh_buffer((length + 2) // 3)  # fmt:skip

            self.length = length
            resized = True

        if self.mFnMesh is not None:
            self.memory = CMemoryManager.from_ptr(int(self.mFnMesh.getRawPoints()), "f", (self.length,))

        # self.commit()
        # self.__init__(self.mDataHandle)
        return resized

    def commit(self):
        """强制提交到 Maya，仅作为外部状态刷新接口"""
        if (self.mObject_mesh is not None) and (self.mDataHandle is not None):
            self.mDataHandle.setMObject(self.mObject_mesh)
            if self.mPlug is not None:
                self.mPlug.setMDataHandle(self.mDataHandle)

    def set_weights(
        self,
        vtx_count: int,
        inf_count: int,
        influence_indices: tuple,
        weights_view1D,
    ):
        """[全量写入] 传入顶点总数、骨骼映射表、完整的一维权重数组"""
        influence_count = len(influence_indices)
        header_size = 2 + influence_count
        total_size = header_size + (vtx_count * influence_count)
        # 重构数组大小(resize 会自行判断是否需要重构)
        resized = self.resize(total_size)

        _float_view = self.memory.view
        _int_view = _float_view.cast("B").cast("i")

        _int_view[0] = vtx_count
        _int_view[1] = influence_count

        indices_mgr = CMemoryManager.from_list(list(influence_indices), "i")
        _int_view[2 : 2 + influence_count] = indices_mgr.view

        if isinstance(weights_view1D, memoryview):
            src_view = weights_view1D.cast("B").cast("f")
        else:
            temp_mgr = CMemoryManager.from_list(list(weights_view1D), "f")
            src_view = temp_mgr.view

        _float_view[header_size:total_size] = src_view

        if resized:
            print("commit")
            self.commit()


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
                - vtx_count (int): 当前图层记录的网格顶点总数。
                - influence_count (int): 当前图层包含的有效骨骼列数。
                - influence_indices (tuple[int, ...]): 当前图层包含的骨骼逻辑索引列表 (只读的 Bone IDs)。
                - weights_view (memoryview): 去除 Header 后的纯权重 1D 数据视图 (float32)。
        """
        if not self.is_valid:
            return 0, 0, (), None

        float_view = self.memory.view
        int_view = float_view.cast("B").cast("i")

        vtx_count = int_view[0]
        influence_count = int_view[1]
        header_size = 2 + influence_count

        influence_indices = tuple(int_view[2:header_size])
        weights_view = float_view[header_size : self.length]

        return vtx_count, influence_count, influence_indices, weights_view

    def set_sparse_weights(self, vtx_indices, bone_local_indices, sparse_weights_1d):
        """[稀疏写入] 传入被修改的顶点ID列表、局部骨骼列索引列表、稀疏权重数组 (专供画刷 Undo/Redo)"""
        if not self.is_valid:
            return

        _float_view = self.memory.view
        int_view = _float_view.cast("B").cast("i")

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
                _float_view[dest_idx] = src_view[src_idx]
        return True


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
        self.weights = WeightsHandle(weights_dataHandle)

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

    def bake_to_final_weights(self, vtx_indices: typing.Sequence[int] = None) -> None:
        """
        [终极调度引擎] 全封闭的图层合并与渲染核心。
        """
        active_layers: list[WeightsLayerItem] = []
        combined_bone_set: set[int] = set()
        vtx_count: int = 0

        # ==========================================
        # 1. 骨骼并集提取 (Union)
        # ==========================================
        for idx, item in self.layers.items():
            if item.enabled:
                # 💥 修复 1：去掉了 item.weights 后面的括号，它是一个属性实例！
                w_handle: WeightsHandle = item.weights

                if w_handle and w_handle.is_valid:
                    # 💥 修复 2：正确匹配 get_weights 的 4 个返回值！
                    _vtx_count, _inf_count, inf_indices, _weights_view = w_handle.get_weights()

                    if _vtx_count > 0:
                        vtx_count = max(vtx_count, _vtx_count)  # 保险起见，取所有图层里最大的顶点数
                        combined_bone_set.update(inf_indices)
                        active_layers.append(item)

        if not active_layers or vtx_count == 0:
            return

        out_influence_indices = tuple(sorted(list(combined_bone_set)))
        out_inf_count = len(out_influence_indices)

        # ==========================================
        # 2. 原地重塑 Output Canvas
        # ==========================================
        header_size = 2 + out_inf_count
        weights_size = vtx_count * out_inf_count
        required_length = header_size + weights_size

        # 💥 修复 3：resize 现在只返回一个 bool，绝对不能强行解包给多个变量！
        # 它已经在内部自动更新了 self.weights 的物理指针
        self.weights.resize(required_length)

        _float_view = self.weights.memory.view
        _int_view = _float_view.cast("B").cast("i")

        _int_view[0] = vtx_count
        _int_view[1] = out_inf_count

        # 💥 优化：摒弃 for 循环，直接用你写好的管理器做内存级拷贝
        if out_inf_count > 0:
            indices_mgr = CMemoryManager.from_list(list(out_influence_indices), "i")
            _int_view[2:header_size] = indices_mgr.view

        # ==========================================
        # 3. 底漆重置 (Zeroing) - 极速清理幽灵数据
        # ==========================================
        if weights_size > 0:
            # 💥 优化：放弃调用 CMemoryManager.fill，直接用最暴力的底层字节覆写！
            # 速度远超 Python 循环，且 100% 免疫结构冲突
            _byte_view = _float_view.cast("B")
            start_byte = header_size * 4
            end_byte = required_length * 4

            # 用纯二进制的 0 瞬间填满整个 Payload 区
            _byte_view[start_byte:end_byte] = b"\x00" * (weights_size * 4)

        # ==========================================
        # 4. Cython 黑盒混合
        # ==========================================
        vtx_indices_view = None
        if vtx_indices is not None and len(vtx_indices) > 0:
            vtx_indices_view = CMemoryManager.from_list(list(vtx_indices), "i").view

        for item in active_layers:
            layer_view = item.weights.memory.view

            m_handle = item.mask
            mask_view = None
            if m_handle and m_handle.is_valid and m_handle.length > 0:
                mask_view = m_handle.memory.view

            cWeightsCoreCython.blend_layer_raw_view(
                _float_view,
                layer_view,
                mask_view,
                1.0,
                vtx_indices_view,
            )

        # ==========================================
        # 5. 💥 致命一击：提交写入
        # ==========================================
        # 这是之前漏掉的最关键一步！Cython 在后台算完并写满物理内存后，
        # 必须通知 Maya，让这个合法的 MObject 挂载到节点管道上。
        self.weights.commit()

    def update(self):
        self.cSkin.setDirty()
