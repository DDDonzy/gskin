from __future__ import annotations


import typing
import array
import functools

import maya.OpenMaya as om1  # type:ignore

from . import apiundo
from . import cBrushCoreCython
from ._cRegistry import SkinRegistry
from .cBufferManager import BufferManager, ensure_memoryview


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


class WeightsHandle:
    """
    权重数据装配器
    采用 Header (int32) + Payload (float32) 混合内存布局。
        ```
        ----------------------------------------------------------------------------------
        | vtx_count (int) | inf_count (int) | bone_indices... (int) | Weights... (float) |
        | <------------------ Header -----------------------------> | <--- Payload ----> |
        ----------------------------------------------------------------------------------
        ```
    """

    # fmt:off
    mDataHandle  : om1.MDataHandle = None
    mPlug        : om1.MPlug       = None
    mObject_mesh : om1.MObject     = None
    mFnMesh      : om1.MFnMesh     = None

    memory       : BufferManager  = None
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
            # 提取短命的 MDataHandle
            # 警告：这里会触发节点求值！
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
            return False
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
        _full_memory = BufferManager.from_ptr(_ptr, "f", (self.max_capacity,))
        _int_view = _full_memory.view.cast("B").cast("i")

        self.vtx_count = _int_view[0]  # 内存地址第一个数据是 权重vertex_count:int
        self.influence_count = _int_view[1]  # 内存地址第二个数据是 权重influence_count:int

        self.length = (2 + self.influence_count) + (self.vtx_count * self.influence_count)  # 内存有效长度
        self.memory = BufferManager.from_ptr(_ptr, "f", (self.length,))  # 生成 cMemoryManager

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
            self.memory = BufferManager.from_ptr(int(self.mFnMesh.getRawPoints()), "f", (self.length,))

        return resized

    def clear(self):

        self.mObject_mesh = om1.MFnMeshData().create()
        self.mFnMesh = None
        self.memory = None
        self.length = -1
        self.max_capacity = -1

        return True

    def commit(self):
        """强制提交到 Maya，仅作为外部状态刷新接口"""
        if (self.mObject_mesh is not None) and (self.mDataHandle is not None):
            self.mDataHandle.setMObject(self.mObject_mesh)
            if self.mPlug is not None:
                self.mPlug.setMDataHandle(self.mDataHandle)


class WeightsLayerItem:
    def __init__(self, cSkin: CythonSkinDeformer, mDataHandle: om1.MDataHandle, logical_idx: int = -1):
        self.cSkin = cSkin
        self.logical_idx = logical_idx

        _handle_weights = mDataHandle.child(cSkin.aLayerWeights)
        _handle_mask = mDataHandle.child(cSkin.aLayerMask)
        _handle_enabled = mDataHandle.child(cSkin.aLayerEnabled)

        self.enabled = _handle_enabled.asBool()
        self.weights = WeightsHandle(_handle_weights)
        self.mask = WeightsHandle(_handle_mask)

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
        提取指定图层的物理内存，并为其实例 `SkinWeightProcessor`。
        """
        _raw_view = self.get_raw_weights(layer_idx, is_mask)
        if _raw_view is None:
            return None

        vtx_count, inf_count, _, weights_1d = self.parse_raw_weights(_raw_view)

        weights_2d = weights_1d.cast("B").cast("f", (vtx_count, inf_count))

        tmp_idx = array.array("i", [0]) * vtx_count
        tmp_bool = array.array("B", [0]) * vtx_count
        tmp_locks = array.array("B", [0]) * inf_count

        # 分配 Undo 内存 (先拉平为 1D，再利用 memoryview 强转为 2D)
        _undo_buffer = array.array("f", [0.0]) * (vtx_count * inf_count)
        tmp_undo_view = memoryview(_undo_buffer).cast("B").cast("f", shape=(vtx_count, inf_count))

        processor = cBrushCoreCython.SkinWeightProcessor(
            weights_2d,
            memoryview(tmp_idx),
            memoryview(tmp_bool),
            memoryview(tmp_locks),
            tmp_undo_view,
        )
        return processor

    @updateDG
    def set_sparse_data(self, layer_idx: int, is_mask: bool, vtx_indices, channel_indices, sparse_values):
        """
        专供 Undo / Redo 闭包调用。
        直接使用 C 级覆盖能力还原快照，彻底告别 Python 循环！
        """
        processor = self._create_processor(layer_idx, is_mask)
        if not processor:
            return

        # 撤销时不需要记录新的 Undo 快照，也不需要复杂的模式，直接暴力 Replace (blend_mode=2)
        processor.set_custom_array(source_values=sparse_values, blend_mode=2, vertex_indices=vtx_indices, channel_indices=channel_indices)

    @updateDG
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
        v_view = ensure_memoryview(vtx_indices, "i") if vtx_indices is not None else None
        b_view = ensure_memoryview(locked_influence_indices, "i") if locked_influence_indices is not None else None
        src_view = ensure_memoryview(weights_1d, "f")
        fal_view = ensure_memoryview(falloff_weights, "f") if falloff_weights is not None else None

        # 开启快照录制
        if backup:
            processor.begin_stroke()

        # Cython 执行带混合模式的覆写
        processor.set_custom_array(source_values=src_view, blend_mode=blend_mode, vertex_indices=v_view, channel_indices=b_view, alpha=alpha, falloff_weights=fal_view)

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
                    self.set_sparse_data(layer_idx, is_mask, mod_vtx, mod_ch, new_sparse)

                def undo():
                    self.set_sparse_data(layer_idx, is_mask, mod_vtx, mod_ch, old_sparse)

                apiundo.commit(redo, undo, execute=False)

        return True

    @updateDG
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
            old_v_cnt, old_i_cnt, old_g_bones, _, _, old_w_1d = self.get_weights(layer_idx, is_mask)

            safe_new_idx = array.array("i", influence_indices)
            safe_new_w = array.array("f", ensure_memoryview(weights_1d, "f"))

            def redo():
                self.rebuild_layer(layer_idx, is_mask, vtx_count, inf_count, safe_new_idx, safe_new_w, backup=False)

            def undo():
                self.rebuild_layer(layer_idx, is_mask, old_v_cnt, old_i_cnt, old_g_bones, old_w_1d, backup=False)

            apiundo.commit(redo, undo, execute=False)

        if vtx_count == 0:
            handle.clear()
            handle.commit()
            return True

        # --- [2. Python 负责重建底层物理内存与骨骼 Header (搭地基)] ---
        total_size = (2 + len(influence_indices)) + (vtx_count * len(influence_indices))
        resized = handle.resize(total_size)
        if not handle.is_valid:
            return False

        _view = handle.memory.view
        _int = _view.cast("B").cast("i")
        _int[0], _int[1] = vtx_count, len(influence_indices)
        if influence_indices:
            _int[2 : 2 + len(influence_indices)] = ensure_memoryview(influence_indices, "i")

        # --- [3. 召唤 Cython 引擎接管纯数据写入 (填数据)] ---
        processor = self._create_processor(layer_idx, is_mask)
        if processor:
            processor.set_custom_array(source_values=ensure_memoryview(weights_1d, "f"), blend_mode=2, vertex_indices=None, channel_indices=None)

        if resized:
            handle.commit()
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

        v_view = ensure_memoryview(vtx_indices, "i") if vtx_indices is not None else None
        b_view = ensure_memoryview(bone_local_indices, "i") if bone_local_indices is not None else None

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
