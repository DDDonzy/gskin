from __future__ import annotations


import typing
import array
import functools

import maya.OpenMaya as om1  # type:ignore

from . import cWeightsCoreCython
from .cMemoryView import CMemoryManager, ensure_memoryview
from ._cRegistry import SkinRegistry
from . import apiundo


if typing.TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer  # type: ignore


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


def updateDG(func):
    """
    [装饰器] 方法执行完毕后，强制调用实例的 self.update() 刷新 Maya 视口。
    无论函数是正常 return 还是引发异常，都能保证 100% 触发。
    """

    @functools.wraps(func)
    def wrapper(self: WeightsManager, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            # 这里的 self 就是 WeightsManager 实例
            self.updateDG()

    return wrapper


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

    @updateDG
    def set_weights(self, layer_idx, is_mask, vtx_indices, bone_local_indices, weights_1d, backup=True):
        handle = self.get_handle(layer_idx, is_mask)
        if handle is None or handle.is_valid is not True:
            raise RuntimeError(f"{handle} is not valid")

        # --- [入口：统一转为视图] ---
        v_indices = ensure_memoryview(vtx_indices, "i")
        b_indices = ensure_memoryview(bone_local_indices, "i")
        src_view = ensure_memoryview(weights_1d, "f")

        weights = handle.memory.view
        _int_view = weights.cast("B").cast("i")
        inf_count = _int_view[1]
        header_size = 2 + inf_count

        # --- [备份：精准转为独立内存] ---
        if backup:
            # 只有开启备份时，才付出转换 array 的开销
            safe_vtx = array.array("i", v_indices)
            safe_bone = array.array("i", b_indices)
            safe_new_weights = array.array("f", src_view)

            # 记录旧数据
            num_vtx = len(v_indices)
            num_bones = len(b_indices)
            backup_values = array.array("f", [0.0] * (num_vtx * num_bones))

            # 写入的同时提取备份
            for i in range(num_vtx):
                vtx_id = v_indices[i]
                row_offset = header_size + vtx_id * inf_count
                for j in range(num_bones):
                    dest_idx = row_offset + b_indices[j]
                    src_idx = i * num_bones + j
                    backup_values[src_idx] = weights[dest_idx]
                    weights[dest_idx] = src_view[src_idx]

            # 提交 Undo 闭包 (使用刚才“转”好的安全 array)
            def redo():
                self.set_weights(layer_idx, is_mask, safe_vtx, safe_bone, safe_new_weights, backup=False)

            def undo():
                self.set_weights(layer_idx, is_mask, safe_vtx, safe_bone, backup_values, backup=False)

            apiundo.commit(redo, undo, execute=False)
            return v_indices, b_indices, backup_values

        else:
            # --- [快速路径：不备份直接刷] ---
            for i in range(len(v_indices)):
                vtx_id = v_indices[i]
                row_offset = header_size + vtx_id * inf_count
                for j in range(len(b_indices)):
                    weights[row_offset + b_indices[j]] = src_view[i * len(b_indices) + j]

        return None, None, None

    @updateDG
    def set_weights_all(self, layer_idx, is_mask, vtx_count, inf_count, influence_indices, weights_1d, backup=True):
        handle = self.get_handle(layer_idx, is_mask)
        if handle is None:
            return False

        # --- [备份：利用解析好的 safe 视图进行转换] ---
        if backup:
            # 1. 获取旧状态 (get_weights 内部已经做好了深拷贝，返回的是绝对安全的 memoryview)
            old_v, old_i, safe_old_indices_view, safe_old_weights_view = self.get_weights(layer_idx, is_mask)

            safe_new_indices = array.array("i", influence_indices)
            safe_new_weights = array.array("f", ensure_memoryview(weights_1d, "f"))

            # 3. 注册闭包
            def redo():
                self.set_weights_all(layer_idx, is_mask, vtx_count, inf_count, safe_new_indices, safe_new_weights, backup=False)

            def undo():
                self.set_weights_all(layer_idx, is_mask, old_v, old_i, safe_old_indices_view, safe_old_weights_view, backup=False)

            apiundo.commit(redo, undo, execute=False)

        # 如果传进来的 vtx_count 为 0，说明这是一个“清空图层”或“撤销回到建图层前”的动作！
        if vtx_count == 0:
            handle.clear()
            handle.commit()
            return True

        # --- [执行：直接利用原始输入写入] ---
        src_view = ensure_memoryview(weights_1d, "f")
        total_size = (2 + len(influence_indices)) + (vtx_count * len(influence_indices))

        resized = handle.resize(total_size)
        if not handle.is_valid:
            return False

        _view = handle.memory.view
        _int = _view.cast("B").cast("i")

        _int[0], _int[1] = vtx_count, len(influence_indices)
        if influence_indices:
            _int[2 : 2 + len(influence_indices)] = ensure_memoryview(influence_indices, "i")

        _view[2 + len(influence_indices) : total_size] = src_view

        if resized:
            handle.commit()
        return True

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
        return handle.memory.view[: handle.length]

    def get_weights(self, layer_idx: int, is_mask: bool):
        """
        获取全部数据的,并统一重新包装为 memoryview 返回。
        Returns:
            tuple: (vtx_count: int, influence_count: int,
                    safe_indices_view: memoryview,
                    safe_weights_view: memoryview)
        """
        raw_view = self.get_raw_weights(layer_idx, is_mask)
        if raw_view is None:
            # 如果是空图层，返回安全的、空的 memoryview
            return 0, 0, memoryview(array.array("i")), memoryview(array.array("f"))

        vtx_count, inf_count, inf_indices_view, weights_view = self.parse_raw_weights(raw_view)

        if inf_indices_view is not None and len(inf_indices_view) > 0:
            safe_indices_view = memoryview(array.array("i", inf_indices_view))
        else:
            safe_indices_view = memoryview(array.array("i"))

        if weights_view is not None and len(weights_view) > 0:
            safe_weights_view = memoryview(array.array("f", weights_view))
        else:
            safe_weights_view = memoryview(array.array("f"))

        return vtx_count, inf_count, safe_indices_view, safe_weights_view

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

        self.weights.commit()

    def updateDG(self):
        self.cSkin.setDirty()
