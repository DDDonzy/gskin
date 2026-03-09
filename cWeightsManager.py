import math
import maya.OpenMaya as om1  # type: ignore

from .cMemoryView import CMemoryManager
from . import cWeightsCoreCython

# =========================================================================
# 1. 基础数据容器与底层装配器 (建议仅内部使用)
# =========================================================================


class WeightsHandle:
    """
    [底层核心] 权重数据装配器
    采用 Header (int32) + Payload (float32) 混合内存布局。
    零拷贝解析顶点数和骨骼映射，支持多层局部压缩。
    """

    def __init__(self):
        self.plug = None
        self.data_handle = None
        self.mObj_mesh = None
        self.fn_mesh = None

        self._is_plug_mode = False
        self.max_capacity = 0

        self.length = 0
        self.memory: CMemoryManager = None

    @classmethod
    def from_plug(cls, plug: om1.MPlug):
        instance = cls()
        instance.plug = plug
        instance._is_plug_mode = True
        try:
            instance.mObj_mesh = plug.asMObject()
        except RuntimeError:
            instance.mObj_mesh = om1.MObject()

        instance._setup_mesh()
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

        instance._setup_mesh()
        return instance

    def _setup_mesh(self):
        if self.fn_mesh is None and self.mObj_mesh and not self.mObj_mesh.isNull() and self.mObj_mesh.hasFn(om1.MFn.kMesh):
            self.fn_mesh = om1.MFnMesh(self.mObj_mesh)
        self._init_lengths()

    def _init_lengths(self):
        if self.fn_mesh is None or self.fn_mesh.numVertices() == 0:
            return

        self.max_capacity = self.fn_mesh.numVertices() * 3
        if self.max_capacity == 0:
            return

        ptr_addr = int(self.fn_mesh.getRawPoints())
        full_memory = CMemoryManager.from_ptr(ptr_addr, "f", (self.max_capacity,))
        float_view = full_memory.view
        int_view = float_view.cast("b").cast("i")

        vtx_count = int_view[0]
        influence_count = int_view[1]

        if vtx_count <= 0 or influence_count <= 0 or influence_count > 20000:
            self.length = 0
            self.memory = CMemoryManager.from_ptr(ptr_addr, "f", (0,))
            return

        header_size = 2 + influence_count
        payload_size = vtx_count * influence_count
        real_length = header_size + payload_size

        if real_length > self.max_capacity:
            self.length = 0
            self.memory = CMemoryManager.from_ptr(ptr_addr, "f", (0,))
            return

        self.length = real_length
        self.memory = CMemoryManager.from_ptr(ptr_addr, "f", (self.length,))

    @property
    def is_valid(self):
        return (self.fn_mesh is not None) and (self.max_capacity > 0) and (self.memory is not None)

    def _rebuild_mesh(self, vtx_count: int):
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

        self.mObj_mesh = mesh_data_obj
        self.fn_mesh = new_mesh_fn
        self.max_capacity = vtx_count * 3

        if self._is_plug_mode:
            self.plug.setMObject(mesh_data_obj)
        elif self.data_handle is not None:
            self.data_handle.setMObject(mesh_data_obj)

    def resize(self, required_length: int):
        if required_length > self.max_capacity:
            dummy_vtx_count = int(math.ceil(required_length / 3.0))
            self._rebuild_mesh(dummy_vtx_count)

        self.length = required_length
        if self.fn_mesh:
            ptr_addr = int(self.fn_mesh.getRawPoints())
            self.memory = CMemoryManager.from_ptr(ptr_addr, "f", (self.length,))

    def commit(self):
        if self._is_plug_mode and self.plug is not None and self.mObj_mesh is not None:
            self.plug.setMObject(self.mObj_mesh)

    def set_weights(self, vtx_count: int, influence_indices: tuple, raw_weights_1d):
        influence_count = len(influence_indices)
        header_size = 2 + influence_count
        total_size = header_size + (vtx_count * influence_count)

        self.resize(total_size)
        float_view = self.memory.view
        int_view = float_view.cast("b").cast("i")

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
        self.commit()

    def get_weights(self):
        if not self.is_valid or self.length == 0:
            return None, 0, 0, ()

        float_view = self.memory.view
        int_view = float_view.cast("b").cast("i")

        vtx_count = int_view[0]
        influence_count = int_view[1]
        header_size = 2 + influence_count

        influence_indices = tuple(int_view[2:header_size])
        pure_weights_view = float_view[header_size : self.length]

        return pure_weights_view, vtx_count, influence_count, influence_indices

    def set_sparse_weights(self, vtx_indices, bone_local_indices, sparse_weights_1d):
        if not self.is_valid or self.length == 0 or not vtx_indices or not bone_local_indices:
            return

        float_view = self.memory.view
        int_view = float_view.cast("b").cast("i")

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

    def fill_with_value(self, value: float):
        if not self.is_valid or self.length == 0:
            return
        full_dest_view = self.memory.view
        cWeightsCoreCython.fill_float_array(full_dest_view[: self.length], value)
        if self.max_capacity > self.length:
            cWeightsCoreCython.fill_float_array(full_dest_view[self.length :], -1.0)
        self.commit()


class WeightsLayerData:
    """纯粹的数据容器类，映射一个逻辑图层"""

    def __init__(
        self,
        index: int,
        enabled: bool,
        weightsHandle: WeightsHandle,
        maskHandle: WeightsHandle,
    ):
        self.index: int = index
        self.enabled: bool = enabled
        self.weightsHandle: WeightsHandle = weightsHandle
        self.maskHandle: WeightsHandle = maskHandle


# =========================================================================
# 2. 🚀 全新统一接口：WeightsManager
# =========================================================================


class WeightsManager:
    """
    高级权重管理器。
    将 Maya 的 Plug 属性和底层的 WeightsHandle 完全封装。
    外部业务只需通过图层 index 进行调配，无需接触底层内存装配细节。
    """

    BASE_LAYER_INDEX = -1

    def __init__(self, node: om1.MObject):
        self.mObj_node = node
        self.fn_node = om1.MFnDependencyNode(node)
        self.layers: dict[int, WeightsLayerData] = {}

        # 初始化时自动从节点同步当前状态
        self.sync_from_node()

    def sync_from_node(self):
        """解析节点当前的权重属性，并建立所有层级的映射"""
        self.layers.clear()

        # 1. 解析 Base Layer (基础权重)
        try:
            base_plug = self.fn_node.findPlug("cWeights", False)
            base_handle = WeightsHandle.from_plug(base_plug)
            self.layers[self.BASE_LAYER_INDEX] = WeightsLayerData(self.BASE_LAYER_INDEX, True, base_handle, None)
        except RuntimeError:
            pass  # 节点可能还未完全初始化，或者找不到该 Plug

        # 2. 解析 Compound Layers (附加权重层)
        try:
            layers_plug = self.fn_node.findPlug("cWeightsLayers", False)
            if not layers_plug.isNull() and layers_plug.isArray():
                for i in range(layers_plug.numElements()):
                    element_plug = layers_plug.elementByPhysicalIndex(i)
                    logical_idx = element_plug.logicalIndex()

                    # 依靠名称抓取子 Plug 更安全
                    enabled_plug = self._get_child_plug(element_plug, "cWeightsLayerEnabled")
                    weights_plug = self._get_child_plug(element_plug, "cWeightsLayer")
                    mask_plug = self._get_child_plug(element_plug, "cWeightsLayerMask")

                    enabled = enabled_plug.asBool() if enabled_plug else False
                    w_handle = WeightsHandle.from_plug(weights_plug) if weights_plug else None
                    m_handle = WeightsHandle.from_plug(mask_plug) if mask_plug else None

                    self.layers[logical_idx] = WeightsLayerData(logical_idx, enabled, w_handle, m_handle)
        except RuntimeError:
            pass

    def _get_child_plug(self, parent_plug: om1.MPlug, child_name: str) -> om1.MPlug:
        """安全提取子 Plug 的工具方法"""
        for i in range(parent_plug.numChildren()):
            child = parent_plug.child(i)
            # 对比短名称或长名称
            name = child.partialName(False, False, False, False, False, True)
            if name == child_name or child.name().endswith(child_name):
                return child
        return None

    # -------------------------------------------------------------------------
    # 图层管理操作
    # -------------------------------------------------------------------------

    def add_layer(self, layer_index: int, enabled: bool = True) -> WeightsLayerData:
        """在节点上创建或获取指定逻辑索引的图层，并建立 Handle 挂载"""
        if layer_index == self.BASE_LAYER_INDEX:
            return self.layers.get(self.BASE_LAYER_INDEX)

        layers_plug = self.fn_node.findPlug("cWeightsLayers", False)
        element_plug = layers_plug.elementByLogicalIndex(layer_index)

        enabled_plug = self._get_child_plug(element_plug, "cWeightsLayerEnabled")
        weights_plug = self._get_child_plug(element_plug, "cWeightsLayer")
        mask_plug = self._get_child_plug(element_plug, "cWeightsLayerMask")

        if enabled_plug:
            enabled_plug.setBool(enabled)

        w_handle = WeightsHandle.from_plug(weights_plug) if weights_plug else WeightsHandle()
        m_handle = WeightsHandle.from_plug(mask_plug) if mask_plug else WeightsHandle()

        layer = WeightsLayerData(layer_index, enabled, w_handle, m_handle)
        self.layers[layer_index] = layer
        return layer

    def remove_layer(self, layer_index: int):
        """移除指定图层 (借助 Maya 命令行或 Modifier 销毁数组元素)"""
        if layer_index == self.BASE_LAYER_INDEX:
            raise ValueError("基础权重层无法被移除！")

        if layer_index in self.layers:
            # 找到对应 Plug 并借助命令移除元素
            layers_plug = self.fn_node.findPlug("cWeightsLayers", False)
            element_plug = layers_plug.elementByLogicalIndex(layer_index)

            # 使用 cmds 删除该 plug 元素，断开连接并清空内存
            import maya.cmds as cmds

            cmds.removeMultiInstance(element_plug.name(), b=True)

            del self.layers[layer_index]

    def set_layer_enabled(self, layer_index: int, enabled: bool):
        """开关图层"""
        if layer_index in self.layers and layer_index != self.BASE_LAYER_INDEX:
            layer = self.layers[layer_index]
            layer.enabled = enabled
            layers_plug = self.fn_node.findPlug("cWeightsLayers", False)
            element_plug = layers_plug.elementByLogicalIndex(layer_index)
            enabled_plug = self._get_child_plug(element_plug, "cWeightsLayerEnabled")
            if enabled_plug:
                enabled_plug.setBool(enabled)

    # -------------------------------------------------------------------------
    # 核心读写操作
    # -------------------------------------------------------------------------

    def set_weights(self, layer_index: int, vtx_count: int, influence_indices: tuple, raw_weights_1d):
        """设置图层全量权重"""
        layer = self._get_safe_layer(layer_index)
        if layer.weightsHandle:
            layer.weightsHandle.set_weights(vtx_count, influence_indices, raw_weights_1d)

    def get_weights(self, layer_index: int):
        """获取纯净图层权重数据与元数据"""
        layer = self._get_safe_layer(layer_index)
        if layer.weightsHandle:
            return layer.weightsHandle.get_weights()
        return None, 0, 0, ()

    def set_sparse_weights(self, layer_index: int, vtx_indices, bone_local_indices, sparse_weights_1d):
        """双重稀疏覆写 (多用于 Undo/Redo 极速覆写)"""
        layer = self._get_safe_layer(layer_index)
        if layer.weightsHandle:
            layer.weightsHandle.set_sparse_weights(vtx_indices, bone_local_indices, sparse_weights_1d)

    def set_mask(self, layer_index: int, vtx_count: int, raw_weights_1d):
        """单独设置图层的遮罩 (Mask 数据也是利用 WeightsHandle，默认骨骼数为 1)"""
        layer = self._get_safe_layer(layer_index)
        if layer.maskHandle:
            # Mask 本质也是一种权重，但是没有骨骼，我们默认传骨骼 ID 为 0 的单通道数据
            layer.maskHandle.set_weights(vtx_count, (0,), raw_weights_1d)

    def get_mask(self, layer_index: int):
        """获取遮罩数据"""
        layer = self._get_safe_layer(layer_index)
        if layer.maskHandle:
            weights, v_count, inf_count, _ = layer.maskHandle.get_weights()
            return weights, v_count
        return None, 0

    def set_sparse_mask(self, layer_index: int, vtx_indices, sparse_weights_1d):
        """稀疏覆写遮罩数据"""
        layer = self._get_safe_layer(layer_index)
        if layer.maskHandle:
            layer.maskHandle.set_sparse_weights(vtx_indices, (0,), sparse_weights_1d)

    def _get_safe_layer(self, layer_index: int) -> WeightsLayerData:
        """内部方法：安全获取图层，如果不存在抛出异常"""
        if layer_index not in self.layers:
            raise ValueError(f"图层 {layer_index} 不存在。请先调用 add_layer 创建它。")
        return self.layers[layer_index]
