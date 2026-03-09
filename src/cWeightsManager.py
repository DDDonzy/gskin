import math
import typing
from enum import IntEnum

import maya.OpenMaya as om1  # type: ignore

from .cMemoryView import CMemoryManager
from . import cWeightsCoreCython


# =========================================================================
# 0. 枚举与常量定义
# =========================================================================


class LayerChildPlug(IntEnum):
    """
    定义图层复合属性 (cWeightsLayers) 下的子属性索引。
    严格对应 nodeInitializer 中 addChild 的顺序，彻底消灭魔法数字 (Magic Numbers)。
    """

    ENABLED = 0  # cWeightsLayerEnabled
    WEIGHTS = 1  # cWeightsLayer
    MASK = 2  # cWeightsLayerMask


# =========================================================================
# 1. 底层内存管家 (WeightsHandle) - 极致的 C 内存读写与显式数据流
# =========================================================================


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
        self.length = 0
        self.memory: CMemoryManager = None

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

    def _rebuild_mesh(self, vtx_count: int):
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
        """自动扩容内存空间，显式返回全新或保留的属性配置"""
        new_mObj_mesh = self.mObj_mesh
        new_fn_mesh = self.fn_mesh
        new_max_capacity = self.max_capacity

        if required_length > self.max_capacity:
            dummy_vtx_count = int(math.ceil(required_length / 3.0))
            # 显式接收重新构建的 Mesh 数据
            (
                new_mObj_mesh,
                new_fn_mesh,
                new_max_capacity,
            ) = self._rebuild_mesh(dummy_vtx_count)

            # (允许的 Maya API 副作用：更新 Plug 连接)
            if self._is_plug_mode and self.plug is not None:
                self.plug.setMObject(new_mObj_mesh)
            elif self.data_handle is not None:
                self.data_handle.setMObject(new_mObj_mesh)

        new_length = required_length
        new_memory = self.memory

        if new_fn_mesh:
            ptr_addr = int(new_fn_mesh.getRawPoints())
            new_memory = CMemoryManager.from_ptr(ptr_addr, "f", (new_length,))

        return new_mObj_mesh, new_fn_mesh, new_max_capacity, new_length, new_memory

    def commit(self):
        """强制提交到 Maya，仅作为外部状态刷新接口"""
        if self._is_plug_mode and self.plug is not None and self.mObj_mesh is not None:
            self.plug.setMObject(self.mObj_mesh)

    # -------------------------------------------------------------------------
    # 核心读写接口 (全量 / 稀疏)
    # -------------------------------------------------------------------------

    @property
    def is_valid(self) -> bool:
        """检查当前 Handle 是否持有有效的底层内存"""
        return self.memory is not None

    def set_weights(self, vtx_count: int, influence_indices: tuple, raw_weights_1d):
        """[全量写入] 传入顶点总数、骨骼映射表、完整的一维权重数组"""
        influence_count = len(influence_indices)
        header_size = 2 + influence_count
        total_size = header_size + (vtx_count * influence_count)

        # 💥 显式装配：通过 resize 返回更新所有核心状态
        self.mObj_mesh, self.fn_mesh, self.max_capacity, self.length, self.memory = self.resize(total_size)

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


class WeightsLayerItem:
    """
    [图层逻辑单元]
    在 update_data 时被实例化，全面缓存 Maya Plug 与底层内存句柄。
    将绘制高频期间的 Maya API 查询开销降至绝对的 0。
    """

    def __init__(self, layer_plug: om1.MPlug):
        # 1. 缓存主属性与索引
        self.layer_plug = layer_plug
        self.logical_index = layer_plug.logicalIndex()

        # 2. 提前提取并缓存所有的子 Plug (避免每次调用 child() 和字符串索引)
        self.enabled_plug = layer_plug.child(LayerChildPlug.ENABLED)
        self.weights_plug = layer_plug.child(LayerChildPlug.WEIGHTS)
        self.mask_plug = layer_plug.child(LayerChildPlug.MASK)

        # 3. 💥 提前装配并缓存底层的内存句柄 (WeightsHandle)
        self.weights_handle = WeightsHandle.from_plug(self.weights_plug)
        self.mask_handle = WeightsHandle.from_plug(self.mask_plug)

    @property
    def is_enabled(self) -> bool:
        return self.enabled_plug.asBool()

    @is_enabled.setter
    def is_enabled(self, val: bool):
        self.enabled_plug.setBool(val)

    def get_weights_handle(self) -> WeightsHandle:
        return self.weights_handle

    def get_mask_handle(self) -> WeightsHandle:
        return self.mask_handle

    def delete(self, modifier: om1.MDGModifier = None):
        do_it_internally = False
        if modifier is None:
            modifier = om1.MDGModifier()
            do_it_internally = True

        modifier.removeMultiInstance(self.layer_plug, True)

        if do_it_internally:
            modifier.doIt()


class WeightsManager:
    def __init__(self, node: om1.MObject):
        self.mObj_node = node
        self.fn_node = om1.MFnDependencyNode(node)

        self.out_plug: om1.MPlug = None
        self.out_handle: WeightsHandle = None
        self.refresh_plug: om1.MPlug = None
        self.layers_plug: om1.MPlug = None

        self.layers: list[WeightsLayerItem] = []

        self.update_data()

    @classmethod
    def from_string(cls, node_name: str) -> "WeightsManager":
        """通过节点名称字符串快速实例化 WeightsManager。

        Args:
            node_name (str): Maya 场景中的节点名称或 DAG 路径。

        Returns:
            WeightsManager: 绑定了该节点的管理实例。

        Raises:
            ValueError: 当传入的节点名称在场景中不存在时抛出。
        """
        sel = om1.MSelectionList()
        try:
            sel.add(node_name)
        except RuntimeError:
            raise ValueError(f"无法找到指定的 Maya 节点: '{node_name}'")

        mObj = om1.MObject()
        sel.getDependNode(0, mObj)

        return cls(mObj)

    def update_data(self):
        """
        [状态同步器]
        一次性扫描 Maya 节点，刷新所有 Plug 缓存与底层内存池。
        当你在 UI 层面添加、删除了图层，或改变了节点连接后，手动调用此函数。
        """
        # 1. 缓存输出目标
        try:
            self.out_plug = self.fn_node.findPlug("cWeights", False)
            self.out_handle = WeightsHandle.from_plug(self.out_plug)
        except RuntimeError:
            raise RuntimeError(f"节点 {self.fn_node.name()} 缺失核心输出属性 'cWeights'！")

        # 2. 缓存刷新插头
        try:
            self.refresh_plug = self.fn_node.findPlug("cRefresh", False)
        except RuntimeError:
            self.refresh_plug = None

        # 3. 缓存整个图层数组架构
        try:
            self.layers_plug = self.fn_node.findPlug("cWeightsLayers", False)
        except RuntimeError:
            self.layers_plug = None

        self.layers.clear()
        if self.layers_plug is not None and not self.layers_plug.isNull() and self.layers_plug.isArray():
            for i in range(self.layers_plug.numElements()):
                element_plug = self.layers_plug.elementByPhysicalIndex(i)
                self.layers.append(WeightsLayerItem(element_plug))

    def get_layer_indices(self) -> list[int]:
        """现在仅仅是读取缓存列表的推导式，极其快速"""
        return [item.logical_index for item in self.layers]

    def get_layer(self, layer_index: int) -> WeightsLayerItem:
        """从缓存池中极速定位图层"""
        for item in self.layers:
            if item.logical_index == layer_index:
                return item
        return None

    def add_layer(self, enabled: bool = True) -> "WeightsLayerItem":
        """
        向 Maya 申请新图层（自动寻找空闲索引并追加到末尾），并强制同步系统缓存。
        """
        next_index = 0

        if self.layers_plug is not None:
            existing_indices = self.get_layer_indices()

            if existing_indices:
                next_index = max(existing_indices) + 1

            self.layers_plug.elementByLogicalIndex(next_index)
            new_element_plug = self.layers_plug.elementByLogicalIndex(next_index)
            enabled_child_plug = new_element_plug.child(LayerChildPlug.ENABLED)
            enabled_child_plug.setBool(enabled)

        self.update_data()

        item = self.get_layer(next_index)
        if item:
            item.is_enabled = enabled

        return item

    def remove_layer(self, layer_index: int, modifier: om1.MDGModifier = None):
        item = self.get_layer(layer_index)
        if item:
            item.delete(modifier)
            # 💥 结构发生变化，刷新缓存并重算
            self.update_data()
            self.bake_to_final_weights(vtx_indices=None)

    def set_layer_enabled(self, layer_index: int, enabled: bool):
        item = self.get_layer(layer_index)
        if item:
            item.is_enabled = enabled
            self.bake_to_final_weights(vtx_indices=None)

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

        active_layers: list["WeightsLayerItem"] = []
        combined_bone_set: set[int] = set()
        vtx_count: int = 0

        # 1. 扫描内存池 (不再有 findPlug，全速 O(N) 遍历)
        for item in self.layers:
            if item.is_enabled:
                w_handle = item.get_weights_handle()
                if w_handle and w_handle.is_valid:
                    active_layers.append(item)

                    _, v_cnt, _, inf_indices = w_handle.get_weights()
                    if v_cnt > 0:
                        vtx_count = v_cnt
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
            self.out_handle.mObj_mesh,
            self.out_handle.fn_mesh,
            self.out_handle.max_capacity,
            self.out_handle.length,
            self.out_handle.memory,
        ) = self.out_handle.resize(required_length)

        int_view = self.out_handle.memory.view.cast("B").cast("i")
        int_view[0] = vtx_count
        int_view[1] = out_inf_count
        for i, bone_id in enumerate(out_influence_indices):
            int_view[2 + i] = bone_id

        # 极速清理幽灵数据
        if weights_size > 0:
            header_byte_offset = header_size * 4
            payload_ptr = self.out_handle.memory.ptr + header_byte_offset
            CMemoryManager.from_ptr(payload_ptr, "f", (weights_size,)).fill(0.0)

        output_view = self.out_handle.memory.view

        # 3. 准备稀疏顶点视图
        vtx_indices_view = None
        if vtx_indices is not None and len(vtx_indices) > 0:
            vtx_indices_view = CMemoryManager.from_list(list(vtx_indices), "i").view

        # 4. Cython 完美接棒
        for item in active_layers:
            layer_view = item.get_weights_handle().memory.view

            m_handle = item.get_mask_handle()
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

    def trigger_evaluation(self):
        """完全规避了 findPlug，如果存在 cRefresh 插头，直接发送脏标记"""
        if self.refresh_plug:
            self.refresh_plug.setInt(self.refresh_plug.asInt() + 1)
