import math
import maya.OpenMaya as om1  # type: ignore

from z_np.src.cMemoryView import CMemoryManager
from z_np.src import cWeightsCoreCython


class WeightsHandle:
    """
    memory_array = [vtx_count] + [inf_count] + [influence_indices...] + [weights_array...]

    权重数据装配器
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

        # 这里的 length 代表的是 Header + Payload 的总浮点数长度
        self.length = 0

        # 唯一对外暴露的底层内存管家（包含 Header 和 Payload）
        self.memory: CMemoryManager = None

    # =========================================================================
    # 1. 工厂方法 (保持不变)
    # =========================================================================
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

    @classmethod
    def from_mesh(cls, fn_mesh):
        instance = cls()
        instance._is_plug_mode = False

        if isinstance(fn_mesh, om1.MFnMesh):
            instance.fn_mesh = fn_mesh
            instance.mObj_mesh = fn_mesh.object()
        else:
            instance.mObj_mesh = fn_mesh

        instance._setup_mesh()
        return instance

    @classmethod
    def from_attr_string(cls, attr_path: str):
        sel = om1.MSelectionList()
        try:
            sel.add(attr_path)
        except RuntimeError:
            raise ValueError(f"找不到指定的属性路径: {attr_path}")

        plug = om1.MPlug()
        sel.getPlug(0, plug)
        return cls.from_plug(plug)

    # =========================================================================
    # 2. 内部装配逻辑 (💥 全新：零拷贝读取 Header 解析真实长度)
    # =========================================================================
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

        # 💥 上帝视角：用 int32 强转读取 Header
        int_view = float_view.cast("b").cast("i")

        vtx_count = int_view[0]
        influence_count = int_view[1]

        # 安全校验：如果是全新创建的垃圾内存，数据会极度异常
        if vtx_count <= 0 or influence_count <= 0 or influence_count > 20000:
            self.length = 0
            self.memory = CMemoryManager.from_ptr(ptr_addr, "f", (0,))
            return

        # 计算真实的总长度 = Header + Payload
        header_size = 2 + influence_count
        payload_size = vtx_count * influence_count
        real_length = header_size + payload_size

        # 防止内存越界崩溃
        if real_length > self.max_capacity:
            self.length = 0
            self.memory = CMemoryManager.from_ptr(ptr_addr, "f", (0,))
            return

        self.length = real_length
        self.memory = CMemoryManager.from_ptr(ptr_addr, "f", (self.length,))

    @property
    def is_valid(self):
        return (self.fn_mesh is not None) and (self.max_capacity > 0) and (self.memory is not None)

    # =========================================================================
    # 3. 容器生命周期管理 (不再有 padding 相关的废逻辑)
    # =========================================================================
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

    # =========================================================================
    # 4. 🚀 核心读写接口 (混合内存装配与拆解)
    # =========================================================================
    def set_weights(self, vtx_count: int, influence_indices: tuple, raw_weights_1d):
        """
        设置图层全量数据：将 Header (元数据) 和 Weights (权重矩阵) 极速写入底层内存
        """
        influence_count = len(influence_indices)
        header_size = 2 + influence_count
        payload_size = vtx_count * influence_count
        total_size = header_size + payload_size

        self.resize(total_size)

        float_view = self.memory.view
        int_view = float_view.cast("b").cast("i")

        # 1. 写入 Header (Type Punning)
        int_view[0] = vtx_count
        int_view[1] = influence_count
        for i, bone_id in enumerate(influence_indices):
            int_view[2 + i] = bone_id

        # 2. 写入 Weights
        if isinstance(raw_weights_1d, memoryview):
            src_view = raw_weights_1d.cast("B").cast("f")
        else:
            temp_mgr = CMemoryManager.from_list(list(raw_weights_1d), "f")
            src_view = temp_mgr.view

        float_view[header_size:total_size] = src_view

        self.commit()

    def get_weights(self):
        """
        获取纯净权重与元数据：供 Cython 引擎和 Deformer 合并使用
        返回: (pure_weights_view, vtx_count, influence_count, influence_indices)
        """
        if not self.is_valid or self.length == 0:
            return None, 0, 0, ()

        float_view = self.memory.view
        int_view = float_view.cast("b").cast("i")

        vtx_count = int_view[0]
        influence_count = int_view[1]
        header_size = 2 + influence_count

        # 提取局部骨骼映射表
        influence_indices = tuple(int_view[2:header_size])

        # 💥 核心魔法：切片出纯纯的权重数据
        pure_weights_view = float_view[header_size : self.length]

        return pure_weights_view, vtx_count, influence_count, influence_indices

    def set_sparse_weights(self, vtx_indices, bone_local_indices, sparse_weights_1d):
        """
        双重稀疏写入接口 (极致内存压缩版)
        与笔刷引擎 end_stroke 返回的压缩数据完美对接！
        专供 Undo/Redo 极速覆写物理内存使用。

        Args:
            vtx_indices: array.array('i') 或 list，修改的顶点 ID
            bone_local_indices: array.array('i') 或 list，实际变动的骨骼局部列 ID
            sparse_weights_1d: array.array('f') 或 memoryview，1D 极限压缩权重快照
        """
        if not self.is_valid or self.length == 0 or not vtx_indices or not bone_local_indices:
            return

        float_view = self.memory.view
        int_view = float_view.cast("b").cast("i")

        # 1. 瞬间从 Header 读取骨骼数量，算出 Payload 的安全起始线
        influence_count = int_view[1]
        header_size = 2 + influence_count

        # 2. 统一将输入转为 memoryview，确保底层的切片赋值速度
        if isinstance(sparse_weights_1d, memoryview):
            # 消除可能存在的格式限制
            src_view = sparse_weights_1d.cast("B").cast("f")
        else:
            # array.array('f') 可以直接套用 memoryview 并强转为 float 视图，零拷贝！
            src_view = memoryview(sparse_weights_1d).cast("B").cast("f")

        num_modified_bones = len(bone_local_indices)

        # 3. 💥 核心插秧逻辑：遍历修改的顶点和列，精准还原到物理内存
        # 因为修改的点和列数量通常极少，这个纯 Python 循环在 1 毫秒内就能跑完
        for i, vtx_id in enumerate(vtx_indices):
            # 这个顶点在底层一维物理内存中的绝对起点（跨过 Header 区）
            row_start = header_size + vtx_id * influence_count

            for j, local_col_id in enumerate(bone_local_indices):
                # 目标地址：跳过 Header -> 找到对应顶点行 -> 偏移到对应的骨骼列
                dest_idx = row_start + local_col_id

                # 源地址：在 1D 压缩数组里的顺序位置 (十字交叉展开)
                src_idx = i * num_modified_bones + j

                # 纯粹的 C 级别内存覆写
                float_view[dest_idx] = src_view[src_idx]

        # 4. 触发 Maya 的脏标记刷新
        self.commit()

    
    # =========================================================================
    # 5. 🌐 Maya 原生 API 对接接口 (MDoubleArray 转换)
    # =========================================================================
    def set_from_maya_array(self, vtx_count: int, influence_indices: tuple, maya_double_array: om1.MDoubleArray):
        """
        [入口] 将 Maya 提取的 MDoubleArray 存入我们的自定义混合内存中。
        通常用于：第一次读取蒙皮权重时 (MFnSkinCluster.getWeights)
        """
        length = maya_double_array.length()
        
        # 1. 将 Maya 的 MDoubleArray 提取为标准 Python 列表
        # (在 OpenMaya 1.0 中，如果直接 list() 报错，则使用列表推导式提取)
        try:
            py_list = list(maya_double_array)
        except TypeError:
            py_list = [maya_double_array[i] for i in range(length)]
            

        self.set_weights(vtx_count, influence_indices, py_list)

    def get_as_maya_array(self) -> om1.MDoubleArray:
        """
        [出口] 将我们底层内存里的纯权重，打包回 Maya 的 MDoubleArray。
        通常用于：图层合并完毕后，写回给 Maya (MFnSkinCluster.setWeights)
        """
        maya_array = om1.MDoubleArray()
        
        # 1. 获取纯净的权重视图（自动去掉了 Header）
        pure_weights, v_count, inf_count, _ = self.get_weights()
        
        if pure_weights is None or v_count == 0:
            return maya_array
            
        payload_size = v_count * inf_count
        maya_array.setLength(payload_size)
        
        # 2. 瞬间把 C 级 memoryview 拍平成纯 Python 列表 (极速零拷贝提取！)
        # 因为 pure_weights 是 memoryview('f')，tolist() 返回的是纯 Python float 列表
        py_list = pure_weights.tolist()
        
        # 3. 将数据注入 Maya 原生结构
        # (注意：OpenMaya 1.0 没有批量设置接口，只能使用 for 循环 set。
        # 但因为提取 py_list 已经是极速，这个 set 循环在 C++ 底层依然很快)
        for i in range(payload_size):
            maya_array.set(py_list[i], i)
            
        return maya_array

    def fill_with_value(self, value: float):
        if not self.is_valid or self.length == 0:
            return

        full_dest_view = self.memory.view

        # 核心数据区填充
        cWeightsCoreCython.fill_float_array(full_dest_view[: self.length], value)

        # 尾部无效区填充
        if self.max_capacity > self.length:
            cWeightsCoreCython.fill_float_array(full_dest_view[self.length :], -1.0)

        self.commit()


class WeightsLayerData:
    # ... 保持不变 ...
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
