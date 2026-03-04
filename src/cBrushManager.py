import typing
from dataclasses import dataclass

# 统一使用相对路径和模块导入
from . import cMemoryView
from . import cBrushCoreCython  # 🟢 假设你的 Cython 文件现在叫 cBrushCore

if typing.TYPE_CHECKING:
    from . import cDisplayNode


# ==========================================#
# 📦 数据结构定义
# ==========================================#
@dataclass
class BrushSettings:
    """存放笔刷半径、强度、模式等配置。"""

    radius: float = 0.5
    strength: float = 0.1
    falloff_type: int = 1  # 默认为 Airbrush (1)
    mode: int = 0  # 0:Add, 1:Sub, 2:Replace, 3:Multiply
    use_surface: bool = True


class BrushTopology:
    """存放从网格解析出的图论邻接表。"""

    __slots__ = ("adjacency_offsets", "adjacency_indices")

    def __init__(self):
        self.adjacency_offsets: cMemoryView.CMemoryManager = None
        self.adjacency_indices: cMemoryView.CMemoryManager = None


class BrushGraphPools:
    """管理算法所需的世代池（Epochs）。"""

    __slots__ = ("vertices_epochs", "brush_epoch")

    def __init__(self):
        self.vertices_epochs: cMemoryView.CMemoryManager = None
        self.brush_epoch: int = 0


class BrushHitState:
    """存放单次命中的结果缓冲区。"""

    __slots__ = ("hit_count", "hit_indices_mgr", "hit_weights_mgr", "hit_center_position")

    def __init__(self, vertex_count: int):
        self.hit_count: int = 0
        self.hit_indices_mgr: "cMemoryView.CMemoryManager" = cMemoryView.CMemoryManager.allocate("i", (vertex_count,))
        self.hit_weights_mgr: "cMemoryView.CMemoryManager" = cMemoryView.CMemoryManager.allocate("f", (vertex_count,))
        self.hit_center_position: tuple = (0.0, 0.0, 0.0)

    def clear(self):
        self.hit_count = 0
        self.hit_center_position = (0.0, 0.0, 0.0)


# ==========================================#
# 🧠 笔刷管理器 (原 WeightBrushCore)
# ==========================================#
class WeightBrushManager:
    def __init__(self, preview_shape: "cDisplayNode.WeightPreviewShape"):
        self.preview_shape = preview_shape
        self.settings = BrushSettings()

        self.topology: BrushTopology = None
        self.pools: BrushGraphPools = None
        self.hit_state: BrushHitState = None

        # 1. 核心步骤：直接从渲染节点的 Mesh 上下文构建邻接表，不经过 InDeformMesh
        self._build_topology_from_cache()

        # 2. 内存链接：将本地 Buffer 指针共享给渲染节点，实现渲染直读
        self.preview_shape.brush_context.brush_hit_indices = self.hit_state.hit_indices_mgr
        self.preview_shape.brush_context.brush_hit_weights = self.hit_state.hit_weights_mgr

        # 3. 初始化骨骼锁 (如果存在蒙皮实例)
        cskin = self.preview_shape.cSkin
        if cskin:
            influences_count = cskin.influences_count
            if not cskin.influences_locks_mgr or len(cskin.influences_locks_mgr.view) != influences_count:
                cskin.influences_locks_mgr = cMemoryView.CMemoryManager.allocate("B", (influences_count,))
                cskin.influences_locks_mgr.view[:] = 0

    def _build_topology_from_cache(self):
        """
        🟢 核心改造：利用渲染节点的 edge_indices 徒手构建邻接表。
        """
        mesh_ctx = self.preview_shape.mesh_context
        vtx_count = mesh_ctx.vertex_count
        edge_view = mesh_ctx.edge_indices.view

        # 构建双向邻接集合
        adj_list = [set() for _ in range(vtx_count)]
        for i in range(len(edge_view) // 2):
            v1 = edge_view[i * 2]
            v2 = edge_view[i * 2 + 1]
            adj_list[v1].add(v2)
            adj_list[v2].add(v1)

        # 转换为 CSR (Compressed Sparse Row) 格式
        offsets = [0] * (vtx_count + 1)
        indices = []
        for i in range(vtx_count):
            offsets[i] = len(indices)
            indices.extend(adj_list[i])
        offsets[vtx_count] = len(indices)

        self.topology = BrushTopology()
        self.topology.adjacency_offsets = cMemoryView.CMemoryManager.from_list(offsets, "i")
        self.topology.adjacency_indices = cMemoryView.CMemoryManager.from_list(indices, "i")

        # 初始化纪元池
        self.pools = BrushGraphPools()
        self.pools.vertices_epochs = cMemoryView.CMemoryManager.allocate("i", (vtx_count,))
        self.pools.vertices_epochs.view[:] = 0

        # 初始化结果缓冲区
        self.hit_state = BrushHitState(vtx_count)

    def teardown(self):
        """清理资源。"""
        if self.preview_shape and self.preview_shape.brush_context:
            self.preview_shape.brush_context.brush_hit_count = 0

    def clear_hit_state(self):
        """清空命中状态（当笔刷离开模型时）。"""
        if self.hit_state:
            self.hit_state.clear()
            self.preview_shape.brush_context.brush_hit_count = 0

    def detect_range(self, center_xyz: tuple, hit_tri: int) -> int:
        """
        调用 Cython Core 进行范围检测。
        """
        mesh_ctx = self.preview_shape.mesh_context
        if not mesh_ctx or not mesh_ctx.vertex_positions:
            return 0

        # 世代同步
        self.pools.brush_epoch += 1

        # 🟢 调用你最新的 calc_brush_weights 接口
        hit_count = cBrushCoreCython.calc_brush_weights(
            mesh_ctx.vertex_positions.view,
            center_xyz,
            hit_tri,
            mesh_ctx.triangle_indices.view,
            self.topology.adjacency_offsets.view,
            self.topology.adjacency_indices.view,
            self.settings.radius,
            self.settings.falloff_type,
            self.settings.use_surface,
            self.pools.brush_epoch,
            self.pools.vertices_epochs.view,
            self.hit_state.hit_indices_mgr.view,
            self.hit_state.hit_weights_mgr.view,
        )

        self.hit_state.hit_count = hit_count
        self.hit_state.hit_center_position = center_xyz

        # 🟢 通知渲染节点有 X 个点需要画出 Brush Color
        self.preview_shape.brush_context.brush_hit_count = hit_count

        return hit_count

    def apply_weight_math(self) -> bool:
        """
        调用 Cython Core 应用权重修改。
        """
        if self.hit_state.hit_count == 0:
            return False

        cskin = self.preview_shape.cSkin
        render_ctx = self.preview_shape.render_context

        if not cskin or render_ctx.paintLayerIndex not in cskin.weightsLayer:
            return False

        layer = cskin.weightsLayer[render_ctx.paintLayerIndex]

        # 根据当前模式提取 2D 权重内存视图
        if render_ctx.paintMask:
            if not layer.maskHandle:
                return False
            modify_view = layer.maskHandle.memory.view
            target_idx = 0
            locks_view = cMemoryView.CMemoryManager.allocate("B", (1,)).view  # 掩码模式无锁
        else:
            if not layer.weightsHandle:
                return False
            modify_view = layer.weightsHandle.memory.view
            target_idx = render_ctx.paintInfluenceIndex
            locks_view = cskin.influences_locks_mgr.view

        # 🟢 调用你最新的接口：完成计算并自动触发归一化
        cBrushCoreCython.skin_weight_brush(self.settings.strength, self.settings.mode, target_idx, locks_view, self.hit_state.hit_indices_mgr.view, self.hit_state.hit_weights_mgr.view, self.hit_state.hit_count, modify_view)
        return True
