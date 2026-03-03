
import typing
from dataclasses import dataclass

# 统一使用相对路径和模块导入
from . import cMemoryView
from . import cBrushCython

if typing.TYPE_CHECKING:
    from . import cDisplayNode
    import maya.api.OpenMaya as om


# ==========================================#
# 📦 数据结构定义
# ==========================================#
@dataclass
class BrushSettings:
    """存放笔刷半径、强度、模式等核心配置。"""
    radius: float = 0.5
    strength: float = 0.1
    falloff_type: int = 0
    mode: int = 0
    use_surface: bool = True


class BrushTopology:
    """存放笔刷算法所需的静态拓扑数据（如图的邻接表）。"""
    __slots__ = ("adjacency_offsets", "adjacency_indices")

    def __init__(self):
        self.adjacency_offsets: cMemoryView.CMemoryManager = None
        self.adjacency_indices: cMemoryView.CMemoryManager = None


class BrushGraphPools:
    """为 Cython 笔刷遍历算法持有可复用的临时内存缓冲区。"""
    __slots__ = ("pool_node_epochs", "pool_dist", "pool_queue", "pool_in_queue")

    def __init__(self):
        self.pool_node_epochs: cMemoryView.CMemoryManager = None
        self.pool_dist: cMemoryView.CMemoryManager = None
        self.pool_queue: cMemoryView.CMemoryManager = None
        self.pool_in_queue: cMemoryView.CMemoryManager = None


class BrushHitState:
    """存放单次笔刷命中检测的结果。"""
    __slots__ = ("hit_count", "hit_indices_mgr", "hit_weights_mgr", "hit_center_position", "hit_center_normal")

    def __init__(self, vertex_count: int):
        self.hit_count: int = 0
        self.hit_indices_mgr: "cMemoryView.CMemoryManager" = cMemoryView.CMemoryManager.allocate("i", (vertex_count,))
        self.hit_weights_mgr: "cMemoryView.CMemoryManager" = cMemoryView.CMemoryManager.allocate("f", (vertex_count,))
        self.hit_center_position: tuple = (0.0, 0.0, 0.0)
        self.hit_center_normal: tuple = (0.0, 1.0, 0.0)

    def clear(self):
        self.hit_count = 0
        self.hit_center_position = (0.0, 0.0, 0.0)
        self.hit_center_normal = (0.0, 1.0, 0.0)


# ==========================================#
# 🧠 核心笔刷逻辑控制器
# ==========================================#
class WeightBrushCore:
    def __init__(self, preview_shape: "cDisplayNode.WeightPreviewShape", mfn_mesh: "om.MFnMesh"):
        self.preview_shape = preview_shape
        self.settings = BrushSettings()  # 核心逻辑拥有自己的配置

        self.topology: BrushTopology = None
        self.pools: BrushGraphPools = None
        self.hit_state: BrushHitState = None

        # 基于网格构建算法所需的数据结构
        self._build_topology_and_pools(mfn_mesh)

        # 将命中状态的内存管理器链接到显示上下文，以便渲染器可以访问它们
        self.preview_shape.brush_context.brush_hit_indices = self.hit_state.hit_indices_mgr
        self.preview_shape.brush_context.brush_hit_weights = self.hit_state.hit_weights_mgr

        # 如果需要，初始化蒙皮节点上的影响锁
        cskin = self.preview_shape.cSkin
        if cskin:
            influences_count = cskin.influences_count
            if not cskin.influences_locks_mgr or len(cskin.influences_locks_mgr.view) != influences_count:
                cskin.influences_locks_mgr = cMemoryView.CMemoryManager.allocate("B", (influences_count,))
                cskin.influences_locks_mgr.view[:] = 0

    def _build_topology_and_pools(self, mfn_mesh: "om.MFnMesh"):
        """
        基于给定的 Maya 网格功能集，生成邻接数据并分配算法所需的内存池。
        这个方法只应在网格拓扑发生改变时调用。
        """
        vertex_count = mfn_mesh.numVertices
        
        # 1. 构建邻接表
        self.topology = BrushTopology()
        adj_offsets = [0] * (vertex_count + 1)
        adj_indices = []
        for i in range(vertex_count):
            connected_verts = mfn_mesh.getConnectedVertices(i)
            adj_offsets[i] = len(adj_indices)
            adj_indices.extend(connected_verts)
        adj_offsets[vertex_count] = len(adj_indices)
        
        self.topology.adjacency_offsets = cMemoryView.CMemoryManager.from_list(adj_offsets, "i")
        self.topology.adjacency_indices = cMemoryView.CMemoryManager.from_list(adj_indices, "i")

        # 2. 分配算法内存池
        self.pools = BrushGraphPools()
        self.pools.pool_node_epochs = cMemoryView.CMemoryManager.allocate("i", (vertex_count,))
        self.pools.pool_dist = cMemoryView.CMemoryManager.allocate("f", (vertex_count,))
        self.pools.pool_queue = cMemoryView.CMemoryManager.allocate("i", (vertex_count,))
        self.pools.pool_in_queue = cMemoryView.CMemoryManager.allocate("B", (vertex_count,))

        # 3. 分配命中状态容器
        self.hit_state = BrushHitState(vertex_count)

    def teardown(self):
        """清理资源，在工具关闭时调用。"""
        if self.preview_shape and self.preview_shape.brush_context:
            self.preview_shape.brush_context.brush_hit_count = 0
        # 其他所有对象 (hit_state, settings, topology, pools) 将被 Python 的垃圾回收自动处理

    def clear_hit_state(self):
        """清空当前的命中结果，在笔刷移出模型时调用。"""
        if self.hit_state:
            self.hit_state.clear()
            self.preview_shape.brush_context.brush_hit_count = 0

    def detect_range(self, center_xyz: tuple, hit_tri: int) -> int:
        """
        从射线命中的三角形开始，在模型表面或空间中检测笔刷范围内的所有顶点，
        并计算它们的权重。
        """
        render_mesh = self.preview_shape.mesh_context
        brush_ctx = self.preview_shape.brush_context
        
        if not render_mesh or not render_mesh.vertex_positions_2d_view:
            return 0

        brush_ctx.brush_epoch += 1

        hit_count = cBrushCython.compute_brush_weights_god_mode(
            center_xyz,
            render_mesh.vertex_positions_2d_view.view,
            render_mesh.triangle_indices.view,
            hit_tri,
            self.topology.adjacency_offsets.view,
            self.topology.adjacency_indices.view,
            self.settings.radius,
            self.settings.falloff_type,
            self.settings.use_surface,
            brush_ctx.brush_epoch,
            self.pools.pool_node_epochs.view,
            self.pools.pool_dist.view,
            self.pools.pool_queue.view,
            self.pools.pool_in_queue.view,
            self.hit_state.hit_indices_mgr.view,
            self.hit_state.hit_weights_mgr.view,
        )

        self.hit_state.hit_count = hit_count
        self.hit_state.hit_center_position = center_xyz
        
        # 更新显示上下文中的命中数量，触发渲染更新
        brush_ctx.brush_hit_count = hit_count
        
        return hit_count

    def apply_weight_math(self) -> bool:
        """应用笔刷的权重算法（增加、替换、平滑等）。"""
        if self.hit_state.hit_count == 0:
            return False

        cskin = self.preview_shape.cSkin
        # 获取当前正在绘制的权重或遮罩的内存视图
        modify_weights2D, target_inf, is_mask = self.preview_shape.active_paint_weights
        
        if modify_weights2D is None or cskin is None:
            return False

        # 如果是遮罩模式，锁视图是虚拟的；否则使用真实的骨骼锁视图
        locks_view = cMemoryView.CMemoryManager.allocate("B", (1,)).view if is_mask else cskin.influences_locks_mgr.view

        cBrushCython.skin_weight_brush(
            self.settings.strength,
            self.settings.mode,
            target_inf,
            locks_view,
            self.hit_state.hit_indices_mgr.view,
            self.hit_state.hit_weights_mgr.view,
            self.hit_state.hit_count,
            modify_weights2D.view,
        )
        return True
