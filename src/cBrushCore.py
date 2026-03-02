import typing
from dataclasses import dataclass

# 统一使用相对路径和模块导入
from . import cMemoryView
from . import cBrushCython

if typing.TYPE_CHECKING:
    from . import cDisplayNode

# ==========================================
# 📦 数据结构定义
# ==========================================
@dataclass
class BrushSettings:
    radius: float = 0.5
    strength: float = 0.1
    falloff_type: int = 0
    mode: int = 0
    use_surface: bool = True


class BrushHitState:
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


# ==========================================
# 🧠 核心笔刷逻辑控制器
# ==========================================
class WeightBrushCore:
    def __init__(self, preview_shape: "cDisplayNode.WeightPreviewShape"):
        self.preview_shape = preview_shape
        self.settings: BrushSettings = self.preview_shape.brush_context.brush_settings
        vertex_count = self.preview_shape.mesh_context.vertex_count
        self.hit_state = BrushHitState(vertex_count)
        self.preview_shape.brush_context.brush_hit_state = self.hit_state

        skin_ctx = self.preview_shape.skin_context
        influences_count = skin_ctx.influences_count
        if not skin_ctx.influences_locks_mgr or len(skin_ctx.influences_locks_mgr.view) != influences_count:
            skin_ctx.influences_locks_mgr = cMemoryView.CMemoryManager.allocate("B", (influences_count,))
            skin_ctx.influences_locks_mgr.view[:] = 0

    def teardown(self):
        if self.preview_shape and self.preview_shape.brush_context: self.preview_shape.brush_context.brush_hit_state = None
        self.hit_state = None; self.settings = None; self.preview_shape = None

    def clear_hit_state(self):
        if self.hit_state: self.hit_state.clear()

    def detect_range(self, center_xyz: tuple, hit_tri: int) -> int:
        mesh_ctx = self.preview_shape.mesh_context
        brush_ctx = self.preview_shape.brush_context
        if not mesh_ctx or not mesh_ctx.rawPoints2D_output: return 0

        brush_ctx.brush_epoch += 1

        self.hit_state.hit_count = cBrushCython.compute_brush_weights_god_mode(
            center_xyz, mesh_ctx.rawPoints2D_output.view, mesh_ctx.tri_indices_2D.view, hit_tri,
            mesh_ctx.adj_offsets.view, mesh_ctx.adj_indices.view, self.settings.radius, self.settings.falloff_type,
            self.settings.use_surface, brush_ctx.brush_epoch, brush_ctx.pool_node_epochs.view,
            brush_ctx.pool_dist.view, brush_ctx.pool_queue.view, brush_ctx.pool_in_queue.view,
            self.hit_state.hit_indices_mgr.view, self.hit_state.hit_weights_mgr.view)

        self.hit_state.hit_center_position = center_xyz
        return self.hit_state.hit_count

    def apply_weight_math(self) -> bool:
        if self.hit_state.hit_count == 0: return False

        skin_ctx = self.preview_shape.skin_context
        modify_weights2D, target_inf, is_mask = self.preview_shape.active_paint_target
        if modify_weights2D is None: return False

        locks_view = cMemoryView.CMemoryManager.allocate("B", (1,)).view if is_mask else skin_ctx.influences_locks_mgr.view

        cBrushCython.skin_weight_brush(
            self.settings.strength, self.settings.mode, target_inf, locks_view, self.hit_state.hit_indices_mgr.view,
            self.hit_state.hit_weights_mgr.view, self.hit_state.hit_count, modify_weights2D.view)
        return True
