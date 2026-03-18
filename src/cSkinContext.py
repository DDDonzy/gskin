from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cBufferManager import BufferManager


# ==============================================================================
# 📦 数据结构：Shape 节点专属的网格与笔刷上下文
# ==============================================================================
# fmt:off
class MeshTopologyContext:
    """
    统一的数据结构，包含变形/渲染所需的所有网格数据。
    严格区分了动态的顶点位置与静态的拓扑/邻接表，方便 BufferManager 统一管理生命周期。
    """
    __slots__ = ("vertex_count",
                 "edge_count",
                 "polygon_count",
                 "triangle_count",
                 "vertex_positions",
                 "triangle_indices",
                 "edge_indices",
                 "v2v_offsets",
                 "v2v_indices",
                 "v2f_offsets",
                 "v2f_indices") # fmt:skip

    def __init__(self):
        self.vertex_count  : int = 0
        self.edge_count    : int = 0
        self.polygon_count : int = 0
        self.triangle_count: int = 0
        # 点位置 Buffer
        self.vertex_positions:BufferManager = None
        # 基础拓扑 Buffer
        self.triangle_indices:BufferManager = None
        self.edge_indices    :BufferManager = None
        # v2v CSR Buffer 
        self.v2v_offsets:BufferManager = None
        self.v2v_indices:BufferManager = None
        # v2f CSR Buffer
        self.v2f_offsets:BufferManager = None
        self.v2f_indices:BufferManager = None

    def clear(self):
        """可选的清理方法：用于显式释放底层的连续内存"""
        self.vertex_positions = None
        self.triangle_indices = None
        self.edge_indices     = None
        self.v2v_offsets      = None
        self.v2v_indices      = None
        self.v2f_offsets      = None
        self.v2f_indices      = None



class BrushHitContext:
    """存放笔刷在当前 Shape 上的运行时状态"""

    __slots__ = (
        "hit_count",
        "hit_indices",
        "hit_weights",
        "hit_center_position",
    )

    def __init__(self):
        self.hit_count    : int            = 0
        self.hit_indices  : BufferManager = None
        self.hit_weights  : BufferManager = None
        self.hit_center_position: tuple = (0.0, 0.0, 0.0)

    @property
    def is_valid(self) -> bool:
        """检查笔刷命中数据是否完整且有效"""
        return (
            self.hit_count > 0
            and self.hit_indices is not None
            and self.hit_weights is not None
        )
    def clear(self):
        self.hit_count = 0
        self.hit_center_position = (0.0, 0.0, 0.0)


class RenderContext:
    """存放 UI 颜色、渲染模式等显示配置的快照"""
    __slots__ = (
        "render_mode",
        "paintLayerIndex",
        "paintInfluenceIndex",
        "paintMask",
        "color_wire",
        "color_point",
        "color_mask_remapA",
        "color_mask_remapB",
        "color_weights_remapA",
        "color_weights_remapB",
        "color_brush_remapA",
        "color_brush_remapB",
    )

    def __init__(
        self,
        render_mode         : int   = 0,
        paintLayerIndex     : int   = -1,
        paintInfluenceIndex : int   = 0,
        paintMask           : bool  = False,
        color_wire          : tuple = (0.0, 1.0, 1.0, 1.0),
        color_point         : tuple = (1.0, 0.0, 0.0, 1.0),
        color_mask_remapA   : tuple = (0.1, 0.1, 0.1, 0.0),
        color_mask_remapB   : tuple = (0.1, 1.0, 0.1, 0.0),
        color_weights_remapA: tuple = (0.0, 0.0, 0.0, 0.0),
        color_weights_remapB: tuple = (1.0, 1.0, 1.0, 0.0),
        color_brush_remapA  : tuple = (1.0, 0.0, 0.0, 1.0),
        color_brush_remapB  : tuple = (1.0, 1.0, 0.0, 1.0),
    ):
        self.render_mode          = render_mode
        self.paintLayerIndex      = paintLayerIndex
        self.paintInfluenceIndex  = paintInfluenceIndex
        self.paintMask            = paintMask
        self.color_wire           = color_wire
        self.color_point          = color_point
        self.color_mask_remapA    = color_mask_remapA
        self.color_mask_remapB    = color_mask_remapB
        self.color_weights_remapA = color_weights_remapA
        self.color_weights_remapB = color_weights_remapB
        self.color_brush_remapA   = color_brush_remapA
        self.color_brush_remapB   = color_brush_remapB
# fmt:on
