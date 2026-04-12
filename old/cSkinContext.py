from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..src.cBufferManager import BufferManager


# ==============================================================================
# 📦 数据结构 Shape 节点专属的网格与笔刷上下文
# ==============================================================================
# fmt:off



class BrushHitContext:
    """存放笔刷在当前 Shape 上的运行时状态"""

    __slots__ = (
        "hit_center_position",
        "hit_count",
        "hit_indices",
        "hit_weights",
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



# fmt:on
