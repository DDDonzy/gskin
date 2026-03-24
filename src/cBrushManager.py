from __future__ import annotations

import array

from dataclasses import dataclass

# 统一使用相对路径和模块导入
from .cBufferManager import BufferManager
from . import cBrushCore2Cython as cBrushCoreCython
from . import apiundo

import typing

if typing.TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer


# ==============================================================================
# 🎛️ 配置结构
# ==============================================================================
@dataclass
class BrushSettings:
    """存放笔刷半径、强度、模式等用户 UI 配置。"""

    # fmt:off
    radius      : float = 1
    strength    : float = 0.1
    iter        : int   = 10
    falloff_type: int   = 1     # 0:Linear, 1:Airbrush, 2:Solid, 3:Dome, 4:Spike
    mode        : int   = 0     # 0:Add, 1:Sub, 2:Replace, 3:Multiply, 4:Smooth, 5:Sharp
    use_surface : bool  = True
    # fmt:on


# ==============================================================================
# 🧠 笔刷管理器 (纯算法装配车间 彻底无状态)
# ==============================================================================
class WeightBrushManager:
    """
    负责将 Maya Shape 中的网格数据与底层 Cython 计算引擎组装在一起。
    管理完整的 Stroke (绘制行程) 生命周期 并接管 Undo/Redo 内存池。
    """

    def __init__(self, cSkin: CythonSkinDeformer):
        self.cSkin = cSkin
        self.settings = BrushSettings

        # sync context
        # 从cSkin 获取 上下文
        self.mesh_ctx = cSkin.mesh_context
        self.brush_ctx = cSkin.brush_context
        self.engine = cSkin.brush_engine

        # 引擎在底层造好了内存 我们把它“挂”到上下文里供全局使用
        # 这样就能保证 Python 层和 C 层读写的绝对是同一块物理内存
        self.brush_ctx.hit_indices = self.engine.raw_hit_indices
        self.brush_ctx.hit_weights = self.engine.raw_hit_falloff
        self.brush_ctx.hit_count = 0  # 初始命中数为 0

        # 预分配 Undo/Redo 撤销系统内存池
        vtx_count = self.mesh_ctx.vertex_count
        inf_count = self.cSkin.influences_count if self.cSkin else 1
        self.modified_vtx_bool_mgr = BufferManager.allocate("B", (vtx_count,))
        self.modified_vtx_indices_mgr = BufferManager.allocate("i", (vtx_count,))
        self.undo_buffer_mgr = BufferManager.allocate("f", (vtx_count, inf_count))
        """ 撤销内存池 [i, j] 一次性预分配全量权重快照空间 用于备份刷写前的原始权重 """

        # Stroke 状态
        self.active_processor: cBrushCoreCython.SkinWeightProcessor = None
        self.active_handle = None
        self.active_influence_idx = 0

    def clear_hit_state(self):
        """清空高亮状态 (光标离开模型时)。"""
        if self.brush_ctx:
            self.brush_ctx.clear()

    def teardown(self):
        """彻底清理内存资源与引用。"""
        self.clear_hit_state()
        self.active_processor = None
        self.active_handle = None

    def process_stroke(self, ray_source: tuple, ray_dir: tuple, action: str) -> tuple:
        """
        封装射线投射、范围检测和笔刷涂抹的完整过程。

        Args:
            ray_source (tuple): 射线起点 (局部空间)
            ray_dir (tuple): 射线方向 (局部空间)
            action (str): 当前鼠标动作 ("hover", "press", "drag")
        Returns:
            tuple: (hit_pos, hit_normal) 供前端绘制光标。若未命中则返回 None。
        """
        if not self.engine:
            return None

        # ----------------------------------------------------------------------------------
        # ray cast
        hit_success, hit_pos, hit_normal, hit_tri, _, _, _ = self.engine.raycast(
            ray_source, ray_dir
        )

        # 未命中处理
        if not hit_success:
            self.clear_hit_state()
            return None

        # 命中处理
        if self.mesh_ctx and self.mesh_ctx.vertex_positions:
            hit_count, _, _ = self.engine.calc_brush_falloff(
                hit_pos,
                hit_tri,
                self.settings.radius,
                self.settings.falloff_type,
                self.settings.use_surface,
            )
            self.brush_ctx.hit_count = hit_count
            self.brush_ctx.hit_center_position = hit_pos

        # 如果是按下或拖拽 执行核心涂抹运算
        if action in ("press", "drag"):
            self.update_stroke()
            self.cSkin.setDirty()

        # 返回局部坐标和法线 供 UI 层转换世界坐标画圈
        return (hit_pos, hit_normal)

    # Stroke (按下 -> 拖拽 -> 松开)
    def begin_stroke(self):
        """
        鼠标按下
        解析 UI 目标 (Layer/Mask/Influence) 提取对应内存并装配 Processor。
        """

        weights_manager = self.cSkin.weights_manager
        layer_idx, is_mask, active_influence_idx, _ = (
            self.cSkin.get_active_paint_weights()
        )
        self.layer_idx = layer_idx  # layer
        self.is_mask = is_mask  # layer
        self.active_influence_idx = active_influence_idx  # influence index

        handle = weights_manager.get_handle(layer_idx, is_mask)
        if not handle.is_valid:
            return
        vtx_count, influences_count, _, weights_1d = handle.parse_raw_weights()
        if vtx_count <= 0:
            return

        weights_2d = weights_1d.cast("B").cast("f", (vtx_count, influences_count))

        # influence locked indices  | 临时创建一个骨骼锁定数组
        self._temp_locks_mgr = BufferManager.allocate("B", (influences_count,))
        # 构造笔刷执行器
        self.active_processor = cBrushCoreCython.SkinWeightProcessor(
            self.engine,
            weights_2d,
            self.modified_vtx_indices_mgr.view,
            self.modified_vtx_bool_mgr.view,
            self._temp_locks_mgr.view,
            self.undo_buffer_mgr.view,
        )

        # 通知 Processor 清空掩码 开始记录历史
        self.active_processor.begin_stroke()

    def update_stroke(self) -> bool:
        if not self.active_processor or self.brush_ctx.hit_count == 0:
            return False

        # 构造临时array
        val_ary = array.array("f", [self.settings.strength])
        idx_ary = array.array("i", [self.active_influence_idx])

        self.active_processor.process_stroke(
            brush_mode=self.settings.mode,
            weights_value=val_ary,
            influences_indices=idx_ary,
            clamp_min=0.0,
            clamp_max=1.0,
            iterations=self.settings.iter,
        )
        return True

    def end_stroke(self):
        """
        鼠标松开
        提取 Undo 历史 并向 Maya 正式提交这一笔的所有修改。
        """
        if not self.active_processor:
            return

        # 返回值: (mod_vertex_indices, mod_channel_indices, old_sparse_ary, new_sparse_ary)
        undo_redo_pack = self.active_processor.end_stroke()

        if undo_redo_pack:
            mod_vtx_idx, mod_ch_idx, old_sparse, new_sparse = undo_redo_pack

            # 提前提取引用 防止闭包晚绑定陷阱
            wm = self.cSkin.weights_manager
            layer = self.layer_idx
            mask = self.is_mask

            # 核心改变 使用专用的稀疏状态还原器 强行覆盖 绝对精准
            def undo():
                wm.set_sparse_weights(layer, mask, mod_vtx_idx, mod_ch_idx, old_sparse)

            def redo():
                wm.set_sparse_weights(layer, mask, mod_vtx_idx, mod_ch_idx, new_sparse)

            # 提交到咱们自己的 API 撤销栈
            apiundo.commit(redo, undo, execute=False)

        self.active_processor = None
        self.active_handle = None
