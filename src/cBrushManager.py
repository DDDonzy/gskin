from __future__ import annotations


import array
import typing
from dataclasses import dataclass

from ._cProfilerCython import MayaNativeProfiler, maya_profile


if typing.TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer


# ==============================================================================
# 🎛️ 配置结构
# ==============================================================================
@dataclass
class BrushSettings:
    """存放笔刷半径、强度、模式等用户 UI 配置。"""

    # fmt:off
    radius              : float = 1
    strength            : float = 1.0
    iter                : int   = 10
    falloff_type        : int   = 1     # 0:Linear, 1:Airbrush, 2:Solid, 3:Dome, 4:Spike
    mode                : int   = 0     # 0:Add, 1:Sub, 2:Replace, 3:Multiply, 4:Smooth, 5:Sharp
    brush_spacing_ratio : float = 0.1
    use_surface         : bool  = True
    _pressure           : float = 1.0   # 内部压力值 由 TabletTracker 维护
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

        # Stroke 状态
        self._stroke_coroutine = None  # 🌟 新增：存放协程上下文
        self.active_influence_idx = 0

        # brush
        self.prev_hit_position: tuple | None = None

    def clear_hit_state(self):
        """清空高亮状态 (光标离开模型时)。"""
        if self.brush_ctx:
            self.brush_ctx.clear()

    def teardown(self):
        """彻底清理内存资源与引用。"""
        self.clear_hit_state()
        if self._stroke_coroutine:
            self._stroke_coroutine.close()
        self._stroke_coroutine = None

    # Stroke (按下 -> 拖拽 -> 松开)
    def begin_stroke(self):
        """
        鼠标按下
        解析 UI 目标 (Layer/Mask/Influence) 提取对应内存并装配 Processor。
        """

        layer_idx, is_mask, active_influence_idx, _ = self.cSkin.get_active_paint_weights()
        self.layer_idx = layer_idx
        self.is_mask = is_mask
        self.active_influence_idx = active_influence_idx

        # 🌟 核心：向 WeightsManager 索要魔法协程
        self._stroke_coroutine = self.cSkin.weights_manager.paint_stroke_coroutine(layer_idx, is_mask)

        # 预激协程 (执行到第一个 yield)，如果底层引擎返回 False，说明该层异常，直接抛弃
        if next(self._stroke_coroutine) is False:
            self._stroke_coroutine = None
            return

        self.engine.lock_mesh()
        self.prev_hit_position = None

    def stroke(self, ray_src, ray_dir, is_pressed=False, value=1.0, pressure=1.0):
        is_hit, hit_pos, hit_normal, hit_tri = self.raycast(ray_src, ray_dir)
        if is_hit and is_pressed is True:
            self._apply_brush(
                hit_pos,
                hit_tri,
                self.prev_hit_position,
                is_pressed,
                value,
                pressure,
            )
            self.prev_hit_position = hit_pos
            return (True, hit_pos, hit_normal)
        return (False, None, None)

    def end_stroke(self):
        """
        鼠标松开
        提取 Undo 历史 并向 Maya 正式提交这一笔的所有修改。
        """
        if self._stroke_coroutine:
            self._stroke_coroutine.close()
            self._stroke_coroutine = None

        self.engine.unlock_mesh()
        self.prev_hit_position = None

    @maya_profile("raycast", 2)
    def raycast(self, ray_source: tuple, ray_dir: tuple) -> tuple:
        """
        [1. 探测阶段] 纯物理射线探测，绝对不触发任何涂抹计算。
        Returns:
            tuple: (is_hit, hit_pos, hit_normal, hit_tri)
        """
        if not self.engine:
            return False, None, None, None

        hit_success, hit_pos, hit_normal, hit_tri, _, _, _ = self.engine.raycast(ray_source, ray_dir)

        if not hit_success:
            self.clear_hit_state()
            return False, None, None, None

        return True, hit_pos, hit_normal, hit_tri

    def _apply_brush(
        self,
        hit_pos: tuple,
        hit_tri: int,
        perv_hit_pos: tuple | None = None,
        is_pressed: bool = False,
        value: float = 1.0,
        pressure: float = 1.0,
    ):
        if perv_hit_pos is None:
            perv_hit_pos = hit_pos

        with MayaNativeProfiler("cal_falloff", 3):
            # 1. 计算笔刷衰减 (保持不变)
            if self.mesh_ctx and self.mesh_ctx.vertex_positions:
                hit_count, _, _ = self.engine.calc_brush_falloff(hit_pos, perv_hit_pos, hit_tri, self.settings.radius, self.settings.falloff_type, self.settings.use_surface)
                self.brush_ctx.hit_count = hit_count
                self.brush_ctx.hit_center_position = hit_pos

        with MayaNativeProfiler("process_stroke", 4):
            # 2. 向协程投喂计算参数
            if is_pressed is True:
                # 如果没按中，或者协程没正常启动，直接退出
                if not self._stroke_coroutine or self.brush_ctx.hit_count == 0:
                    return

                val_ary = array.array("f", [value])
                idx_ary = array.array("i", [self.active_influence_idx])

                # 🌟 把当前帧的所有参数打包成 kwargs 字典
                stroke_kwargs = {
                    "brush_mode": self.settings.mode,
                    "weights_value": val_ary,
                    "influences_indices": idx_ary,
                    "pressure": pressure,
                    "clamp_min": 0.0,
                    "clamp_max": 1.0,
                    "iterations": self.settings.iter,
                }

                # 🌟 魔法触发：用 send() 把数据喂给在 yield 处苦苦等待的底层引擎！
                self._stroke_coroutine.send(stroke_kwargs)

        with MayaNativeProfiler("set_dirty", 5):
            if hit_count > 0:
                self.cSkin.fast_preview_deform(self.brush_ctx.hit_indices, hit_count)
