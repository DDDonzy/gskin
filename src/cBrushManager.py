from __future__ import annotations

import typing
from dataclasses import dataclass

# 统一使用相对路径和模块导入
from . import cBufferManager
from . import cBrushCoreCython as cBrushCoreCython
from . import apiundo

from ._cProfilerCython import MayaNativeProfiler, maya_profile

if typing.TYPE_CHECKING:
    from . import cDisplayNode


# ==============================================================================
# 🎛️ 配置结构
# ==============================================================================
@dataclass
class BrushSettings:
    """存放笔刷半径、强度、模式等用户 UI 配置。"""

    # fmt:off
    radius      : float = 0.5
    strength    : float = 0.1
    iter        : int   = 10
    falloff_type: int   = 1     # 0:Linear, 1:Airbrush, 2:Solid, 3:Dome, 4:Spike
    mode        : int   = 0     # 0:Add, 1:Sub, 2:Replace, 3:Multiply, 4:Smooth, 5:Sharp
    use_surface : bool  = True
    # fmt:on


# ==============================================================================
# 🧠 笔刷管理器 (纯算法装配车间，彻底无状态)
# ==============================================================================
class WeightBrushManager:
    """
    负责将 Maya Shape 中的网格数据与底层 Cython 计算引擎组装在一起。
    管理完整的 Stroke (绘制行程) 生命周期，并接管 Undo/Redo 内存池。
    """

    @maya_profile(7, "init")
    def __init__(self, preview_shape: cDisplayNode.WeightPreviewShape):
        self.shape = preview_shape
        self.cSkin = preview_shape.cSkin
        self.settings = BrushSettings

        # 提取基础上下文
        self.mesh_ctx = preview_shape.mesh_context
        self.brush_ctx = preview_shape.brush_context
        vtx_count = self.mesh_ctx.vertex_count
        tri_count = len(self.mesh_ctx.triangle_indices.view) // 3

        # =================================================================
        # 🚀 1. 极简实例化：只喂给引擎最基础的物理数据，剩下的全交到底层黑盒！
        # =================================================================
        self.engine = cBrushCoreCython.CoreBrushEngine(
            self.mesh_ctx.vertex_positions.reshape((vtx_count, 3)).view,
            self.mesh_ctx.triangle_indices.reshape((tri_count, 3)).view,
            self.mesh_ctx.edge_indices.view,  # 传入 1D 的边缘数组
        )

        # =================================================================
        # 🚀 2. 反向提取：引擎在底层造好了内存，我们把它“挂”到上下文里供全局使用
        # 这样就能保证 Python 层和 C 层读写的绝对是同一块物理内存！
        # =================================================================
        self.brush_ctx.hit_indices = self.engine.out_hit_indices
        self.brush_ctx.hit_weights = self.engine.out_hit_falloff
        self.brush_ctx.hit_count = 0  # 初始命中数为 0

        # =================================================================
        # 🛡️ 3. 预分配 Undo/Redo 撤销系统内存池
        # =================================================================
        inf_count = self.cSkin.influences_count if self.cSkin else 1
        self.modified_vtx_bool_mgr = cBufferManager.BufferManager.allocate("B", (vtx_count,))
        self.modified_vtx_indices_mgr = cBufferManager.BufferManager.allocate("i", (vtx_count,))
        self.undo_buffer_mgr = cBufferManager.BufferManager.allocate("f", (vtx_count, inf_count))
        """ 撤销内存池 [i, j]：一次性预分配全量权重快照空间，用于备份刷写前的原始权重 """

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

    @maya_profile(0, "raycast")
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
        # deform 输出的 点位置信息不一定是同一个内存地址的，在并行模式下，可能多个内存地址切换
        # 所以每次tick的时候，要更新笔刷底层的点位置信息，直接传入内存地址即可。
        vtx_count = self.mesh_ctx.vertex_count
        new_view = self.mesh_ctx.vertex_positions.reshape((vtx_count, 3)).view
        self.engine.update_vertex_positions(new_view)
        # ----------------------------------------------------------------------------------
        with MayaNativeProfiler("Raycast", 6):
            # ray cast
            hit_success, hit_pos, hit_normal, hit_tri, _, _, _ = self.engine.raycast(ray_source, ray_dir)
            
        # 未命中处理
        if not hit_success:
            # if action == "hover":
            self.clear_hit_state()
            return None

        with MayaNativeProfiler("Falloff", 5):
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

        # 4. 动作分发：如果是按下或拖拽，执行核心涂抹运算
        if action in ("press", "drag"):
            self.update_stroke()
            self.cSkin.setDirty()

        # 5. 返回局部坐标和法线，供 UI 层转换世界坐标画圈

        return (hit_pos, hit_normal)

    # ==============================================================================
    # 🖌️ Stroke 生命周期管理 (按下 -> 拖拽 -> 松开)
    # ==============================================================================
    @maya_profile(2, "begin_stroke")
    def begin_stroke(self):
        """
        鼠标按下
        解析 UI 目标 (Layer/Mask/Influence)，提取对应内存并装配 Processor。
        """
        weights_manager = self.cSkin.weights_manager
        render_ctx = self.shape.render_context

        layer_idx = render_ctx.paintLayerIndex  # layer
        is_mask = render_ctx.paintMask  # mask

        self.layer_idx = layer_idx  # layer
        self.is_mask = is_mask  # layer
        self.active_influence_idx = render_ctx.paintInfluenceIndex  # influence index

        handle = weights_manager.get_handle(layer_idx, is_mask)
        if not handle.is_valid:
            return

        raw_weights = weights_manager.get_raw_weights(layer_idx, is_mask)
        vtx_count, influences_count, _, weights_1d = weights_manager.parse_raw_weights(raw_weights)
        if vtx_count <= 0:
            return

        weights_2d = weights_1d.cast("B").cast("f", (vtx_count, influences_count))
        # influence locked indices
        self._temp_locks_mgr = cBufferManager.BufferManager.allocate("B", (influences_count,))

        self.active_processor = cBrushCoreCython.SkinWeightProcessor(
            weights_2d,
            self.modified_vtx_indices_mgr.view,
            self.modified_vtx_bool_mgr.view,
            self._temp_locks_mgr.view,
            self.undo_buffer_mgr.view,
        )

        # 通知 Processor 清空掩码，开始记录历史
        self.active_processor.begin_stroke()

    @maya_profile(3, "update_stroke")
    def update_stroke(self) -> bool:
        """
        按下鼠标持续拖拽Tick，计算内存权重
        """
        if not self.active_processor or self.brush_ctx.hit_count == 0:
            return False

        self.active_processor.apply_weight_single(
            brush_mode      = self.settings.mode,
            value           = self.settings.strength,
            channel_index   = self.active_influence_idx,
            vertex_count    = self.brush_ctx.hit_count,          # 传入点数
            vertex_indices  = self.brush_ctx.hit_indices,   # 传入顶点数组
            falloff_weights = self.brush_ctx.hit_weights,    # 入衰减权重数组
            iterations      = self.settings.iter,
        )  # fmt:skip
        return True

    @maya_profile(5, "end_stroke")
    def end_stroke(self):
        """
        鼠标松开
        提取 Undo 历史，并向 Maya 正式提交这一笔的所有修改。
        """
        if not self.active_processor:
            return

        # 返回值: (mod_vertex_indices, mod_channel_indices, old_sparse_ary, new_sparse_ary)
        undo_redo_pack = self.active_processor.end_stroke()

        if undo_redo_pack:
            mod_vtx_idx, mod_ch_idx, old_sparse, new_sparse = undo_redo_pack

            # 提前提取引用，防止闭包晚绑定陷阱
            wm = self.cSkin.weights_manager
            layer = self.layer_idx
            mask = self.is_mask

            # ✨ 核心改变：使用专用的稀疏状态还原器，强行覆盖，绝对精准！
            def undo():
                wm.set_sparse_data(layer, mask, mod_vtx_idx, mod_ch_idx, old_sparse)

            def redo():
                wm.set_sparse_data(layer, mask, mod_vtx_idx, mod_ch_idx, new_sparse)

            # 提交到咱们自己的 API 撤销栈
            apiundo.commit(redo, undo, execute=False)

        # 清除挂载状态，让 Python 回收临时对象
        self.active_processor = None
        self.active_handle = None
