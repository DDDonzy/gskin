from __future__ import annotations

import typing
from dataclasses import dataclass

# 统一使用相对路径和模块导入
from . import cMemoryView
from . import cBrushCoreCython
from .cSkinContext import MeshTopologyContext, BrushHitContext

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
    mode        : int   = 1     # 0:Add, 1:Sub, 2:Replace, 3:Multiply, 4:Smooth, 5:Sharp
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

    def __init__(self, preview_shape: cDisplayNode.WeightPreviewShape):
        self.shape = preview_shape
        self.cSkin = preview_shape.cSkin
        self.settings = BrushSettings()

        # 1. 提取基础上下文
        self.mesh_ctx = preview_shape.mesh_context
        self.brush_ctx = preview_shape.brush_context
        vtx_count = self.mesh_ctx.vertex_count
        tri_count = len(self.mesh_ctx.triangle_indices.view) // 3

        # 2. 🟢 极限重构：调用 Cython 构建纯 C 级别的 CSR 邻接表
        self.mesh_ctx = self._build_csr_topology(self.mesh_ctx)

        # 3. 预分配：命中结果缓冲区
        self.brush_ctx.hit_indices = cMemoryView.CMemoryManager.allocate("i", (vtx_count,))
        self.brush_ctx.hit_weights = cMemoryView.CMemoryManager.allocate("f", (vtx_count,))
        self.brush_ctx.hit_count = 0

        # 4. 预分配：引擎广度优先搜索 (BFS) 所需的世代掩码
        self.vertices_epochs = cMemoryView.CMemoryManager.allocate("i", (vtx_count,))

        # 5. 🚀 装载底盘：实例化 CoreBrushEngine
        self.engine = cBrushCoreCython.CoreBrushEngine(
            self.mesh_ctx.vertex_positions.reshape((vtx_count, 3)).view,
            self.mesh_ctx.triangle_indices.reshape((tri_count, 3)).view,
            self.mesh_ctx.adjacency_offsets.view,
            self.mesh_ctx.adjacency_indices.view,
            self.vertices_epochs.view,
            self.brush_ctx.hit_indices.view,
            self.brush_ctx.hit_weights.view,
        )

        # 6. 预分配：Undo/Redo 撤销系统内存池
        inf_count = self.cSkin.influences_count if self.cSkin else 1
        self.modified_vtx_bool_mgr = cMemoryView.CMemoryManager.allocate("B", (vtx_count,))
        self.modified_indices_mgr = cMemoryView.CMemoryManager.allocate("i", (vtx_count,))
        self.undo_buffer_mgr = cMemoryView.CMemoryManager.allocate("f", (vtx_count, inf_count))

        # Stroke
        self.active_processor: cBrushCoreCython.SkinWeightProcessor = None
        self.active_handle = None
        self.active_target_idx = 0

    def refresh_mesh_positions(self):
        """同步最新的网格顶点数据到引擎"""
        # 假设这里你从 preview_shape 获取了最新的顶点上下文
        vtx_count = self.mesh_ctx.vertex_count

        # 重新 reshape 获取 2D 视图
        new_view = self.mesh_ctx.vertex_positions.reshape((vtx_count, 3)).view

        # 💥 告诉 Cython 引擎：指针换了，盯紧这块新内存！
        self.engine.update_vertex_positions(new_view)

    def _build_csr_topology(self, mesh_ctx: MeshTopologyContext) -> MeshTopologyContext:
        """
        [隐式修改，显式返回]
        利用渲染节点的去重边索引，调用 Cython 算法极速构建 CSR 邻接表。
        """
        vtx_count = mesh_ctx.vertex_count
        edge_view = mesh_ctx.edge_indices.view

        # CSR 所需的三个核心连续内存
        offsets_mgr = cMemoryView.CMemoryManager.allocate("i", (vtx_count + 1,))
        indices_mgr = cMemoryView.CMemoryManager.allocate("i", (len(edge_view),))
        temp_cursor = cMemoryView.CMemoryManager.allocate("i", (vtx_count,))

        # 💥 呼叫 Cython C 引擎进行降维打击 (毫秒级完成)
        cBrushCoreCython.build_csr_topology(vtx_count, edge_view, offsets_mgr.view, indices_mgr.view, temp_cursor.view)

        mesh_ctx.adjacency_offsets = offsets_mgr
        mesh_ctx.adjacency_indices = indices_mgr
        return mesh_ctx

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

        self.refresh_mesh_positions()

        # 1. 底层射线检测
        hit_success, hit_pos, hit_normal, hit_tri, _, _, _ = self.engine.raycast(ray_source, ray_dir)

        # 2. 未命中处理
        if not hit_success:
            if action == "hover":
                self.clear_hit_state()
            return None

        # 3. 命中处理, 进行空间球体或拓扑扫描
        if self.mesh_ctx and self.mesh_ctx.vertex_positions:
            hit_count, _, _ = self.engine.calc_brush_weights(
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
    def begin_stroke(self):
        """
        [阶段 1: 鼠标按下]
        解析 UI 目标 (Layer/Mask/Influence)，提取对应内存并装配 Processor。
        """
        shape = self.shape
        render_ctx = self.shape.render_context

        handle, _ = shape.active_paint_weights
        if handle is None or not handle.is_valid:
            return

        self.active_handle = handle

        vtx_count, influences_count, _, weights_1d = handle.get_weights()
        if vtx_count <= 0:
            return
        if render_ctx.paintMask:
            self.active_target_idx = 0
            weights_2d = weights_1d.cast("B").cast("f", (vtx_count, 1))
            # 必须挂载到 self 上，防止被 Python 的垃圾回收器(GC)吃掉导致闪退
            self._temp_locks_mgr = cMemoryView.CMemoryManager.allocate("B", (1,))
        else:
            self.active_target_idx = render_ctx.paintInfluenceIndex
            weights_2d = weights_1d.cast("B").cast("f", (vtx_count, influences_count))
            # 挂载到 self
            self._temp_locks_mgr = cMemoryView.CMemoryManager.allocate("B", (influences_count,))


        self.active_processor = cBrushCoreCython.SkinWeightProcessor(
            self.engine,
            weights_2d,
            self.modified_indices_mgr.view,
            self.modified_vtx_bool_mgr.view,
            self._temp_locks_mgr.view,
            self.undo_buffer_mgr.view,
        )

        # 通知 Processor 清空掩码，开始记录历史
        self.active_processor.begin_stroke()


    def update_stroke(self) -> bool:
        """
        鼠标拖拽，计算内存权重
        """
        if not self.active_processor or self.brush_ctx.hit_count == 0:
            return False

        # 底层内存极限狂飙
        self.active_processor.apply_weight(self.settings.strength, self.settings.mode, self.active_target_idx, self.settings.iter)
        return True

    def end_stroke(self):
        """
        鼠标松开
        提取 Undo 历史，并向 Maya 正式提交这一笔的所有修改。
        """
        if not self.active_processor or not self.active_handle:
            return

        # 1. 结束行程，获取撤销恢复包
        # undo_redo_pack = (count, indices_view, undo_view, redo_view)
        undo_redo_pack = self.active_processor.end_stroke()
        # print(undo_redo_pack)

        # if undo_redo_pack:
        #     count, indices_view, undo_view, redo_view = undo_redo_pack
        #     channels = 1 if self.shape.render_context.paintMask else self.cSkin.influences_count
        #     vtx_count = self.cSkin.vertex_count

        #     # 脱水打包
        #     indices_pack = tuple(indices_view[:count])
        #     undo_pack = tuple(tuple(undo_view[i, j] for j in range(channels)) for i in range(count))
        #     redo_pack = tuple(tuple(redo_view[i, j] for j in range(channels)) for i in range(count))

        #     # ==========================================
        #     # 🚀 见证奇迹的时刻：用 partial 绑定参数！
        #     # ==========================================
        #     from functools import partial
        #     undo_partial = partial(self.execute_sparse_update,
        #                            self.active_handle, vtx_count, channels, indices_pack, undo_pack)

        #     redo_partial = partial(self.execute_sparse_update,
        #                            self.active_handle, vtx_count, channels, indices_pack, redo_pack)

        #     # 塞进万能壳
        #     from .cBrushCommand import CallbackCmd
        #     import maya.cmds as cmds

        #     CallbackCmd._staging_data = {
        #         'undo': undo_partial,
        #         'redo': redo_partial
        #     }

        #     cmds.cCallbackCmd()  # 搞定！入栈！

        # 清除挂载状态
        self.active_processor = None
        self.active_handle = None

    def execute_sparse_update(self, handle, vtx_count, channels, indices, weights_data):
        """还原权重的干活函数"""
        if not handle or not handle.is_valid:
            return

        target_view = handle.memory.reshape((vtx_count, channels)).view
        for i, vtx_idx in enumerate(indices):
            v_weights = weights_data[i]
            for c in range(channels):
                target_view[vtx_idx, c] = v_weights[c]

        self.cSkin.setDirty()
