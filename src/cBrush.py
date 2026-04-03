from __future__ import annotations

import math
import array


from gskin.src.cSkinDeform import MeshTopologyContext
import maya.cmds as cmds
import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui

from ._cRegistry import SkinRegistry
from .cBrushTabletInput import TabletTracker
from .cBrushInterpolator import LinearStrokeInterpolator, SplineStrokeInterpolator
from .cBrushSettings import BrushSettings
from .cWeightsManager import StrokeParameters

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer
    from .cBrushCore2Cython import CoreBrushEngine
    from .cSkinContext import BrushHitContext


def maya_useNewAPI():
    pass


# ==============================================================================
# 🧠 笔刷交互与算法总控室 (高内聚版本)
# ==============================================================================
class WeightBrushContext(omui.MPxContext):
    """
    笔刷交互上下文。
    直接与 Maya 视口交互，并接管底层 Cython 引擎的数据传输与 Undo/Redo 生命周期。
    """

    brushLine = 2.0
    brushColor = om.MColor((1.0, 0.5, 0.0, 1.0))
    brushPresColor = om.MColor((1.0, 1.0, 1.0, 1.0))

    def __init__(self):
        super().__init__()

        self.settings = BrushSettings

        # 核心跟踪器
        self.stroke_tracker: LinearStrokeInterpolator = None
        self.tablet_tracker: TabletTracker = None

        # 视口与网格上下文
        self.fn_mesh: om.MFnMesh = None
        self.mesh_dag_path: om.MDagPath = None
        self._view: omui.M3dView = None

        # 射线缓存 (避免每帧重复创建对象)
        self._ray_source = om.MPoint()
        self._ray_direction = om.MVector()

        # 引擎与上下文指针
        self.cSkin: CythonSkinDeformer = None
        self.engine: CoreBrushEngine = None
        self.mesh_ctx: MeshTopologyContext = None
        self.brush_ctx: BrushHitContext = None

        # 状态机变量
        self._isPressed: bool = False
        self._stroke_coroutine = None
        self._active_influence_idx = 0
        self._prev_hit_position: tuple | None = None

    def toolOnSetup(self, event):
        """工具启动时的初始化装配"""
        self._into_brush()
        self.stroke_tracker = LinearStrokeInterpolator()
        self.tablet_tracker = TabletTracker()
        self.tablet_tracker.start()
        self.tablet_tracker.last_pressure = 0.0

    def toolOffCleanup(self):
        """工具退出 通知清理内存"""
        self._clear_hit_state()
        if self._stroke_coroutine:
            self._stroke_coroutine.close()
            self._stroke_coroutine = None
        if self.engine:
            self.engine.unlock_mesh()
        self.__init__()

    # ==========================================================================
    # 🖱️ 核心交互事件 (Event Loop)
    # ==========================================================================

    def doPtrMoved(self, event, drawMgr, context):
        """悬停阶段 单次检测与高亮"""
        is_hit, hit_pos, hit_normal = self._process_single_point(event.position[0], event.position[1], pressure=1.0, is_pressed=False)
        if is_hit:
            self.draw_cursor(hit_pos, hit_normal, drawMgr)

    def doPress(self, event, drawMgr, context):
        """按下阶段 锁定内存、装配协程并强制绘制第一笔"""
        self._isPressed = True
        self._prev_hit_position = None

        # 1. 解析目标图层并向 WeightsManager 索要涂抹协程
        layer_idx, is_mask, inf_idx, _ = self.cSkin.get_active_paint_weights()
        self._active_influence_idx = inf_idx

        self._stroke_coroutine = self.cSkin.weights_manager.paint_stroke_coroutine(layer_idx, is_mask)
        if next(self._stroke_coroutine) is False:
            self._stroke_coroutine = None  # 引擎装配失败，直接抛弃
            return

        self.engine.lock_mesh()

        # 2. 启动插值器记录
        self.stroke_tracker.begin_stroke(*event.position, self.tablet_tracker.pressure)
        self.tablet_tracker.last_pressure = self.tablet_tracker.pressure

        # 3. 不绘制,只raycast,获取第一笔的命中位置与法线,并刷新光标
        is_hit, hit_pos, hit_normal = self._process_single_point(event.position[0], event.position[1], pressure=0, is_pressed=False)
        if is_hit:
            self.draw_cursor(hit_pos, hit_normal, drawMgr)

    def doDrag(self, event, drawMgr, context):
        """拖拽阶段 纯数据驱动的批量涂抹 (已修复双重射线与硬件跳变)"""

        # 获取笔刷轨迹插值点
        interp_points = self.stroke_tracker.drag_stroke(*event.position, self.tablet_tracker.pressure)

        # 3. 如果产生了插值点，依次执行真实涂抹计算
        if interp_points:
            for x, y, p in interp_points:
                self._process_single_point(x, y, pressure=p, is_pressed=True)
                self.tablet_tracker.last_pressure = p  # 更新压力缓存

        # 4. 针对当前鼠标最新位置进行一次单纯探测，刷新光标 (只探测)
        is_hit, hit_pos, hit_normal = self._process_single_point(event.position[0], event.position[1], pressure=self.tablet_tracker.pressure, is_pressed=False)

        if is_hit:
            self._update_dynamic_spacing(hit_pos)
            self.draw_cursor(hit_pos, hit_normal, drawMgr)
        else:
            self.refresh()

    def doRelease(self, event, drawMgr, context):
        """松开阶段 闭合最后一点并提交撤销栈"""

        self._process_single_point(event.position[0], event.position[1], pressure=self.tablet_tracker.last_pressure, is_pressed=True)

        self._isPressed = False

        # 2. 闭合插值器与协程
        self.stroke_tracker.end_stroke(*event.position)
        if self._stroke_coroutine:
            self._stroke_coroutine.close()
            self._stroke_coroutine = None

        if self.engine:
            self.engine.unlock_mesh()

        self._clear_hit_state()
        self._prev_hit_position = None
        self.refresh()

    # ==========================================================================
    # ⚙️ 核心处理逻辑
    # ==========================================================================

    def _process_single_point(self, screen_x: float, screen_y: float, pressure: float, is_pressed: bool):
        """
        单点处理函数：射线检测 -> 算衰减 -> 发送协程 -> 视口预览
        Returns:
            tuple: (is_hit, hit_pos, hit_normal)
        """
        if not self.engine:
            return False, None, None

        # 1. 获取射线
        self._view.viewToWorld(int(screen_x), int(screen_y), self._ray_source, self._ray_direction)
        inv_matrix = self.mesh_dag_path.inclusiveMatrixInverse()
        ray_src_obj = tuple(self._ray_source * inv_matrix)[0:3]
        ray_dir_obj = tuple(self._ray_direction * inv_matrix)

        # 2. 物理射线求交
        hit_success, hit_pos, hit_normal, hit_tri, _, _, _ = self.engine.raycast(ray_src_obj, ray_dir_obj)

        if not hit_success:
            self._clear_hit_state()
            return False, None, None

        # 3. 涂抹与更新
        if is_pressed and self._stroke_coroutine:
            prev_pos = self._prev_hit_position if self._prev_hit_position else hit_pos

            # 计算衰减
            hit_count, active_indices, active_weights = self.engine.calc_brush_falloff(hit_pos, prev_pos, hit_tri, self.settings.radius, self.settings.falloff_type, self.settings.use_surface)

            # 更新笔刷命中上下文
            self.brush_ctx.hit_indices = active_indices
            self.brush_ctx.hit_weights = active_weights
            self.brush_ctx.hit_count = hit_count
            self.brush_ctx.hit_center_position = hit_pos

            if hit_count > 0:
                params = StrokeParameters(
                    brush_mode         = self.settings.mode,
                    weights_value      = array.array("f", [self.settings.strength]),
                    influences_indices = array.array("i", [self._active_influence_idx]),
                    pressure           = pressure,
                    iterations         = self.settings.iter,
                )  # fmt:skip

                self._stroke_coroutine.send(params)

                self.cSkin.fast_preview_deform(active_indices, hit_count)

            self._prev_hit_position = hit_pos

        return True, hit_pos, hit_normal

    # ==========================================================================
    # 🎨 渲染与辅助工具
    # ==========================================================================

    def _clear_hit_state(self):
        """清空高亮状态 (光标离开模型时)。"""
        if self.brush_ctx:
            self.brush_ctx.clear()

    def draw_cursor(self, hit_pos_obj, hit_normal_obj, drawMgr):
        """绘制绿色光标"""
        self.refresh()
        radius = self.settings.radius
        color = self.brushColor if not self._isPressed else self.brushPresColor

        cursor_pos = om.MPoint(hit_pos_obj) * self.mesh_dag_path.inclusiveMatrix()
        inv_transpose_matrix = self.mesh_dag_path.inclusiveMatrixInverse().transpose()
        cursor_normal = om.MVector(hit_normal_obj) * inv_transpose_matrix

        drawMgr.beginDrawable()
        drawMgr.setColor(color)
        drawMgr.setLineWidth(self.brushLine)
        drawMgr.circle(cursor_pos, cursor_normal, radius)
        drawMgr.endDrawable()

    def _update_dynamic_spacing(self, hit_pos_obj):
        """将 3D 半径投影为 2D 像素，并动态更新插值器"""
        world_matrix = self.mesh_dag_path.inclusiveMatrix()
        hit_pos_world = om.MPoint(hit_pos_obj) * world_matrix

        cam_path = self._view.getCamera()
        right_dir = om.MFnCamera(cam_path).rightDirection(om.MSpace.kWorld)
        edge_pos_world = hit_pos_world + (right_dir * self.settings.radius)

        cx, cy, _ = self._view.worldToView(hit_pos_world)
        ex, ey, _ = self._view.worldToView(edge_pos_world)

        pixel_radius = math.hypot(ex - cx, ey - cy)
        final_spacing = max(1.0, pixel_radius * self.settings.brush_spacing_ratio)

        self.stroke_tracker.spacing = final_spacing

    def refresh(self):
        self._view.refresh(False, True)

    def _into_brush(self):
        """连接 Maya 选择集与底层数据架构"""
        try:
            mSel: om.MSelectionList = om.MGlobal.getActiveSelectionList()
            if mSel.isEmpty():
                raise RuntimeError("请先选择一个模型。")

            dag_path: om.MDagPath = mSel.getDagPath(0)
            if dag_path.hasFn(om.MFn.kTransform):
                try:
                    dag_path.extendToShape()
                except RuntimeError:
                    raise RuntimeError("选中的 Transform 节点下没有找到 Shape。")  # noqa: B904

            if not dag_path.hasFn(om.MFn.kMesh):
                raise RuntimeError("选中的对象不是多边形网格 (Mesh) ")

            self.mesh_dag_path = dag_path
            self.fn_mesh = om.MFnMesh(self.mesh_dag_path)

            cSkin_node_name = "cSkinDeformer1"
            self.cSkin: CythonSkinDeformer = SkinRegistry.from_instance_by_string(cSkin_node_name)

            if not self.cSkin:
                raise RuntimeError("未提取到 cSkinDeformer 的 Python 实例。")

            # 挂载上下文
            self._view = omui.M3dView.active3dView()
            self.engine = self.cSkin.brush_engine
            self.mesh_ctx = self.cSkin.mesh_context
            self.brush_ctx = self.cSkin.brush_hit_context

            print(f"[Brush] 引擎测试版启动成功 已暴力连接至 -> {cSkin_node_name}")

        except Exception:
            import traceback

            err_msg = traceback.format_exc()
            cmds.evalDeferred(lambda: cmds.setToolTo("selectSuperContext"))
            om.MGlobal.displayError(f"笔刷初始化失败:\n{err_msg}")


# ==============================================================================
# 🔌 命令注册器
# ==============================================================================
class WeightBrushContextCmd(omui.MPxContextCommand):
    COMMAND_NAME = "cBrushCtx"

    def __init__(self):
        super().__init__()

    def makeObj(self):
        return WeightBrushContext()

    @staticmethod
    def creator():
        return WeightBrushContextCmd()
