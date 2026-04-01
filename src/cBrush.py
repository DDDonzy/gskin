from __future__ import annotations

import maya.cmds as cmds
import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui


from ._cRegistry import SkinRegistry
from .cBrushTabletInput import TabletTracker
from .cBrushInterpolator import LinearStrokeInterpolator, SplineStrokeInterpolator
from .cBrushManager import WeightBrushManager, BrushSettings

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer


def maya_useNewAPI():
    pass


class WeightBrushContext(omui.MPxContext):
    """
    笔刷交互上下文 (UI 视图层)。
    所有底层数据、范围判定和运算逻辑均交由 WeightBrushManager 一站式处理。
    """

    brushLine = 2.0
    brushColor = om.MColor((1.0, 0.5, 0.0, 1.0))
    brushPresColor = om.MColor((1.0, 1.0, 1.0, 1.0))

    def __init__(self):
        super().__init__()

        # 核心管理器与追踪器
        self.brush_manager: WeightBrushManager = None

        self.stroke_tracker = LinearStrokeInterpolator()

        # 视口与网格上下文
        self.fn_mesh: om.MFnMesh = None
        self.mesh_dag_path: om.MDagPath = None
        self._view: omui.M3dView = None

        # 射线缓存 (避免每帧重复创建对象)
        self._ray_source = om.MPoint()
        self._ray_direction = om.MVector()

        # UI 状态
        self._isPressed = False
        # Brush Pressure 由 TabletTracker 维护 并通过 BrushSettings._pressure 实时传递给 BrushManager
        self.tablet_tracker: TabletTracker = None

    def toolOnSetup(self, event):
        """工具启动时的初始化装配"""
        self._into_brush()
        # 启动数位板追踪器 开始采集压力数据
        self.tablet_tracker = TabletTracker()
        self.tablet_tracker.start()

    def toolOffCleanup(self):
        """工具退出 通知管理器清理内存"""
        if self.brush_manager:
            self.brush_manager.teardown()
        self.__init__()

    # === Event

    def doPtrMoved(self, event, drawMgr, context):
        """悬停阶段 单次检测与高亮"""
        # get local ray & ray_dir
        ray_src, ray_dir = self.get_ray_from_screen(*event.position)
        #  raycast
        is_hit, hit_pos, hit_normal, _ = self.brush_manager.raycast(ray_src, ray_dir)
        # draw cursor
        if is_hit:
            self.draw_cursor((True, hit_pos, hit_normal), drawMgr)

    def doPress(self, event, drawMgr, context):
        """按下阶段 锁定内存并强制绘制第一笔"""
        self._isPressed = True

        # redo/undo record
        if self.brush_manager:
            self.brush_manager.begin_stroke()

        # 手绘板压力值传递给 BrushSettings 以供 BrushManager 使用

        # brush path interpolator
        interp_points = self.stroke_tracker.begin_stroke(*event.position, self.tablet_tracker.pressure)

        last_hit_result = None
        for x, y, p in interp_points:
            ray_src, ray_dir = self.get_ray_from_screen(x, y)
            last_hit_result = self.brush_manager.stroke(ray_src, ray_dir, is_pressed=True, value=BrushSettings.strength, pressure=p)

        if last_hit_result[0] is True:
            self._update_dynamic_spacing(last_hit_result)
            self.draw_cursor(last_hit_result, drawMgr)

    def doDrag(self, event, drawMgr, context):
        """拖拽阶段 纯数据驱动的批量涂抹"""

        # 手绘板压力值传递给 BrushSettings 以供 BrushManager 使用
        BrushSettings._pressure = self.tablet_tracker.pressure
        print(f"Current Pressure: {BrushSettings._pressure:.3f}")

        # brush path interpolator
        interp_points = self.stroke_tracker.drag_stroke(*event.position, self.tablet_tracker.pressure)
        interp_points_is_empty = not interp_points

        if not interp_points:
            interp_points = [(*event.position, self.tablet_tracker.pressure)]  # 即使没有移动，也要持续更新压力值
        for x, y, p in interp_points:
            ray_src, ray_dir = self.get_ray_from_screen(x, y)
            # raycast
            self.brush_manager.stroke(ray_src, ray_dir, is_pressed=not interp_points_is_empty, value=BrushSettings.strength, pressure=p)

        # draw
        draw_raycast = self.brush_manager.raycast(*self.get_ray_from_screen(*event.position))
        if draw_raycast[0] is True:
            _ = (draw_raycast[0], draw_raycast[1], draw_raycast[2])
            self._update_dynamic_spacing(_)
            self.draw_cursor(_, drawMgr)
        else:
            self.refresh()

    def doRelease(self, event, drawMgr, context):
        """松开阶段 强制闭合端点并提交撤销栈"""
        self._isPressed = False

        self.stroke_tracker.end_stroke(*event.position)

        if self.brush_manager:
            self.brush_manager.end_stroke()
            self.brush_manager.clear_hit_state()

    # ==============================================================================

    def get_ray_from_screen(self, screen_x: float, screen_y: float) -> tuple:
        """辅助函数：将屏幕 2D 坐标转换为局部空间的射线 (source, dir)"""
        self._view.viewToWorld(int(screen_x), int(screen_y), self._ray_source, self._ray_direction)
        inv_matrix = self.mesh_dag_path.inclusiveMatrixInverse()
        ray_src_obj = self._ray_source * inv_matrix
        ray_dir_obj = self._ray_direction * inv_matrix
        return tuple(ray_src_obj)[0:3], tuple(ray_dir_obj)

    def draw_cursor(self, hit_result, drawMgr):
        """专门负责解析结果 更新光标位置 并通知 Maya 刷新画面"""
        if not hit_result:
            return

        is_hit, hit_pos_obj, hit_normal_obj = hit_result

        if is_hit is False:
            return

        self.refresh()
        radius = self.brush_manager.settings.radius
        color = self.brushColor if not self._isPressed else self.brushPresColor
        brush_line = self.brushLine

        cursor_pos = om.MPoint(hit_pos_obj) * self.mesh_dag_path.inclusiveMatrix()
        inv_transpose_matrix = self.mesh_dag_path.inclusiveMatrixInverse().transpose()
        cursor_normal = om.MVector(hit_normal_obj) * inv_transpose_matrix

        drawMgr.beginDrawable()
        drawMgr.setColor(color)
        drawMgr.setLineWidth(brush_line)
        drawMgr.circle(cursor_pos, cursor_normal, radius)
        drawMgr.endDrawable()

    def _update_dynamic_spacing(self, hit_result):
        """将 3D 半径投影为 2D 像素，并动态更新插值器"""
        if not hit_result or not hit_result[0]:  # 如果没击中模型，就不更新
            return

        _, hit_pos_obj, _ = hit_result
        radius_3d = self.brush_manager.settings.radius

        spacing_percentage = BrushSettings.brush_spacing_ratio

        # 1. 局部坐标转世界坐标
        world_matrix = self.mesh_dag_path.inclusiveMatrix()
        hit_pos_world = om.MPoint(hit_pos_obj) * world_matrix

        # 2. 获取当前摄像机的 Right 向量 (用于在屏幕平面上偏移)
        cam_path = self._view.getCamera()
        right_dir = om.MFnCamera(cam_path).rightDirection(om.MSpace.kWorld)

        # 3. 在 3D 空间中，向右侧推移一个笔刷半径的距离
        edge_pos_world = hit_pos_world + (right_dir * radius_3d)

        # 4. 把中心点和边缘点，投影回 2D 屏幕坐标！
        cx, cy, _ = self._view.worldToView(hit_pos_world)
        ex, ey, _ = self._view.worldToView(edge_pos_world)

        # 5. 用勾股定理算出这根 3D 半径在当前屏幕上到底有几个像素长

        pixel_radius = math.hypot(ex - cx, ey - cy)

        # 6. 计算最终的动态间距 (限制最低不能小于 1 个像素，防止死循环爆炸)
        final_spacing = max(1.0, pixel_radius * spacing_percentage)

        # 7. 实时更新给底层纯数据插值器！
        self.stroke_tracker.spacing = final_spacing

    def refresh(self):
        # 刷新 view 视图
        self._view.refresh(False, True)
        pass

    def _into_brush(self):
        try:
            mSel: om.MSelectionList = om.MGlobal.getActiveSelectionList()
            if mSel.isEmpty():
                raise RuntimeError("请先选择一个模型。")

            # 1. 🛡️ 安全获取 Shape 的 MDagPath (用于计算世界/局部矩阵)
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

            # =======================================================
            # 💥 2. 暴力硬编码获取渲染节点 (用于测试)
            # =======================================================
            cSkin_node_name = "cSkinDeformer1"
            self.cSkin: CythonSkinDeformer = SkinRegistry.from_instance_by_string(cSkin_node_name)

            if not self.cSkin:
                raise RuntimeError("未提取到 cSkinDeformer 的 Python 实例。")

            # 3. 🚀 装配无状态笔刷系统
            self._view = omui.M3dView.active3dView()
            self.brush_manager = WeightBrushManager(self.cSkin)

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
