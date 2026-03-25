from __future__ import annotations

import maya.cmds as cmds
import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui
import maya.api.OpenMayaRender as omr


from .cBrushManager import WeightBrushManager
from ._cRegistry import SkinRegistry

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer


def maya_useNewAPI():
    pass


class WeightBrushContext(omui.MPxContext):
    """
    笔刷交互上下文 (UI 视图层)。
    只负责 捕获鼠标事件、发射射线、绘制 UI 光标。
    所有底层数据、范围判定和运算逻辑均交由 WeightBrushManager 一站式处理。
    """

    brushLine = 2.0
    brushColor = om.MColor((1.0, 0.5, 0.0, 1.0))
    brushPresColor = om.MColor((1.0, 1.0, 1.0, 1.0))

    def __init__(self):
        super().__init__()

        # 核心管理器 (Controller)
        self.brush_manager: WeightBrushManager = None

        # 视口与网格上下文
        self.fn_mesh: om.MFnMesh = None
        self.mesh_dag_path: om.MDagPath = None
        self._view: omui.M3dView = None

        # 射线缓存 (避免每帧重复创建对象)
        self._ray_source = om.MPoint()
        self._ray_direction = om.MVector()

        # UI 光标绘制缓存 (世界空间)
        self._isPressed = False

        # 笔刷连续性插值状态
        self._last_mouse_x = None
        self._last_mouse_y = None
        self.brush_spacing_pixels = 5  # 插值阈值(像素) 越小越连贯 建议 3~5

    def toolOnSetup(self, event):
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
            # 💥 2. 暴力硬编码：：： 过名字获取渲染节点 (用于测试)
            # =======================================================
            cSkin_node_name = "cSkinDeformer1"
            self.cSkin: CythonSkinDeformer = SkinRegistry.from_instance_by_string(cSkin_node_name)

            if not self.cSkin:
                raise RuntimeError("未提取到 WeightPreviewShape 的 Python 实例。")

            # 3. 🚀 装配无状态笔刷系统
            self._view = omui.M3dView.active3dView()
            self.brush_manager = WeightBrushManager(self.cSkin)

            print(f"[Brush] 引擎测试版启动成功 已暴力连接至 -> {cSkin_node_name}")

        except Exception:
            import traceback

            err_msg = traceback.format_exc()
            cmds.evalDeferred(lambda: cmds.setToolTo("selectSuperContext"))
            om.MGlobal.displayError(f"笔刷初始化失败:\n{err_msg}")

    def toolOffCleanup(self):
        """工具退出 通知管理器清理内存"""
        print("off")
        if self.brush_manager:
            self.brush_manager.teardown()
        self.__init__()

    def doPtrMoved(self, event, drawMgr, context):
        """悬停阶段 单次检测与高亮"""
        x, y = event.position
        result = self.brush_process(x, y, "hover")
        self.draw_cursor(result, drawMgr)

    def doPress(self, event, drawMgr, context):
        """按下阶段 锁定内存 记录初始坐标"""
        self._isPressed = True
        x, y = event.position
        self._last_mouse_x = x
        self._last_mouse_y = y

        if self.brush_manager:
            self.brush_manager.begin_stroke()

        result = self.brush_process(x, y, "press")
        self.draw_cursor(result, drawMgr)

    def doDrag(self, event, drawMgr, context):
        """拖拽阶段 疯狂涂抹 + 极速补点插值"""
        curr_x, curr_y = event.position

        # 防御性代码 防止还没按下就触发拖拽
        if self._last_mouse_x is None or self._last_mouse_y is None:
            self._last_mouse_x, self._last_mouse_y = curr_x, curr_y

        import math

        dist_2d = math.hypot(curr_x - self._last_mouse_x, curr_y - self._last_mouse_y)

        last_result = None

        # 🌟 1. 判断是否需要插值补点
        if dist_2d >= self.brush_spacing_pixels:
            steps = int(dist_2d / self.brush_spacing_pixels)

            # 开始补点循环 沿着鼠标轨迹密集开火
            for i in range(1, steps + 1):
                t = i / steps
                interp_x = self._last_mouse_x + (curr_x - self._last_mouse_x) * t
                interp_y = self._last_mouse_y + (curr_y - self._last_mouse_y) * t

                # 狂暴计算 不刷新UI
                last_result = self.brush_process(interp_x, interp_y, "drag")

            # 更新记忆坐标
            self._last_mouse_x = curr_x
            self._last_mouse_y = curr_y

            # 🌟 2. 循环结束后，，， 最后一个点的结果去刷新视口 1 次
            self.draw_cursor(last_result, drawMgr)

        else:
            # 距离太小 (比如只移动了 1 个像素) -> 触发“物理防抖”
            # 直接无视这次事件 不产生任何开销 拯救 GC
            pass

    def doRelease(self, event, drawMgr, context):
        """松开阶段 结束行程"""
        self._isPressed = False
        self._last_mouse_x = None
        self._last_mouse_y = None

        if self.brush_manager:
            self.brush_manager.end_stroke()
            self.brush_manager.clear_hit_state()

    def brush_process(self, screen_x: float, screen_y: float, stroke_action: str):
        """只负责发射射线和运算底层权重 返回击中结果 绝对不触发视口重绘"""
        if not self.brush_manager:
            return None

        # raycast position & direction to local space
        self._view.viewToWorld(int(screen_x), int(screen_y), self._ray_source, self._ray_direction)
        inv_matrix = self.mesh_dag_path.inclusiveMatrixInverse()
        ray_source_obj = self._ray_source * inv_matrix
        ray_dir_obj = self._ray_direction * inv_matrix

        return self.brush_manager.process_stroke(tuple(ray_source_obj)[0:3], tuple(ray_dir_obj), stroke_action)

    def _refresh_viewport(self):
        """触发 Shape 和 3D 视口刷新"""
        self._view.refresh(False, False)

    def draw_cursor(self, hit_result, drawMgr):
        """专门负责解析结果 更新光标位置 并通知 Maya 刷新画面"""
        is_hit, hit_pos_obj, hit_normal_obj = hit_result

        if is_hit is False:
            return

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
