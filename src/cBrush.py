import typing
import maya.cmds as cmds
import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui
import maya.api.OpenMayaRender as omr

# 统一使用相对路径和模块导入
from . import cDisplayNode
from . import cBrushManager


def maya_useNewAPI():
    pass


class WeightBrushContext(omui.MPxContext):
    """
    笔刷交互上下文 (UI 视图层)。
    只负责：捕获鼠标事件、发射射线、绘制 UI 光标。
    所有底层数据、范围判定和运算逻辑均交由 WeightBrushManager 一站式处理。
    """

    brushLine = 2.0
    brushColor = om.MColor((1.0, 0.5, 0.0, 1.0))
    brushPresColor = om.MColor((1.0, 1.0, 1.0, 1.0))

    def __init__(self):
        super(WeightBrushContext, self).__init__()

        # 核心管理器 (Controller)
        self.manager: "cBrushManager.WeightBrushManager" = None
        self.preview_shape: "cDisplayNode.WeightPreviewShape" = None

        # 视口与网格上下文
        self.fn_mesh: om.MFnMesh = None
        self.mesh_dag_path: om.MDagPath = None
        self._view: omui.M3dView = None

        # 射线缓存 (避免每帧重复创建对象)
        self._ray_source = om.MPoint()
        self._ray_direction = om.MVector()

        # UI 光标绘制缓存 (世界空间)
        self._cursor_pos: om.MPoint = None
        self._cursor_normal: om.MVector = None
        self._isPressed = False

        # 笔刷连续性插值状态
        self._last_mouse_x = None
        self._last_mouse_y = None
        self.brush_spacing_pixels = 1.0  # 插值阈值(像素)：越小越连贯，建议 3~5

    def toolOnSetup(self, event):
        """工具启动：硬编码测试版，直接抓取 WeightPreview1"""
        try:
            mSel = om.MGlobal.getActiveSelectionList()
            if mSel.isEmpty():
                raise RuntimeError("请先选择一个模型。")

            # 1. 🛡️ 安全获取 Shape 的 MDagPath (用于计算世界/局部矩阵)
            dag_path = mSel.getDagPath(0)
            if dag_path.hasFn(om.MFn.kTransform):
                try:
                    dag_path.extendToShape()
                except RuntimeError:
                    raise RuntimeError("选中的 Transform 节点下没有找到 Shape。")

            if not dag_path.hasFn(om.MFn.kMesh):
                raise RuntimeError("选中的对象不是多边形网格 (Mesh)！")

            self.mesh_dag_path = dag_path
            self.fn_mesh = om.MFnMesh(self.mesh_dag_path)

            # =======================================================
            # 💥 2. 暴力硬编码：直接通过名字获取渲染节点 (用于测试)
            # =======================================================
            target_node_name = "WeightPreviewShape1"
            try:
                mSel_preview = om.MGlobal.getSelectionListByName(target_node_name)
                mObj_preview = mSel_preview.getDependNode(0)
            except RuntimeError:
                raise RuntimeError(f"场景中根本找不到名叫 '{target_node_name}' 的节点！请确认大纲视图。")

            fn_node = om.MFnDependencyNode(mObj_preview)

            if fn_node.typeName == cDisplayNode.NODE_NAME:
                self.preview_shape = fn_node.userNode()
            else:
                raise RuntimeError(f"虽然找到了 {target_node_name}，但它的类型不是 {cDisplayNode.NODE_NAME}。")

            if not self.preview_shape:
                raise RuntimeError("未提取到 WeightPreviewShape 的 Python 实例。")

            # 3. 🚀 装配无状态笔刷系统
            self._view = omui.M3dView.active3dView()
            self.manager = cBrushManager.WeightBrushManager(self.preview_shape)

            print(f"[Brush] 引擎测试版启动成功！已暴力连接至 -> {target_node_name}")

        except Exception as e:
            import traceback

            err_msg = traceback.format_exc()
            cmds.evalDeferred(lambda: cmds.setToolTo("selectSuperContext"))
            om.MGlobal.displayError(f"笔刷初始化失败:\n{err_msg}")

    def toolOffCleanup(self):
        """工具退出：通知管理器清理内存"""
        print("off")
        if self.manager:
            self.manager.teardown()
        self.__init__()

    def _update_ui_cursor_and_refresh(self, result, drawMgr):
        """专门负责解析结果，更新光标位置，并通知 Maya 刷新画面"""
        if result is None:
            self._cursor_pos = None
        else:
            hit_pos_obj, hit_normal_obj = result
            # 将局部命中点转回世界空间，供 doDraw 画圈使用
            self._cursor_pos = om.MPoint(hit_pos_obj) * self.mesh_dag_path.inclusiveMatrix()
            inv_transpose_matrix = self.mesh_dag_path.inclusiveMatrixInverse().transpose()
            self._cursor_normal = om.MVector(hit_normal_obj) * inv_transpose_matrix

        self._refresh_viewport()
        self._draw_brush_cursor(drawMgr)

    # ==============================================================================
    # 🖱️ 鼠标事件路由 (VP2 签名规范)
    # ==============================================================================
    def doPtrMoved(self, event, drawMgr, context):
        """悬停阶段：单次检测与高亮"""
        x, y = event.position
        result = self._fire_ray_and_process(x, y, "hover")
        self._update_ui_cursor_and_refresh(result, drawMgr)

    def doPress(self, event, drawMgr, context):
        """按下阶段：锁定内存，记录初始坐标"""
        self._isPressed = True
        x, y = event.position
        self._last_mouse_x = x
        self._last_mouse_y = y

        if self.manager:
            self.manager.begin_stroke()
            
        result = self._fire_ray_and_process(x, y, "press")
        self._update_ui_cursor_and_refresh(result, drawMgr)

    def doDrag(self, event, drawMgr, context):
        """拖拽阶段：疯狂涂抹 + 极速补点插值"""
        curr_x, curr_y = event.position

        # 防御性代码：防止还没按下就触发拖拽
        if self._last_mouse_x is None or self._last_mouse_y is None:
            self._last_mouse_x, self._last_mouse_y = curr_x, curr_y

        import math
        dist_2d = math.hypot(curr_x - self._last_mouse_x, curr_y - self._last_mouse_y)

        last_result = None

        # 🌟 1. 判断是否需要插值补点
        if dist_2d >= self.brush_spacing_pixels:
            steps = int(dist_2d / self.brush_spacing_pixels)
            
            # 开始补点循环：沿着鼠标轨迹密集开火
            for i in range(1, steps + 1):
                t = i / steps
                interp_x = self._last_mouse_x + (curr_x - self._last_mouse_x) * t
                interp_y = self._last_mouse_y + (curr_y - self._last_mouse_y) * t

                # 狂暴计算，不刷新UI
                last_result = self._fire_ray_and_process(interp_x, interp_y, "drag")

            # 更新记忆坐标
            self._last_mouse_x = curr_x
            self._last_mouse_y = curr_y
            
            # 🌟 2. 循环结束后，统一拿最后一个点的结果去刷新视口 1 次
            self._update_ui_cursor_and_refresh(last_result, drawMgr)

        else:
            # 距离太小 (比如只移动了 1 个像素) -> 触发“物理防抖”
            # 直接无视这次事件，不产生任何开销，拯救 GC！
            pass

    def doRelease(self, event, drawMgr, context):
        """松开阶段：结束行程"""
        self._isPressed = False
        self._last_mouse_x = None
        self._last_mouse_y = None
        
        if self.manager:
            self.manager.end_stroke()
            self.manager.clear_hit_state()

        self._cursor_pos = None
        self._refresh_viewport() # 离开时最后刷新一次

    # ==============================================================================
    # 🧠 核心计算分发 (纯数学版，无 UI 刷新开销)
    # ==============================================================================
    def _fire_ray_and_process(self, screen_x: float, screen_y: float, stroke_action: str):
        """只负责发射射线和运算底层权重，返回击中结果，绝对不触发视口重绘"""
        if not self.manager:
            return None

        # viewToWorld 严格要求 int 类型的像素坐标
        ix, iy = int(screen_x), int(screen_y)
        self._view.viewToWorld(ix, iy, self._ray_source, self._ray_direction)

        inv_matrix = self.mesh_dag_path.inclusiveMatrixInverse()
        ray_source_obj = self._ray_source * inv_matrix
        ray_dir_obj = self._ray_direction * inv_matrix

        # 直接呼叫 Manager 的一站式流水线！
        return self.manager.process_stroke(tuple(ray_source_obj)[0:3], tuple(ray_dir_obj), stroke_action)
    

    def _refresh_viewport(self):
        """触发 Shape 和 3D 视口刷新"""
        if self.preview_shape and not self.preview_shape._mObj.isNull():
            omr.MRenderer.setGeometryDrawDirty(self.preview_shape._mObj, True)
            pass
        if self._view:
            self._view.refresh(False, False)

    # ==============================================================================
    # 🎨 VP2 光标绘制 (画出跟随模型的笔刷圆圈)
    # ==============================================================================
    def _draw_brush_cursor(self, drawMgr):
        if not self.manager or not self._cursor_pos or not self._cursor_normal:
            return

        radius = self.manager.settings.radius
        color = self.brushColor if not self._isPressed else self.brushPresColor

        drawMgr.beginDrawable()
        drawMgr.setColor(color)
        drawMgr.setLineWidth(self.brushLine)
        # 沿着法线在世界坐标系画圆
        drawMgr.circle(self._cursor_pos, self._cursor_normal, radius)
        drawMgr.endDrawable()


# ==============================================================================
# 🔌 命令注册器
# ==============================================================================
class WeightBrushContextCmd(omui.MPxContextCommand):
    COMMAND_NAME = "cBrushCtx"

    def __init__(self):
        super(WeightBrushContextCmd, self).__init__()

    def makeObj(self):
        return WeightBrushContext()

    @staticmethod
    def creator():
        return WeightBrushContextCmd()
