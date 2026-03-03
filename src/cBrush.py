import typing
import maya.cmds as cmds
import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui
import maya.api.OpenMayaRender as omr

# 统一使用相对路径和模块导入
from . import cDisplayNode
from . import cBrushCore
from . import cRaycastCython

# 这是一个外部模块，我们保持原样导入
from m_utils.dag.getHistory import get_history

if typing.TYPE_CHECKING:
    pass

def maya_useNewAPI():
    pass

class WeightBrushContext(omui.MPxContext):
    brushLine = 2.0
    brushColor = om.MColor((1.0, 0.5, 0.0, 1.0))
    brushPresColor = om.MColor((1.0, 1, 1.0, 1.0))

    def __init__(self):
        super(WeightBrushContext, self).__init__()
        self.core: "cBrushCore.WeightBrushCore" = None
        self.preview_shape: "cDisplayNode.WeightPreviewShape" = None
        self._shape_path: str = None
        self.fn_mesh: om.MFnMesh = None
        self.mesh_dag_path: om.MDagPath = None
        self._view: omui.M3dView = None
        self._ray_source = om.MPoint()
        self._ray_direction = om.MVector()
        self._hit_result = None
        self._isPressed = False

    def toolOnSetup(self, event):
        try:
            mSel = om.MGlobal.getActiveSelectionList()
            shape_mDag = mSel.getDagPath(0).extendToShape()
            self._shape_path = shape_mDag.fullPathName()
            self.fn_mesh = om.MFnMesh(shape_mDag)
            self.mesh_dag_path = shape_mDag

            skin_node_name = get_history(self._shape_path, type="cSkinDeformer")[0]
            mSel_skin = om.MGlobal.getSelectionListByName(skin_node_name)
            mObj_skin = mSel_skin.getDependNode(0)
            cskin_fn = om.MFnDependencyNode(mObj_skin)
            output_geom_plug = cskin_fn.findPlug("outputGeom", False).elementByLogicalIndex(0)
            connected_plugs = output_geom_plug.connectedTo(False, True)

            for plug in connected_plugs:
                node = plug.node()
                if om.MFnDependencyNode(node).typeName == cDisplayNode.NODE_NAME:
                    self.preview_shape = om.MFnDependencyNode(node).userNode()
                    break

            if not self.preview_shape: raise RuntimeError("在网格上找不到连接的 WeightPreviewShape 节点。")

            self._view = omui.M3dView.active3dView()
            self.core = cBrushCore.WeightBrushCore(self.preview_shape, self.fn_mesh)
            print("[Brush] 引擎已启动。已连接到 WeightPreviewShape。")

        except Exception as e:
            cmds.evalDeferred(lambda: cmds.setToolTo("selectSuperContext"))
            om.MGlobal.displayError(f"笔刷初始化失败: {e}")
            raise

    def toolOffCleanup(self):
        if self.core: self.core.teardown()
        self.__init__()

    def doPress(self, event, drawMgr, context): self._shoot_ray_and_process(event, True, drawMgr)
    def doDrag(self, event, drawMgr, context): self._shoot_ray_and_process(event, True, drawMgr)
    def doPtrMoved(self, event, drawMgr, context): self._shoot_ray_and_process(event, False, drawMgr)
    def doRelease(self, event, drawMgr, context): self._isPressed = False

    def _raycast(self, ray_source_MPoint, ray_dir_MVector):
        if not self.preview_shape or not self.preview_shape.mesh_context: return None
        render_mesh = self.preview_shape.mesh_context
        source_arr, dir_arr = tuple(ray_source_MPoint)[0:3], tuple(ray_dir_MVector)
        
        hit_success, hit_pos_obj, hit_normal_obj, hit_tri, _, _, _ = cRaycastCython.raycast(
            source_arr, dir_arr, render_mesh.vertex_positions.view, render_mesh.triangle_indices.view)
        
        if hit_success:
            return hit_pos_obj, hit_normal_obj, hit_tri
        return None

    def _shoot_ray_and_process(self, event, is_pressed, drawMgr):
        self._isPressed = is_pressed
        last_hit = self._hit_result
        x, y = event.position
        self._view.viewToWorld(x, y, self._ray_source, self._ray_direction)
        if not self.preview_shape or not self.preview_shape.mesh_context: return

        inv_matrix = self.mesh_dag_path.inclusiveMatrixInverse()
        ray_source_obj, ray_dir_obj = self._ray_source * inv_matrix, self._ray_direction * inv_matrix
        
        self._hit_result = self._raycast(ray_source_obj, ray_dir_obj)
        
        if self._hit_result is None:
            if last_hit is not None and self.core: self.core.clear_hit_state(); self._refresh_viewport()
            return

        hit_point_obj, hit_normal_obj, hit_tri = self._hit_result
        
        self.core.hit_state.hit_center_normal = hit_normal_obj
        self.core.detect_range(hit_point_obj, hit_tri)
        if is_pressed: self.core.apply_weight_math()
        self._refresh_viewport()
        
        hit_point_world = om.MPoint(hit_point_obj) * self.mesh_dag_path.inclusiveMatrix()
        inv_transpose_matrix = self.mesh_dag_path.inclusiveMatrixInverse().transpose()
        hit_normal_world = om.MVector(hit_normal_obj) * inv_transpose_matrix

        self._draw_brush_cursor(drawMgr, hit_point_world, hit_normal_world)

    def _refresh_viewport(self):
        if self.preview_shape and not self.preview_shape.mObj.isNull(): omr.MRenderer.setGeometryDrawDirty(self.preview_shape.mObj, True)
        if self._view: self._view.refresh(False, False)

    def _draw_brush_cursor(self, drawMgr, hit_point, hit_normal):
        if not self.core: return
        color = self.brushColor if not self._isPressed else self.brushPresColor
        drawMgr.beginDrawable(); drawMgr.setColor(color); drawMgr.setLineWidth(self.brushLine)
        drawMgr.circle(om.MPoint(hit_point), om.MVector(hit_normal), self.core.settings.radius); drawMgr.endDrawable()

class WeightBrushContextCmd(omui.MPxContextCommand):
    COMMAND_NAME = "cBrushCtx"
    def __init__(self): super(WeightBrushContextCmd, self).__init__()
    def makeObj(self): return WeightBrushContext()
    @staticmethod
    def creator(): return WeightBrushContextCmd()
