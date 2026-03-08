import ctypes

import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui
import maya.api.OpenMayaRender as omr

from . import cBoundingBoxCython

from . import cColorCython as cColor
from .cMemoryView import CMemoryManager
from ._cRegistry import SkinRegistry
from .cSkinContext import BrushHitContext, MeshTopologyContext, RenderContext


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer


NODE_NAME = "WeightPreviewShape"
NODE_ID = om.MTypeId(0x80005)

DRAW_CLASSIFICATION = "drawdb/geometry/WeightPreview"
DRAW_REGISTRAR = "WeightPreviewShapeRegistrar"


# ==============================================================================
# 🎨 视口渲染器 (View)
# ==============================================================================
class WeightGeometryOverride(omr.MPxGeometryOverride):
    __slots__ = (
        "_mObject_shape",
        "_mFnDep_shape",
        "_class_shape",
        #
        "_topology_dirty",
        "_last_render_vtx_count",
        "_render_ctx",
        "_cpv_shader",
        "_wire_shader",
        "_point_shader",
    )
    RENDER_POINTS = True
    RENDER_LINE = False
    RENDER_POLYGONS = True

    points_size = 6.0
    lines_width = 1

    def __init__(self, mObjectShape):
        super(WeightGeometryOverride, self).__init__(mObjectShape)

        self._mObject_shape: om.MObject = mObjectShape
        self._mFnDep_shape: om.MFnDependencyNode = om.MFnDependencyNode(mObjectShape)
        self._class_shape: WeightPreviewShape = self._mFnDep_shape.userNode()

        # 拓扑快照 (仅拓扑改变时更新)
        self._topology_dirty: bool = True
        self._last_render_vtx_count = 0

        # 🎨 render context
        self._render_ctx = RenderContext()

        # 初始化着色器
        shader_mgr = omr.MRenderer.getShaderManager()
        self._cpv_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVSolidShader)

        self._wire_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVThickLineShader)
        self._wire_shader.setParameter("lineWidth", [WeightGeometryOverride.lines_width, WeightGeometryOverride.lines_width])

        self._point_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVFatPointShader)
        self._point_shader.setParameter("pointSize", [WeightGeometryOverride.points_size, WeightGeometryOverride.points_size])

    def updateDG(self):
        shape = self._class_shape
        cSkin = shape.cSkin

        if (not cSkin) or (not shape._deformMesh_plug.isConnected):
            return

        # 更新模型信息
        render_mesh = shape.update_mesh()

        # 拦截无效模型
        if (render_mesh.vertex_count == 0) or (render_mesh.triangle_indices is None):
            return

        # 直接比较顶点数。如果不等，说明拓扑变了，重置显存拷贝开关
        if self._last_render_vtx_count != render_mesh.vertex_count:
            self._last_render_vtx_count = render_mesh.vertex_count
            self._topology_dirty = True

        # 同步渲染配置(权重层,配色等)
        self._render_ctx = shape._update_render_context()

    def populateGeometry(self, requirements, renderItems, data):
        render_mesh = self._class_shape.mesh_context
        vtx_count = render_mesh.vertex_count
        vtx_pos = render_mesh.vertex_positions

        for req in requirements.vertexRequirements():
            if req.semantic == omr.MGeometry.kPosition:
                if vtx_pos and vtx_pos.ptr:
                    vtx_buf = data.createVertexBuffer(req)
                    vtx_addr = vtx_buf.acquire(vtx_count * 3, True)
                    if vtx_addr:
                        step = vtx_count * 12
                        ctypes.memmove(vtx_addr, vtx_pos.ptr, step)
                        ctypes.memmove(vtx_addr + step, vtx_pos.ptr, step)
                        ctypes.memmove(vtx_addr + step * 2, vtx_pos.ptr, step)
                        vtx_buf.commit(vtx_addr)

            elif req.semantic == omr.MGeometry.kColor:
                color_buf = data.createVertexBuffer(req)
                color_addr = color_buf.acquire(vtx_count * 3, True)
                if color_addr:
                    color_view = CMemoryManager.from_ptr(color_addr, "f", (vtx_count * 3, 4)).view
                    weights = self._class_shape.active_paint_weights

                    if weights is not None:
                        if self._render_ctx.paintMask:
                            cColor.render_gradient(weights, color_view[0:vtx_count], self._render_ctx.color_mask_remapA, self._render_ctx.color_mask_remapB)
                        elif self._render_ctx.render_mode == 1:
                            cColor.render_gradient(weights, color_view[0:vtx_count], self._render_ctx.color_weights_remapA, self._render_ctx.color_weights_remapB)
                        else:
                            cColor.render_heatmap(weights, color_view[0:vtx_count])
                    else:
                        cColor.render_fill(color_view[0:vtx_count], (0.0, 0.0, 1.0, 1.0))

                    cColor.render_fill(color_view[vtx_count : 2 * vtx_count], self._render_ctx.color_wire)

                    brush_ctx = self._class_shape.brush_context
                    if brush_ctx.is_valid:
                        cColor.render_brush_gradient(
                            color_view[2 * vtx_count : 3 * vtx_count],
                            brush_ctx.hit_indices.view,
                            brush_ctx.hit_weights.view,
                            brush_ctx.hit_count,
                            self._render_ctx.color_brush_remapA,
                            self._render_ctx.color_brush_remapB,
                        )

                    color_buf.commit(color_addr)

        for item in renderItems:
            item_name = item.name()

            if (item_name == "WeightSolidItem") and (render_mesh.triangle_indices):
                if self._topology_dirty:
                    num_indices = render_mesh.triangle_indices.view.nbytes // 4
                    i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32)
                    i_addr = i_buf.acquire(num_indices, True)
                    if i_addr:
                        ctypes.memmove(i_addr, render_mesh.triangle_indices.ptr, num_indices * 4)
                        i_buf.commit(i_addr)
                        item.associateWithIndexBuffer(i_buf)

            elif (item_name == "WeightWireItem") and (render_mesh.edge_indices):
                if self._topology_dirty:
                    num_indices = render_mesh.edge_indices.view.nbytes // 4
                    i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32)
                    i_addr = i_buf.acquire(num_indices, True)
                    if i_addr:
                        cColor.offset_indices_direct(render_mesh.edge_indices.ptr, int(i_addr), num_indices, vtx_count)
                        i_buf.commit(i_addr)
                        item.associateWithIndexBuffer(i_buf)

            elif item_name == "BrushDebugPoints":
                brush_ctx = self._class_shape.brush_context
                if brush_ctx.is_valid:
                    i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32)
                    i_addr = i_buf.acquire(brush_ctx.hit_count, True)
                    if i_addr:
                        cColor.offset_indices_direct(brush_ctx.hit_indices.ptr, int(i_addr), brush_ctx.hit_count, 2 * vtx_count)
                        i_buf.commit(i_addr)
                        item.associateWithIndexBuffer(i_buf)

        self._topology_dirty = False

    def _setup_render_item(self, renderItems, name, geom_type, shader, depth_priority=None):
        idx = renderItems.indexOf(name)
        if idx < 0:
            item = omr.MRenderItem.create(name, omr.MRenderItem.MaterialSceneItem, geom_type)
            renderItems.append(item)
        else:
            item = renderItems[idx]

        item.setDrawMode(omr.MGeometry.kAll)
        item.setShader(shader)
        if depth_priority is not None:
            # 兼容不同版本 API 的属性/方法差异
            try:
                item.depthPriority = depth_priority
            except AttributeError:
                item.setDepthPriority(depth_priority)
        item.enable(True)

    def updateRenderItems(self, objPath, renderItems):
        if WeightGeometryOverride.RENDER_POLYGONS:
            self._setup_render_item(renderItems, "WeightSolidItem", omr.MGeometry.kTriangles, self._cpv_shader)

        if WeightGeometryOverride.RENDER_LINE:
            self._setup_render_item(renderItems, "WeightWireItem", omr.MGeometry.kLines, self._wire_shader, omr.MRenderItem.sActiveWireDepthPriority)

        if WeightGeometryOverride.RENDER_POINTS:
            self._setup_render_item(renderItems, "BrushDebugPoints", omr.MGeometry.kPoints, self._point_shader, omr.MRenderItem.sActivePointDepthPriority)
            idx = renderItems.indexOf("BrushDebugPoints")
            if idx >= 0:
                item = renderItems[idx]
                brush_ctx = self._class_shape.brush_context if self._class_shape else None
                item.enable(brush_ctx is not None and brush_ctx.is_valid)

    def cleanUp(self):
        pass

    @staticmethod
    def creator(obj):
        return WeightGeometryOverride(obj)

    def supportedDrawAPIs(self):
        return omr.MRenderer.kAllDevices


# ==============================================================================
# 🎛️ 自定义 Shape 节点注册: 监听连接，管理网格上下文与显示配置
# ==============================================================================
class WeightPreviewShape(om.MPxSurfaceShape):
    aLayer = None
    aInfluence = None
    aMask = None
    aInDeformMesh = None
    # 🎨渲染与色彩属性句柄
    aRenderMode = None
    aColorWire = None
    aColorPoint = None
    aColorMaskRemapA = None
    aColorMaskRemapB = None
    aColorWeightsRemapA = None
    aColorWeightsRemapB = None
    aColorBrushRemapA = None
    aColorBrushRemapB = None

    __slots__ = (
        "mesh_context",
        "brush_context",
        "render_context",
        "_boundingBox",
        "_last_cSkin",
        "_mObj",
        "_layer_plug",
        "_mask_plug",
        "_influence_plug",
        "_deformMesh_plug",
        "_renderMode_plug",
        "_colorWire_plug",
        "_colorPoint_plug",
        "_colorMaskRemapA_plug",
        "_colorMaskRemapB_plug",
        "_colorWeightsRemapA_plug",
        "_colorWeightsRemapB_plug",
        "_colorBrushRemapA_plug",
        "_colorBrushRemapB_plug",
    )

    def __init__(self):
        super(WeightPreviewShape, self).__init__()
        self._boundingBox = om.MBoundingBox(om.MPoint((-10, -10, -10)), om.MPoint((10, 10, 10)))

        # 💥 实例缓存池：再也不用每次去注册表捞了
        self._last_cSkin = None

        # 初始化上下文 (代替以前的全局 DATA)
        self.mesh_context = MeshTopologyContext()
        self.brush_context = BrushHitContext()

        # 占位：如果有需要在UI直接访问的笔刷设置，可以在这里实例化
        # 🎨 渲染显示数据
        self.render_context = RenderContext()

    def postConstructor(self):
        self._mObj = self.thisMObject()
        # 预先获取所有需要用到的 Plug
        self._layer_plug = om.MPlug(self._mObj, self.aLayer)
        self._mask_plug = om.MPlug(self._mObj, self.aMask)
        self._influence_plug = om.MPlug(self._mObj, self.aInfluence)
        self._deformMesh_plug = om.MPlug(self._mObj, self.aInDeformMesh)
        self._renderMode_plug = om.MPlug(self._mObj, self.aRenderMode)
        self._colorWire_plug = om.MPlug(self._mObj, self.aColorWire)
        self._colorPoint_plug = om.MPlug(self._mObj, self.aColorPoint)
        self._colorMaskRemapA_plug = om.MPlug(self._mObj, self.aColorMaskRemapA)
        self._colorMaskRemapB_plug = om.MPlug(self._mObj, self.aColorMaskRemapB)
        self._colorWeightsRemapA_plug = om.MPlug(self._mObj, self.aColorWeightsRemapA)
        self._colorWeightsRemapB_plug = om.MPlug(self._mObj, self.aColorWeightsRemapB)
        self._colorBrushRemapA_plug = om.MPlug(self._mObj, self.aColorBrushRemapA)
        self._colorBrushRemapB_plug = om.MPlug(self._mObj, self.aColorBrushRemapB)
        super().postConstructor()

    @property
    def cSkin(self) -> "CythonSkinDeformer":
        """获取绑定的 cSkin 实例"""
        if self._last_cSkin is None:
            if not self._deformMesh_plug.isConnected:
                return None
            connected_plugs = self._deformMesh_plug.connectedTo(True, False)
            if not connected_plugs:
                return None
            mObj_skin = connected_plugs[0].node()
            self._last_cSkin = SkinRegistry.get_instance_by_api2(mObj_skin)
        return self._last_cSkin

    @property
    def active_paint_weights(self) -> tuple["CMemoryManager", int, bool] | tuple[None, None, None]:
        """
        直接提取当前需要渲染的 1D 权重视图 (Mask 或指定骨骼权重)。
        """
        cSkin = self.cSkin
        ctx = self.render_context

        # 1. 前置拦截：无数据或层级未激活
        if not cSkin or ctx.paintLayerIndex not in cSkin.weightsLayer:
            return None

        layer = cSkin.weightsLayer[ctx.paintLayerIndex]

        # 2. 遮罩模式 (Mask)
        if ctx.paintMask:
            handle = layer.maskHandle
            if handle and handle.is_valid:
                # Mask 本身就是单通道 1D 数据
                return handle.memory.view.cast("B").cast("f")
            return None

        # 3. 权重模式 (Weights)
        handle = layer.weightsHandle
        if handle and handle.is_valid:
            cols = cSkin.influences_count
            safe_idx = max(0, min(ctx.paintInfluenceIndex, cols - 1))
            # 直接利用 Python 内存视图的步长切片提取 1D 数据！
            return handle.memory.view.cast("B").cast("f")[safe_idx::cols]

        return None

    def update_mesh(self):
        """
        零开销直接从 cSkin 中“白嫖”顶点的物理指针。不再经过 API 调用。
        Update:
            mesh_context.vertex_positions
            mesh_context.vertex_count
            mesh_context.edge_indices
            mesh_context.triangle_indices
        """
        # 强制 DG 计算，确保上游的 deform 运行过了
        self._deformMesh_plug.asMObject()

        cSkin = self.cSkin
        if not cSkin:
            return

        vtx_count = cSkin.vertex_count
        if not vtx_count:
            return

        # 如果顶点数变了，触发拓扑缓存重建
        if self.mesh_context.vertex_count != vtx_count:
            # 拓扑不会每帧变，用 om 重新捞一遍边和邻接表
            mFnMesh = om.MFnMesh(self._deformMesh_plug.asMObject())
            self._update_topology(mFnMesh)

        # 跨域读取！
        self.mesh_context.vertex_positions = cSkin.rawPoints_output_mgr
        return self.mesh_context

    def _update_topology(self, mFnMesh: om.MFnMesh):
        """
        [补全] 拓扑数据生成器
        使用 mFnMesh (OpenMaya) 生成三角形和边索引数据用于渲染。
        Update:
            mesh_context.vertex_count
            mesh_context.edge_indices
            mesh_context.triangle_indices
        """
        vtx_count = mFnMesh.numVertices
        self.mesh_context.vertex_count = vtx_count

        # 提取三角形索引
        _, tri_vtx_indices = mFnMesh.getTriangles()
        self.mesh_context.triangle_indices = CMemoryManager.from_list(list(tri_vtx_indices), "i")

        # 提取边索引
        num_edges = mFnMesh.numEdges
        edge_indices = [0] * (num_edges * 2)
        for i in range(num_edges):
            p1, p2 = mFnMesh.getEdgeVertices(i)
            edge_indices[i * 2] = p1
            edge_indices[i * 2 + 1] = p2
        self.mesh_context.edge_indices = CMemoryManager.from_list(edge_indices, "i")

    def _update_render_context(self):
        """将前端 Plug 属性同步到 Shape 的本地状态中"""

        def get_color(plug: om.MPlug, alpha=1.0):
            return (
                plug.child(0).asFloat(),
                plug.child(1).asFloat(),
                plug.child(2).asFloat(),
                alpha,
            )

        # fmt:off
        self.render_context = RenderContext(
            paintLayerIndex      = self._layer_plug.asInt(),
            paintInfluenceIndex  = self._influence_plug.asInt(),
            paintMask            = self._mask_plug.asBool(),
            render_mode          = self._renderMode_plug.asInt(),
            color_wire           = get_color(self._colorWire_plug, 1.0),
            color_point          = get_color(self._colorPoint_plug, 1.0),
            color_mask_remapA    = get_color(self._colorMaskRemapA_plug, 0.0),
            color_mask_remapB    = get_color(self._colorMaskRemapB_plug, 0.0),
            color_weights_remapA = get_color(self._colorWeightsRemapA_plug, 0.0),
            color_weights_remapB = get_color(self._colorWeightsRemapB_plug, 0.0),
            color_brush_remapA   = get_color(self._colorBrushRemapA_plug, 1.0),
            color_brush_remapB   = get_color(self._colorBrushRemapB_plug, 1.0),
        )
        # fmt:on
        return self.render_context

    def connectionBroken(self, plug, otherPlug, asSrc):
        if plug == self._deformMesh_plug:
            self._last_cSkin = None
        return super().connectionBroken(plug, otherPlug, asSrc)

    def setDependentsDirty(self, plug, plugArray):
        attr = plug.attribute()
        dirty_attrs = (
            self.aInDeformMesh,
            self.aLayer,
            self.aMask,
            self.aInfluence,
            self.aRenderMode,
            self.aColorWire,
            self.aColorPoint,
            self.aColorMaskRemapA,
            self.aColorMaskRemapB,
            self.aColorWeightsRemapA,
            self.aColorWeightsRemapB,
            self.aColorBrushRemapA,
            self.aColorBrushRemapB,
        )
        if attr in dirty_attrs:
            omr.MRenderer.setGeometryDrawDirty(self._mObj, True)
        return super().setDependentsDirty(plug, plugArray)

    def preEvaluation(self, context, evaluationNode):

        dirty_attrs = (self.aInDeformMesh, self.aLayer, self.aMask, self.aInfluence, self.aRenderMode, self.aColorWire, self.aColorPoint, self.aColorMaskRemapA, self.aColorMaskRemapB, self.aColorWeightsRemapA, self.aColorWeightsRemapB, self.aColorBrushRemapA, self.aColorBrushRemapB)
        if any(evaluationNode.dirtyPlugExists(a) for a in dirty_attrs):
            omr.MRenderer.setGeometryDrawDirty(self._mObj, True)

        super().preEvaluation(context, evaluationNode)

    @staticmethod
    def initialize():
        nAttr = om.MFnNumericAttribute()
        tAttr = om.MFnTypedAttribute()
        eAttr = om.MFnEnumAttribute()

        WeightPreviewShape.aInDeformMesh = tAttr.create("inDeformMesh", "idm", om.MFnData.kMesh)
        tAttr.hidden = True
        tAttr.storable = False
        WeightPreviewShape.addAttribute(WeightPreviewShape.aInDeformMesh)

        WeightPreviewShape.aLayer = nAttr.create("layer", "lyr", om.MFnNumericData.kInt, 0)
        nAttr.storable = True
        nAttr.channelBox = True
        WeightPreviewShape.addAttribute(WeightPreviewShape.aLayer)

        WeightPreviewShape.aMask = nAttr.create("mask", "msk", om.MFnNumericData.kBoolean, False)
        nAttr.storable = True
        nAttr.channelBox = True
        WeightPreviewShape.addAttribute(WeightPreviewShape.aMask)

        WeightPreviewShape.aInfluence = nAttr.create("influence", "ifn", om.MFnNumericData.kInt, 0)
        nAttr.storable = True
        nAttr.channelBox = True
        WeightPreviewShape.addAttribute(WeightPreviewShape.aInfluence)

        WeightPreviewShape.aRenderMode = eAttr.create("renderMode", "rm", 0)
        eAttr.addField("Heatmap", 0)
        eAttr.addField("Alpha", 1)
        eAttr.storable = True
        WeightPreviewShape.addAttribute(WeightPreviewShape.aRenderMode)

        def add_color(long_name, short_name, default_rgb):
            attr = nAttr.createColor(long_name, short_name)
            nAttr.default = default_rgb
            nAttr.storable = True
            WeightPreviewShape.addAttribute(attr)
            return attr

        # fmt:off
        WeightPreviewShape.aColorWire          = add_color("colorWire"        , "cwir", (0.0, 1.0, 1.0)) 
        WeightPreviewShape.aColorPoint         = add_color("colorPoint"       , "cpnt", (1.0, 0.0, 0.0))
        WeightPreviewShape.aColorMaskRemapA    = add_color("colorMaskRemapA"  , "cmra", (0.1, 0.1, 0.1))
        WeightPreviewShape.aColorMaskRemapB    = add_color("colorMaskRemapB"  , "cmrb", (0.1, 1.0, 0.1))
        WeightPreviewShape.aColorWeightsRemapA = add_color("colorWeightsRemapA", "cwra", (0.0, 0.0, 0.0))
        WeightPreviewShape.aColorWeightsRemapB = add_color("colorWeightsRemapB", "cwrb", (1.0, 1.0, 1.0))
        WeightPreviewShape.aColorBrushRemapA   = add_color("colorBrushRemapA"  , "cbra", (1.0, 0.0, 0.0))
        WeightPreviewShape.aColorBrushRemapB   = add_color("colorBrushRemapB"  , "cbrb", (1.0, 1.0, 0.0))
        # fmt:on

    def isBounded(self):
        return True

    def boundingBox(self):
        render_mesh = self.mesh_context
        if render_mesh and render_mesh.vertex_positions:
            boxMin, boxMax = cBoundingBoxCython.compute_bbox_fast(render_mesh.vertex_positions.view, render_mesh.vertex_count)
            self._boundingBox = om.MBoundingBox(om.MPoint(boxMin), om.MPoint(boxMax))
        return self._boundingBox

    @staticmethod
    def creator():
        return WeightPreviewShape()


class WeightPreviewShapeUI(omui.MPxSurfaceShapeUI):
    def __init__(self):
        super(WeightPreviewShapeUI, self).__init__()

    @staticmethod
    def creator():
        return WeightPreviewShapeUI()
