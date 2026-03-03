import ctypes

import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui
import maya.api.OpenMayaRender as omr

from . import cBoundingBoxCython
from .cMemoryView import CMemoryManager
from ._cRegistry import SkinRegistry
from . import cColorCython as cColor
from . import _profile

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer


NODE_NAME = "WeightPreviewShape"
NODE_ID = om.MTypeId(0x80005)

DRAW_CLASSIFICATION = "drawdb/geometry/WeightPreview"
DRAW_REGISTRAR = "WeightPreviewShapeRegistrar"


# ==============================================================================
# 📦 数据结构：Shape 节点专属的网格与笔刷上下文
# ==============================================================================
# fmt:off
class RenderableMesh:
    """统一的数据结构，包含渲染所需的所有网格数据，包括动态的顶点位置和静态的拓扑。"""
    __slots__ = (
        "vertex_count",
        "vertex_positions",
        "vertex_positions_2d_view",
        "triangle_indices",
        "edge_indices",
    )

    def __init__(self):
        self.vertex_count: int = 0
        self.vertex_positions: CMemoryManager = None
        self.vertex_positions_2d_view: CMemoryManager = None
        self.triangle_indices: CMemoryManager = None
        self.edge_indices: CMemoryManager = None


class BrushDisplayContext:
    """存放笔刷在当前 Shape 上的运行时状态"""

    __slots__ = (
        "brush_hit_count",
        "brush_hit_indices",
        "brush_hit_weights",
        "brush_epoch",
    )

    def __init__(self):
        self.brush_hit_count  : int            = 0
        self.brush_hit_indices: CMemoryManager = None
        self.brush_hit_weights: CMemoryManager = None
        self.brush_epoch      : int            = 1


class RenderState:
    """存放 UI 颜色、渲染模式等显示配置的快照"""
    __slots__ = (
        "render_mode",
        "paintLayerIndex",
        "paintInfluenceIndex",
        "paintMask",
        "color_wire",
        "color_point",
        "color_mask_remapA",
        "color_mask_remapB",
        "color_weights_remapA",
        "color_weights_remapB",
        "color_brush_remapA",
        "color_brush_remapB",
    )

    def __init__(
        self,
        render_mode         : int   = 0,
        paintLayerIndex     : int   = -1,
        paintInfluenceIndex : int   = 0,
        paintMask           : bool  = False,
        color_wire          : tuple = (0.0, 1.0, 1.0, 1.0),
        color_point         : tuple = (1.0, 0.0, 0.0, 1.0),
        color_mask_remapA   : tuple = (0.1, 0.1, 0.1, 0.0),
        color_mask_remapB   : tuple = (0.1, 1.0, 0.1, 0.0),
        color_weights_remapA: tuple = (0.0, 0.0, 0.0, 0.0),
        color_weights_remapB: tuple = (1.0, 1.0, 1.0, 0.0),
        color_brush_remapA  : tuple = (1.0, 0.0, 0.0, 1.0),
        color_brush_remapB  : tuple = (1.0, 1.0, 0.0, 1.0),
    ):
        self.render_mode          = render_mode
        self.paintLayerIndex      = paintLayerIndex
        self.paintInfluenceIndex  = paintInfluenceIndex
        self.paintMask            = paintMask
        self.color_wire           = color_wire
        self.color_point          = color_point
        self.color_mask_remapA    = color_mask_remapA
        self.color_mask_remapB    = color_mask_remapB
        self.color_weights_remapA = color_weights_remapA
        self.color_weights_remapB = color_weights_remapB
        self.color_brush_remapA   = color_brush_remapA
        self.color_brush_remapB   = color_brush_remapB
    # fmt:on


# ==============================================================================
# 🎨 视口渲染器 (View)
# ==============================================================================
class WeightGeometryOverride(omr.MPxGeometryOverride):
    __slots__ = (
        "_mObject_shape",
        "_mFnDep_shape",
        "_shape_class",
        "_cached_vertex_count",
        "_cached_solid_mgr",
        "_cached_wire_mgr",
        "_cached_point_mgr",
        "_indices_initialized",
        "_last_topo_cache",
        "_cached_vertex_positions_mgr",
        "_cached_weights_1d",
        "_cached_brush_hit_count",
        "_cached_brush_hit_indices",
        "_cached_brush_hit_weights",
        "_state",
        "_cpv_shader",
        "_wire_shader",
        "_point_shader",
    )
    RENDER_POINTS = True
    RENDER_LINE = True
    RENDER_POLYGONS = True

    points_size = 1.0
    lines_width = 1.0

    def __init__(self, mObjectShape):
        super(WeightGeometryOverride, self).__init__(mObjectShape)

        self._mObject_shape: om.MObject = mObjectShape
        self._mFnDep_shape: om.MFnDependencyNode = om.MFnDependencyNode(mObjectShape)
        self._shape_class: WeightPreviewShape = self._mFnDep_shape.userNode()

        # 拓扑快照 (仅拓扑改变时更新)
        self._cached_vertex_count = 0
        self._cached_solid_mgr: CMemoryManager = None
        self._cached_wire_mgr: CMemoryManager = None
        self._cached_point_mgr: CMemoryManager = None
        self._indices_initialized: bool = False
        self._last_topo_cache = None

        # 💥 渲染负载快照 (Render Payload) - 每一帧都会硬性刷新
        self._cached_vertex_positions_mgr = None
        self._cached_weights_1d = None
        self._cached_brush_hit_count = 0
        self._cached_brush_hit_indices = None
        self._cached_brush_hit_weights = None

        # 🎨 状态系统：取代散乱的 _cached_xxx 声明
        self._state = RenderState()

        # 初始化着色器
        shader_mgr = omr.MRenderer.getShaderManager()
        self._cpv_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVSolidShader)

        self._wire_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVThickLineShader)
        self._wire_shader.setParameter("lineWidth", [WeightGeometryOverride.lines_width, WeightGeometryOverride.lines_width])

        self._point_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVFatPointShader)
        self._point_shader.setParameter("pointSize", [WeightGeometryOverride.points_size, WeightGeometryOverride.points_size])

    def updateDG(self):
        with _profile.MicroProfiler(target_runs=100, enable=False) as prof:
            shape = self._shape_class
            cSkin = shape.cSkin

            if not cSkin or not shape._deformMesh_plug.isConnected:
                return

            prof.step("updateDG:---------预处理与解算")

            # 1. 强制拉取上游的网格解算结果
            shape.update_mesh_points()

            render_mesh = shape.render_mesh
            if not render_mesh.vertex_positions:
                return

            prof.step("updateDG:---------更新模型结构")
            # 2. 同步 UI 状态到 Shape 内部
            shape.sync_ui_state_to_blackboard()

            _cache = self._get_topology_index_buffers(render_mesh)
            if _cache:
                (
                    self._cached_solid_mgr,
                    self._cached_wire_mgr,
                    self._cached_point_mgr,
                    self._cached_vertex_count,
                ) = _cache
                if self._last_topo_cache is not _cache:
                    self._indices_initialized = False
                    self._last_topo_cache = _cache

            self._cached_vertex_positions_mgr = render_mesh.vertex_positions

            # 1. 直接获取 Shape 侧同步好的状态包
            self._state = shape.render_state

            # 2. 获取权重数据
            weights2D_mgr, target_idx, is_mask = shape.active_paint_target
            self._cached_weights_1d = None
            if weights2D_mgr is not None and weights2D_mgr.view is not None:
                mv_2d = weights2D_mgr.view
                cols = mv_2d.shape[1] if len(mv_2d.shape) > 1 else 1
                safe_idx = max(0, min(target_idx, cols - 1))
                mv_1d_flat = mv_2d.cast("B").cast("f")
                self._cached_weights_1d = mv_1d_flat[safe_idx::cols]

            # Brush DATA
            brush_ctx = shape.brush_context
            self._cached_brush_hit_count = brush_ctx.brush_hit_count
            self._cached_brush_hit_indices = brush_ctx.brush_hit_indices
            self._cached_brush_hit_weights = brush_ctx.brush_hit_weights

            prof.step("updateDG:---------准备数据结束")

    def populateGeometry(self, requirements, renderItems, data):
        N = self._cached_vertex_count
        points_mgr = self._cached_vertex_positions_mgr

        for req in requirements.vertexRequirements():
            if req.semantic == omr.MGeometry.kPosition:
                if points_mgr and points_mgr.ptr:
                    vtx_buf = data.createVertexBuffer(req)
                    vtx_addr = vtx_buf.acquire(N * 3, True)
                    if vtx_addr:
                        stride = N * 12
                        ctypes.memmove(vtx_addr, points_mgr.ptr, stride)
                        ctypes.memmove(vtx_addr + stride, points_mgr.ptr, stride)
                        ctypes.memmove(vtx_addr + stride * 2, points_mgr.ptr, stride)
                        vtx_buf.commit(vtx_addr)

            elif req.semantic == omr.MGeometry.kColor:
                color_buf = data.createVertexBuffer(req)
                color_addr = color_buf.acquire(N * 3, True)
                if color_addr:
                    color_view = CMemoryManager.from_ptr(color_addr, "f", (N * 3, 4)).view

                    if self._cached_weights_1d is not None:
                        if self._state.paintMask:
                            cColor.render_gradient(self._cached_weights_1d, color_view[0:N], self._state.color_mask_remapA, self._state.color_mask_remapB)
                        elif self._state.render_mode == 1:
                            cColor.render_gradient(self._cached_weights_1d, color_view[0:N], self._state.color_weights_remapA, self._state.color_weights_remapB)
                        else:
                            cColor.render_heatmap(self._cached_weights_1d, color_view[0:N])
                    else:
                        cColor.render_fill(color_view[0:N], (0.0, 0.0, 1.0, 1.0))

                    cColor.render_fill(color_view[N : 2 * N], self._state.color_wire)

                    if self._cached_brush_hit_count > 0 and self._cached_brush_hit_indices and self._cached_brush_hit_weights:
                        cColor.render_brush_gradient(
                            color_view[2 * N : 3 * N],
                            self._cached_brush_hit_indices.view,
                            self._cached_brush_hit_weights.view,
                            self._cached_brush_hit_count,
                            self._state.color_brush_remapA,
                            self._state.color_brush_remapB,
                        )
                    color_buf.commit(color_addr)

        for item in renderItems:
            item_name = item.name()

            if item_name == "WeightSolidItem" and self._cached_solid_mgr:
                if not self._indices_initialized:
                    mgr = self._cached_solid_mgr
                    num_indices = mgr.view.nbytes // 4
                    i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32)
                    i_addr = i_buf.acquire(num_indices, True)
                    if i_addr:
                        ctypes.memmove(i_addr, mgr.ptr, num_indices * 4)
                        i_buf.commit(i_addr)
                        item.associateWithIndexBuffer(i_buf)

            elif item_name == "WeightWireItem" and self._cached_wire_mgr:
                if not self._indices_initialized:
                    mgr = self._cached_wire_mgr
                    num_indices = mgr.view.nbytes // 4
                    i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32)
                    i_addr = i_buf.acquire(num_indices, True)
                    if i_addr:
                        cColor.offset_indices_direct(mgr.ptr, int(i_addr), num_indices, N)
                        i_buf.commit(i_addr)
                        item.associateWithIndexBuffer(i_buf)

            elif item_name == "BrushDebugPoints":
                if self._cached_brush_hit_count > 0 and self._cached_brush_hit_indices:
                    i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32)
                    i_addr = i_buf.acquire(self._cached_brush_hit_count, True)
                    if i_addr:
                        cColor.offset_indices_direct(
                            self._cached_brush_hit_indices.ptr,
                            int(i_addr),
                            self._cached_brush_hit_count,
                            2 * N,
                        )
                        i_buf.commit(i_addr)
                        item.associateWithIndexBuffer(i_buf)

        self._indices_initialized = True

    def _get_topology_index_buffers(self, render_mesh: RenderableMesh):
        N = render_mesh.vertex_count

        if N == 0 or render_mesh.triangle_indices is None:
            return None

        if (self._cached_vertex_count == N) and (self._cached_solid_mgr is not None):
            return (
                self._cached_solid_mgr,
                self._cached_wire_mgr,
                self._cached_point_mgr,
                self._cached_vertex_count,
            )

        new_solid_mgr = render_mesh.triangle_indices
        new_wire_mgr = render_mesh.edge_indices
        new_point_mgr = CMemoryManager.from_list(list(range(N)), "i")

        return new_solid_mgr, new_wire_mgr, new_point_mgr, N

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
                brush_ctx = self._shape_class.brush_context if self._shape_class else None
                item.enable(brush_ctx is not None and brush_ctx.brush_hit_count > 0)

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
        "render_mesh",
        "brush_context",
        "render_state",
        "_boundingBox",
        "_cached_cSkin",
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
        self._cached_cSkin = None

        # 初始化上下文 (代替以前的全局 DATA)
        self.render_mesh = RenderableMesh()
        self.brush_context = BrushDisplayContext()

        # 占位：如果有需要在UI直接访问的笔刷设置，可以在这里实例化
        # 🎨 渲染显示数据
        self.render_state = RenderState()

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
        if self._cached_cSkin is None:
            if not self._deformMesh_plug.isConnected:
                return None
            connected_plugs = self._deformMesh_plug.connectedTo(True, False)
            if not connected_plugs:
                return None
            mObj_skin = connected_plugs[0].node()
            self._cached_cSkin = SkinRegistry.get_instance_by_api2(mObj_skin)
        return self._cached_cSkin

    @property
    def active_paint_target(self) -> tuple["CMemoryManager", int, bool] | tuple[None, None, None]:
        """
        💥 从扁平化的 cSkin 中捞出当前需要渲染/绘制的层级物理内存。
        """
        cSkin = self.cSkin
        state = self.render_state
        if not cSkin or (state.paintLayerIndex not in cSkin.weightsLayer):
            return None, None, None

        active_layer = cSkin.weightsLayer[state.paintLayerIndex]

        if state.paintMask:
            if not active_layer.maskHandle or not active_layer.maskHandle.is_valid:
                return None, None, None
            return active_layer.maskHandle.memory.reshape((self.render_mesh.vertex_count, 1)), 0, True
        else:
            if not active_layer.weightsHandle or not active_layer.weightsHandle.is_valid:
                return None, None, None
            return active_layer.weightsHandle.memory.reshape((self.render_mesh.vertex_count, cSkin.influences_count)), state.paintInfluenceIndex, False

    def update_mesh_points(self):
        """
        零开销直接从 cSkin 中“白嫖”顶点的物理指针。不再经过 API 调用。
        """
        # 强制 DG 计算，确保上游的 deform 运行过了
        _ = self._deformMesh_plug.asMObject()

        cSkin = self.cSkin
        if not cSkin:
            return

        vtx_count = cSkin.vertex_count
        if not vtx_count:
            return

        # 如果顶点数变了，触发拓扑缓存重建
        if self.render_mesh.vertex_count != vtx_count:
            # 拓扑不会每帧变，用 om 重新捞一遍边和邻接表
            mFnMesh = om.MFnMesh(self._deformMesh_plug.asMObject())
            self._build_renderable_mesh_topology(mFnMesh)

        # 写入 Mesh Context
        self.render_mesh.vertex_count = vtx_count

        # 💥 跨域读取！
        self.render_mesh.vertex_positions = cSkin.rawPoints_output_mgr

        # 将原始一维指针映射为 N*3 的二维视图供笔刷用
        if self.render_mesh.vertex_positions:
            ptr = self.render_mesh.vertex_positions.ptr
            self.render_mesh.vertex_positions_2d_view = CMemoryManager.from_ptr(ptr, "f", (vtx_count, 3))

    def _build_renderable_mesh_topology(self, mFnMesh: om.MFnMesh):
        """
        [补全] 拓扑数据生成器
        使用 mFnMesh (OpenMaya) 生成三角形和边索引数据用于渲染。
        """
        render_mesh = self.render_mesh
        vtx_count = mFnMesh.numVertices
        render_mesh.vertex_count = vtx_count

        # 提取三角形索引
        _, tri_vtx_indices = mFnMesh.getTriangles()
        render_mesh.triangle_indices = CMemoryManager.from_list(list(tri_vtx_indices), "i")

        # 提取边索引
        num_edges = mFnMesh.numEdges
        edge_indices = [0] * (num_edges * 2)
        for i in range(num_edges):
            p1, p2 = mFnMesh.getEdgeVertices(i)
            edge_indices[i * 2] = p1
            edge_indices[i * 2 + 1] = p2
        render_mesh.edge_indices = CMemoryManager.from_list(edge_indices, "i")

    def sync_ui_state_to_blackboard(self):
        """将前端 Plug 属性同步到 Shape 的本地状态中"""

        def get_color(plug: om.MPlug, alpha=1.0):
            return (
                plug.child(0).asFloat(),
                plug.child(1).asFloat(),
                plug.child(2).asFloat(),
                alpha,
            )

        # fmt:off
        self.render_state = RenderState(
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


    def connectionBroken(self, plug, otherPlug, asSrc):
        if plug == self._deformMesh_plug:
            self._cached_cSkin = None
        return super().connectionBroken(plug, otherPlug, asSrc)

    def setDependentsDirty(self, plug, plugArray):
        attr = plug.attribute()
        dirty_attrs = (
            self.aInDeformMesh, self.aLayer, self.aMask, self.aInfluence, self.aRenderMode,
            self.aColorWire, self.aColorPoint, self.aColorMaskRemapA, self.aColorMaskRemapB,
            self.aColorWeightsRemapA, self.aColorWeightsRemapB, 
            self.aColorBrushRemapA, self.aColorBrushRemapB,
        )
        if attr in dirty_attrs:
            omr.MRenderer.setGeometryDrawDirty(self._mObj, True)
        return super().setDependentsDirty(plug, plugArray)

    def preEvaluation(self, context, evaluationNode):

        dirty_attrs = (
            self.aInDeformMesh, self.aLayer, self.aMask, self.aInfluence, self.aRenderMode,
            self.aColorWire, self.aColorPoint, self.aColorMaskRemapA, self.aColorMaskRemapB,
            self.aColorWeightsRemapA, self.aColorWeightsRemapB, 
            self.aColorBrushRemapA, self.aColorBrushRemapB
        )
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
        render_mesh = self.render_mesh
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
