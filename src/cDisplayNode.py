
import ctypes
from dataclasses import dataclass

import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui
import maya.api.OpenMayaRender as omr

from . import cBoundingBoxCython
from .cMemoryView import CMemoryManager
from ._cRegistry import SkinRegistry
from z_np.src import cColorCython as cColor
from . import _profile

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer
    from .cBrushCore import BrushHitState




NODE_NAME = "WeightPreviewShape"
NODE_ID = om.MTypeId(0x80005)

DRAW_CLASSIFICATION = "drawdb/geometry/WeightPreview"
DRAW_REGISTRAR = "WeightPreviewShapeRegistrar"


# ==============================================================================
# 📦 数据结构：Shape 节点专属的网格与笔刷上下文
# ==============================================================================
@dataclass
class MeshDisplayContext:
    """专供视口显示与笔刷射线检测的拓扑与坐标缓存"""

    vertex_count: int = 0
    rawPoints_output: CMemoryManager = None
    rawPoints2D_output: CMemoryManager = None

    # 拓扑缓存 (需在拓扑改变时重建)
    tri_indices_2D: CMemoryManager = None
    tri_to_face_map: CMemoryManager = None
    base_edge_indices: CMemoryManager = None
    adj_offsets: CMemoryManager = None
    adj_indices: CMemoryManager = None


@dataclass
class BrushDisplayContext:
    """存放笔刷在当前 Shape 上的运行时状态"""

    brush_hit_state: "BrushHitState" = None
    brush_epoch: int = 1

    # 距离和队列的缓存池
    pool_node_epochs: CMemoryManager = None
    pool_dist: CMemoryManager = None
    pool_queue: CMemoryManager = None
    pool_in_queue: CMemoryManager = None
    pool_touched: CMemoryManager = None


# ==============================================================================
# 🎨 视口渲染器 (View)
# ==============================================================================
class WeightGeometryOverride(omr.MPxGeometryOverride):
    RENDER_POINTS = True
    RENDER_LINE = True
    RENDER_POLYGONS = True

    points_size = 1.0
    lines_width = 1.0

    def __init__(self, mObjectShape):
        super(WeightGeometryOverride, self).__init__(mObjectShape)

        self.mObject_shape: om.MObject = mObjectShape
        self.mFnDep_shape: om.MFnDependencyNode = om.MFnDependencyNode(mObjectShape)
        self.shape_class: WeightPreviewShape = self.mFnDep_shape.userNode()

        # 拓扑快照 (仅拓扑改变时更新)
        self._cached_vertex_count = 0
        self._cached_solid_mgr: CMemoryManager = None
        self._cached_wire_mgr: CMemoryManager = None
        self._cached_point_mgr: CMemoryManager = None
        self._indices_initialized: bool = False
        self._last_topo_cache = None

        # 💥 渲染负载快照 (Render Payload) - 每一帧都会硬性刷新
        self._cached_raw_points_mgr = None
        self._cached_weights_1d = None
        self._cached_hit_state = None

        # 缓存的UI状态
        self._cached_paintMask = False
        self._cached_render_mode = 0
        self._cached_c_wire = (0.0, 1.0, 1.0, 1.0)
        self._cached_c_point = (1.0, 0.0, 0.0, 1.0)
        self._cached_c_mask_remapA = (0.1, 0.1, 0.1, 0.0)
        self._cached_c_mask_remapB = (0.1, 1.0, 0.1, 0.0)
        self._cached_c_weights_remapA = (0.0, 0.0, 0.0, 0.0)
        self._cached_c_weights_remapB = (1.0, 1.0, 1.0, 0.0)
        self._cached_c_brush_remapA = (1.0, 0.0, 0.0, 1.0)
        self._cached_c_brush_remapB = (1.0, 1.0, 0.0, 1.0)

        self.renderStatus: bool = False

        # 初始化着色器
        shader_mgr = omr.MRenderer.getShaderManager()
        self.cpv_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVSolidShader)

        self.wire_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVThickLineShader)
        self.wire_shader.setParameter("lineWidth", [WeightGeometryOverride.lines_width, WeightGeometryOverride.lines_width])

        self.point_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVFatPointShader)
        self.point_shader.setParameter("pointSize", [WeightGeometryOverride.points_size, WeightGeometryOverride.points_size])

    def updateDG(self):
        with _profile.MicroProfiler(target_runs=100, enable=False) as prof:
            self.renderStatus = False

            shape = self.shape_class
            cSkin = shape.cSkin

            if not cSkin or not shape.deformMesh_plug.isConnected:
                return

            prof.step("updateDG:---------预处理与解算")

            # 1. 强制拉取上游的网格解算结果
            shape.update_mesh_points()

            mesh_ctx = shape.mesh_context
            if not mesh_ctx.rawPoints_output:
                return

            prof.step("updateDG:---------更新模型结构")
            # 2. 同步 UI 状态到 Shape 内部
            shape.sync_ui_state_to_blackboard()

            # region Topology Cache
            _cache = self._get_topology_index_buffers(mesh_ctx)
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
            # endregion

            self._cached_raw_points_mgr = mesh_ctx.rawPoints_output

            # region Color & Weights
            weights2D_mgr, target_idx, is_mask = shape.active_paint_target
            self._cached_weights_1d = None
            if weights2D_mgr is not None and weights2D_mgr.view is not None:
                mv_2d = weights2D_mgr.view
                cols = mv_2d.shape[1] if len(mv_2d.shape) > 1 else 1
                safe_idx = max(0, min(target_idx, cols - 1))
                mv_1d_flat = mv_2d.cast("B").cast("f")
                self._cached_weights_1d = mv_1d_flat[safe_idx::cols]

            self._cached_paintMask = is_mask
            self._cached_render_mode = shape.render_mode
            self._cached_c_wire = shape.color_wire
            self._cached_c_point = shape.color_point
            self._cached_c_mask_remapA = shape.color_mask_remapA
            self._cached_c_mask_remapB = shape.color_mask_remapB
            self._cached_c_weights_remapA = shape.color_weights_remapA
            self._cached_c_weights_remapB = shape.color_weights_remapB
            self._cached_c_brush_remapA = shape.color_brush_remapA
            self._cached_c_brush_remapB = shape.color_brush_remapB
            # endregion

            # Brush DATA
            self._cached_hit_state = shape.brush_context.brush_hit_state

            self.renderStatus = True
            prof.step("updateDG:---------准备数据结束")

    def populateGeometry(self, requirements, renderItems, data):
        if not self.renderStatus:
            return

        N = self._cached_vertex_count
        points_mgr = self._cached_raw_points_mgr

        for req in requirements.vertexRequirements():
            if req.semantic == omr.MGeometry.kPosition:
                if points_mgr and points_mgr.ptr_addr:
                    vtx_buf = data.createVertexBuffer(req)
                    vtx_addr = vtx_buf.acquire(N * 3, True)
                    if vtx_addr:
                        stride = N * 12
                        ctypes.memmove(vtx_addr, points_mgr.ptr_addr, stride)
                        ctypes.memmove(vtx_addr + stride, points_mgr.ptr_addr, stride)
                        ctypes.memmove(vtx_addr + stride * 2, points_mgr.ptr_addr, stride)
                        vtx_buf.commit(vtx_addr)

            elif req.semantic == omr.MGeometry.kColor:
                color_buf = data.createVertexBuffer(req)
                color_addr = color_buf.acquire(N * 3, True)
                if color_addr:
                    color_view = CMemoryManager.from_ptr(color_addr, "f", (N * 3, 4)).view

                    if self._cached_weights_1d is not None:
                        if self._cached_paintMask:
                            cColor.render_gradient(self._cached_weights_1d, color_view[0:N], self._cached_c_mask_remapA, self._cached_c_mask_remapB)
                        elif self._cached_render_mode == 1:
                            cColor.render_gradient(self._cached_weights_1d, color_view[0:N], self._cached_c_weights_remapA, self._cached_c_weights_remapB)
                        else:
                            cColor.render_heatmap(self._cached_weights_1d, color_view[0:N])
                    else:
                        cColor.render_fill(color_view[0:N], (0.0, 0.0, 1.0, 1.0))

                    cColor.render_fill(color_view[N : 2 * N], self._cached_c_wire)

                    hit_state = self._cached_hit_state
                    if hit_state and hit_state.hit_count > 0:
                        cColor.render_brush_gradient(
                            color_view[2 * N : 3 * N],
                            hit_state.hit_indices_mgr.view,
                            hit_state.hit_weights_mgr.view,
                            hit_state.hit_count,
                            self._cached_c_brush_remapA,
                            self._cached_c_brush_remapB,
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
                        ctypes.memmove(i_addr, mgr.ptr_addr, num_indices * 4)
                        i_buf.commit(i_addr)
                        item.associateWithIndexBuffer(i_buf)

            elif item_name == "WeightWireItem" and self._cached_wire_mgr:
                if not self._indices_initialized:
                    mgr = self._cached_wire_mgr
                    num_indices = mgr.view.nbytes // 4
                    i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32)
                    i_addr = i_buf.acquire(num_indices, True)
                    if i_addr:
                        cColor.offset_indices_direct(mgr.ptr_addr, int(i_addr), num_indices, N)
                        i_buf.commit(i_addr)
                        item.associateWithIndexBuffer(i_buf)

            elif item_name == "BrushDebugPoints":
                hit_state = self._cached_hit_state
                if hit_state and (hit_state.hit_count > 0):
                    i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32)
                    i_addr = i_buf.acquire(hit_state.hit_count, True)
                    if i_addr:
                        cColor.offset_indices_direct(
                            hit_state.hit_indices_mgr.ptr_addr,
                            int(i_addr),
                            hit_state.hit_count,
                            2 * N,
                        )
                        i_buf.commit(i_addr)
                        item.associateWithIndexBuffer(i_buf)

        self._indices_initialized = True

    def _get_topology_index_buffers(self, mesh_ctx: MeshDisplayContext):
        N = mesh_ctx.vertex_count
        if N == 0 or mesh_ctx.tri_indices_2D is None:
            return None

        if (self._cached_vertex_count == N) and (self._cached_solid_mgr is not None):
            return (
                self._cached_solid_mgr,
                self._cached_wire_mgr,
                self._cached_point_mgr,
                self._cached_vertex_count,
            )

        new_solid_mgr = mesh_ctx.tri_indices_2D
        new_wire_mgr = mesh_ctx.base_edge_indices
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
            item.setDepthPriority(depth_priority)
        item.enable(True)

    def updateRenderItems(self, objPath, renderItems):
        if WeightGeometryOverride.RENDER_POLYGONS:
            self._setup_render_item(renderItems, "WeightSolidItem", omr.MGeometry.kTriangles, self.cpv_shader)

        if WeightGeometryOverride.RENDER_LINE:
            self._setup_render_item(renderItems, "WeightWireItem", omr.MGeometry.kLines, self.wire_shader, omr.MRenderItem.sActiveWireDepthPriority)

        if WeightGeometryOverride.RENDER_POINTS:
            self._setup_render_item(renderItems, "BrushDebugPoints", omr.MGeometry.kPoints, self.point_shader, omr.MRenderItem.sActivePointDepthPriority)
            idx = renderItems.indexOf("BrushDebugPoints")
            if idx >= 0:
                item = renderItems[idx]
                hit_state = self.shape_class.brush_context.brush_hit_state if self.shape_class else None
                item.enable(hit_state is not None and hit_state.hit_count > 0)

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

    def __init__(self):
        super(WeightPreviewShape, self).__init__()
        self._boundingBox = om.MBoundingBox(om.MPoint((-10, -10, -10)), om.MPoint((10, 10, 10)))

        # 💥 实例缓存池：再也不用每次去注册表捞了
        self._cached_cSkin = None

        # 初始化上下文 (代替以前的全局 DATA)
        self.mesh_context = MeshDisplayContext()
        self.brush_context = BrushDisplayContext()

        # 占位：如果有需要在UI直接访问的笔刷设置，可以在这里实例化
        from .cBrushCore import BrushSettings

        self.brush_settings = BrushSettings()

        # 用户交互参数 (UI 层直接读写)
        self.paintLayerIndex: int = -1
        self.paintInfluenceIndex: int = 0
        self.paintMask: bool = False

        # 🎨 色彩与显示配置 (灵活到可以做皮肤主题)
        self.color_wire = (0.0, 1.0, 1.0, 1.0)
        self.color_point = (1.0, 0.0, 0.0, 1.0)
        self.color_mask_remapA = (0.1, 0.1, 0.1, 0.0)
        self.color_mask_remapB = (0.1, 1.0, 0.1, 0.0)
        self.color_weights_remapA = (0.0, 0.0, 0.0, 0.0)
        self.color_weights_remapB = (1.0, 1.0, 1.0, 0.0)
        self.color_brush_remapA = (1.0, 0.0, 0.0, 1.0)
        self.color_brush_remapB = (1.0, 1.0, 0.0, 1.0)
        self.render_mode = 0

    def postConstructor(self):
        self.mObj = self.thisMObject()
        self.layer_plug = om.MPlug(self.mObj, self.aLayer)
        self.mask_plug = om.MPlug(self.mObj, self.aMask)
        self.influence_plug = om.MPlug(self.mObj, self.aInfluence)
        self.deformMesh_plug = om.MPlug(self.mObj, self.aInDeformMesh)
        return super().postConstructor()

    @property
    def cSkin(self) -> "CythonSkinDeformer":
        """获取绑定的 cSkin 实例"""
        if self._cached_cSkin is None:
            if not self.deformMesh_plug.isConnected:
                return None
            connected_plugs = self.deformMesh_plug.connectedTo(True, False)
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
        if not cSkin or (self.paintLayerIndex not in cSkin.weightsLayer):
            return None, None, None

        active_layer = cSkin.weightsLayer[self.paintLayerIndex]

        if self.paintMask:
            if not active_layer.maskHandle or not active_layer.maskHandle.is_valid:
                return None, None, None
            return active_layer.maskHandle.memory.reshape((self.mesh_context.vertex_count, 1)), 0, True
        else:
            if not active_layer.weightsHandle or not active_layer.weightsHandle.is_valid:
                return None, None, None
            return active_layer.weightsHandle.memory.reshape((self.mesh_context.vertex_count, cSkin.influences_count)), self.paintInfluenceIndex, False

    def update_mesh_points(self):
        """
        💥 零开销获取：直接从 cSkin 中“白嫖”顶点的物理指针。不再经过 API 调用。
        """
        # 强制 DG 计算，确保上游的 deform 运行过了
        _ = self.deformMesh_plug.asMObject()

        cSkin = self.cSkin
        if not cSkin:
            return

        vtx_count = cSkin.vertex_count
        if not vtx_count:
            return

        # 如果顶点数变了，触发拓扑缓存重建
        if self.mesh_context.vertex_count != vtx_count:
            # 拓扑不会每帧变，用 om2 重新捞一遍边和邻接表
            import maya.api.OpenMaya as om2

            mFnMesh_om2 = om2.MFnMesh(self.deformMesh_plug.asMObject())
            self._build_topology_cache(mFnMesh_om2, vtx_count)

        # 写入 Mesh Context
        self.mesh_context.vertex_count = vtx_count

        # 💥 跨域读取！
        self.mesh_context.rawPoints_output = cSkin.rawPoints_output_mgr

        # 将原始一维指针映射为 N*3 的二维视图供笔刷用
        if self.mesh_context.rawPoints_output:
            ptr = self.mesh_context.rawPoints_output.ptr_addr
            self.mesh_context.rawPoints2D_output = CMemoryManager.from_ptr(ptr, "f", (vtx_count, 3))

    def _build_topology_cache(self, mFnMesh_om2, vtx_count: int):
        """
        [预留接口] 拓扑数据生成器
        使用 mFnMesh_om2 (OpenMaya 2.0) 生成三角形索引、邻接表等拓扑数据。
        """
        # 示例：
        # self.mesh_context.tri_indices_2D = ...
        # self.mesh_context.base_edge_indices = ...
        # self.mesh_context.adj_offsets = ...
        # self.mesh_context.adj_indices = ...

        # 同时为笔刷分配对应顶点数量的缓存池
        self.brush_context.pool_node_epochs = CMemoryManager.allocate("i", (vtx_count,))
        self.brush_context.pool_dist = CMemoryManager.allocate("f", (vtx_count,))
        self.brush_context.pool_queue = CMemoryManager.allocate("i", (vtx_count,))
        self.brush_context.pool_in_queue = CMemoryManager.allocate("B", (vtx_count,))
        self.brush_context.pool_touched = CMemoryManager.allocate("i", (vtx_count,))

    def sync_ui_state_to_blackboard(self):
        """将前端 Plug 属性同步到 Shape 的本地状态中"""
        self.paintLayerIndex     = self.layer_plug.asInt()  # fmt:skip
        self.paintInfluenceIndex = self.influence_plug.asInt()  # fmt:skip
        self.paintMask           = self.mask_plug.asBool()  # fmt:skip

        self.render_mode = om.MPlug(self.thisMObject(), self.aRenderMode).asInt()

        def get_color(attr_obj, alpha=1.0):
            plug = om.MPlug(self.thisMObject(), attr_obj)
            vec = plug.asFloatVector()
            return (vec.x, vec.y, vec.z, alpha)

        # fmt:off
        self.color_wire           = get_color(self.aColorWire          , 1.0)  # fmt:skip
        self.color_point          = get_color(self.aColorPoint         , 1.0)  # fmt:skip
        self.color_mask_remapA    = get_color(self.aColorMaskRemapA    , 0.0)  # fmt:skip
        self.color_mask_remapB    = get_color(self.aColorMaskRemapB    , 0.0)  # fmt:skip
        self.color_weights_remapA = get_color(self.aColorWeightsRemapA , 0.0)  # fmt:skip
        self.color_weights_remapB = get_color(self.aColorWeightsRemapB , 0.0)  # fmt:skip
        self.color_brush_remapA   = get_color(self.aColorBrushRemapA   , 1.0)  # fmt:skip
        self.color_brush_remapB   = get_color(self.aColorBrushRemapB   , 1.0)  # fmt:skip
        # fmt:on

    def connectionBroken(self, plug, otherPlug, asSrc):
        if plug == self.deformMesh_plug:
            self._cached_cSkin = None
        return super().connectionBroken(plug, otherPlug, asSrc)

    def setDependentsDirty(self, plug, plugArray):
        attr = plug.attribute()
        if attr in (self.aInDeformMesh, self.aLayer, self.aMask, self.aInfluence):
            omr.MRenderer.setGeometryDrawDirty(self.thisMObject(), True)
        return super().setDependentsDirty(plug, plugArray)

    def postEvaluation(self, context, evaluationNode, evalType):
        omr.MRenderer.setGeometryDrawDirty(self.thisMObject(), True)
        super().postEvaluation(context, evaluationNode, evalType)

    def preEvaluation(self, context, evaluationNode):
        if evaluationNode.dirtyPlugExists(self.aInDeformMesh) or evaluationNode.dirtyPlugExists(self.aLayer) or evaluationNode.dirtyPlugExists(self.aInfluence) or evaluationNode.dirtyPlugExists(self.aMask):
            omr.MRenderer.setGeometryDrawDirty(self.thisMObject(), True)

        super().preEvaluation(context, evaluationNode)

    @staticmethod
    def initialize():
        nAttr = om.MFnNumericAttribute()
        tAttr: om.MFnTypedAttribute = om.MFnTypedAttribute()
        eAttr = om.MFnEnumAttribute()

        tAttr: om.MFnTypedAttribute = om.MFnTypedAttribute()
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

        WeightPreviewShape.aColorWire = add_color("colorWire", "cwir", (0.0, 1.0, 1.0))
        WeightPreviewShape.aColorPoint = add_color("colorPoint", "cpnt", (1.0, 0.0, 0.0))
        WeightPreviewShape.aColorMaskRemapA = add_color("colorMaskRemapA", "cmra", (0.1, 0.1, 0.1))
        WeightPreviewShape.aColorMaskRemapB = add_color("colorMaskRemapB", "cmrb", (0.1, 1.0, 0.1))
        WeightPreviewShape.aColorWeightsRemapA = add_color("colorWeightsRemapA", "cwra", (0.0, 0.0, 0.0))
        WeightPreviewShape.aColorWeightsRemapB = add_color("colorWeightsRemapB", "cwrb", (1.0, 1.0, 1.0))
        WeightPreviewShape.aColorBrushRemapA = add_color("colorBrushRemapA", "cbra", (1.0, 0.0, 0.0))
        WeightPreviewShape.aColorBrushRemapB = add_color("colorBrushRemapB", "cbrb", (1.0, 1.0, 0.0))

    def isBounded(self):
        return True

    def boundingBox(self):
        mesh_ctx = self.mesh_context
        if mesh_ctx and mesh_ctx.rawPoints_output:
            boxMin, boxMax = cBoundingBoxCython.compute_bbox_fast(mesh_ctx.rawPoints_output.view, mesh_ctx.vertex_count)
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
