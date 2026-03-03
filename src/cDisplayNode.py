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


NODE_NAME = "WeightPreviewShape"
NODE_ID = om.MTypeId(0x80005)

DRAW_CLASSIFICATION = "drawdb/geometry/WeightPreview"
DRAW_REGISTRAR = "WeightPreviewShapeRegistrar"


# ==============================================================================
# 📦 数据结构：Shape 节点专属的网格与笔刷上下文
# ==============================================================================
@dataclass(frozen=True, slots=True)
class TopologyCache:
    """存放网格的静态拓扑数据，仅在拓扑改变时重建"""

    # fmt:off
    vertex_count      : int            = 0
    tri_indices_2D    : CMemoryManager = None
    tri_to_face_map   : CMemoryManager = None
    base_edge_indices : CMemoryManager = None
    adj_offsets       : CMemoryManager = None
    adj_indices       : CMemoryManager = None
    # fmt:on


@dataclass(slots=True)
class MeshDisplayContext:
    """专供视口显示与笔刷射线检测的拓扑与坐标缓存"""

    # 几何状态 (每帧都可能更新)
    rawPoints_output: CMemoryManager = None
    rawPoints2D_output: CMemoryManager = None
    # 拓扑缓存 (仅在拓扑改变时更新)
    topology: TopologyCache = None


@dataclass(slots=True)
class BrushDisplayContext:
    """存放笔刷在当前 Shape 上的运行时状态"""

    # fmt:off
    # 笔刷渲染数据 (由外部工具填充)
    brush_hit_count  : int             = 0
    brush_hit_indices: CMemoryManager  = None
    brush_hit_weights: CMemoryManager  = None
    brush_epoch      : int             = 1
    # 距离和队列的缓存池 (供笔刷算法使用)
    pool_node_epochs : CMemoryManager  = None
    pool_dist        : CMemoryManager  = None
    pool_queue       : CMemoryManager  = None
    pool_in_queue    : CMemoryManager  = None
    pool_touched     : CMemoryManager  = None
    # fmt:on


@dataclass(frozen=True, slots=True)
class RenderState:
    """存放 UI 颜色、渲染模式等显示配置的快照"""

    # fmt:off
    render_mode         : int   = 0
    paintLayerIndex     : int   = -1
    paintInfluenceIndex : int   = 0
    paintMask           : bool  = False
    color_wire          : tuple = (0.0, 1.0, 1.0, 1.0)
    color_point         : tuple = (1.0, 0.0, 0.0, 1.0)
    color_mask_remapA   : tuple = (0.1, 0.1, 0.1, 0.0)
    color_mask_remapB   : tuple = (0.1, 1.0, 0.1, 0.0)
    color_weights_remapA: tuple = (0.0, 0.0, 0.0, 0.0)
    color_weights_remapB: tuple = (1.0, 1.0, 1.0, 0.0)
    color_brush_remapA  : tuple = (1.0, 0.0, 0.0, 1.0)
    color_brush_remapB  : tuple = (1.0, 1.0, 0.0, 1.0)
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
        "_cached_raw_points_mgr",
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
        self._cached_raw_points_mgr = None
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

            # region Color & Weights & State System
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
            # endregion

            # Brush DATA
            brush_ctx = shape.brush_context
            self._cached_brush_hit_count = brush_ctx.brush_hit_count
            self._cached_brush_hit_indices = brush_ctx.brush_hit_indices
            self._cached_brush_hit_weights = brush_ctx.brush_hit_weights

            prof.step("updateDG:---------准备数据结束")

    def populateGeometry(self, requirements, renderItems, data):
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
                if self._cached_brush_hit_count > 0 and self._cached_brush_hit_indices:
                    i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32)
                    i_addr = i_buf.acquire(self._cached_brush_hit_count, True)
                    if i_addr:
                        cColor.offset_indices_direct(
                            self._cached_brush_hit_indices.ptr_addr,
                            int(i_addr),
                            self._cached_brush_hit_count,
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

    def __init__(self):
        super(WeightPreviewShape, self).__init__()
        self._boundingBox = om.MBoundingBox(om.MPoint((-10, -10, -10)), om.MPoint((10, 10, 10)))

        # 💥 实例缓存池：再也不用每次去注册表捞了
        self._cached_cSkin = None

        # 初始化上下文 (代替以前的全局 DATA)
        self.mesh_context = MeshDisplayContext()
        self.brush_context = BrushDisplayContext()

        # 占位：如果有需要在UI直接访问的笔刷设置，可以在这里实例化
        # 🎨 渲染显示数据
        self.render_state = RenderState()

    def postConstructor(self):
        self.mObj = self.thisMObject()
        # 预先获取所有需要用到的 Plug
        self.layer_plug = om.MPlug(self.mObj, self.aLayer)
        self.mask_plug = om.MPlug(self.mObj, self.aMask)
        self.influence_plug = om.MPlug(self.mObj, self.aInfluence)
        self.deformMesh_plug = om.MPlug(self.mObj, self.aInDeformMesh)
        self.renderMode_plug = om.MPlug(self.mObj, self.aRenderMode)
        self.colorWire_plug = om.MPlug(self.mObj, self.aColorWire)
        self.colorPoint_plug = om.MPlug(self.mObj, self.aColorPoint)
        self.colorMaskRemapA_plug = om.MPlug(self.mObj, self.aColorMaskRemapA)
        self.colorMaskRemapB_plug = om.MPlug(self.mObj, self.aColorMaskRemapB)
        self.colorWeightsRemapA_plug = om.MPlug(self.mObj, self.aColorWeightsRemapA)
        self.colorWeightsRemapB_plug = om.MPlug(self.mObj, self.aColorWeightsRemapB)
        self.colorBrushRemapA_plug = om.MPlug(self.mObj, self.aColorBrushRemapA)
        self.colorBrushRemapB_plug = om.MPlug(self.mObj, self.aColorBrushRemapB)
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
        state = self.render_state
        if not cSkin or (state.paintLayerIndex not in cSkin.weightsLayer):
            return None, None, None

        active_layer = cSkin.weightsLayer[state.paintLayerIndex]

        if state.paintMask:
            if not active_layer.maskHandle or not active_layer.maskHandle.is_valid:
                return None, None, None
            return active_layer.maskHandle.memory.reshape((self.mesh_context.vertex_count, 1)), 0, True
        else:
            if not active_layer.weightsHandle or not active_layer.weightsHandle.is_valid:
                return None, None, None
            return active_layer.weightsHandle.memory.reshape((self.mesh_context.vertex_count, cSkin.influences_count)), state.paintInfluenceIndex, False

    def update_mesh_points(self):
        """
        零开销直接从 cSkin 中“白嫖”顶点的物理指针。不再经过 API 调用。
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
        [补全] 拓扑数据生成器
        使用 mFnMesh_om2 (OpenMaya 2.0) 生成三角形索引、邻接表等拓扑数据。
        """
        # 1. 提取三角形索引与面映射 (用于渲染和笔刷射线检测)
        tri_counts, tri_vtx_indices = mFnMesh_om2.getTriangles()
        self.mesh_context.tri_indices_2D = CMemoryManager.from_list(list(tri_vtx_indices), "i")

        tri_to_face = []
        for face_idx, count in enumerate(tri_counts):
            tri_to_face.extend([face_idx] * count)
        self.mesh_context.tri_to_face_map = CMemoryManager.from_list(tri_to_face, "i")

        # 2. 提取边索引 (用于线框显示) 并构建邻接关系
        num_edges = mFnMesh_om2.numEdges
        edge_indices = [0] * (num_edges * 2)
        adj_list = [set() for _ in range(vtx_count)]

        for i in range(num_edges):
            u, v = mFnMesh_om2.getEdgeVertices(i)
            edge_indices[i * 2] = u
            edge_indices[i * 2 + 1] = v
            # 双向连接
            adj_list[u].add(v)
            adj_list[v].add(u)

        self.mesh_context.base_edge_indices = CMemoryManager.from_list(edge_indices, "i")

        # 3. 将邻接表转换为高性能的 CSR (Compressed Sparse Row) 格式
        flat_adj = []
        adj_offsets = [0] * (vtx_count + 1)
        for i in range(vtx_count):
            neighbors = sorted(list(adj_list[i]))
            flat_adj.extend(neighbors)
            adj_offsets[i + 1] = adj_offsets[i] + len(neighbors)

        self.mesh_context.adj_offsets = CMemoryManager.from_list(adj_offsets, "i")
        self.mesh_context.adj_indices = CMemoryManager.from_list(flat_adj, "i")

        # 同时为笔刷分配对应顶点数量的缓存池
        self.brush_context.pool_node_epochs = CMemoryManager.allocate("i", (vtx_count,))
        self.brush_context.pool_dist = CMemoryManager.allocate("f", (vtx_count,))
        self.brush_context.pool_queue = CMemoryManager.allocate("i", (vtx_count,))
        self.brush_context.pool_in_queue = CMemoryManager.allocate("B", (vtx_count,))
        self.brush_context.pool_touched = CMemoryManager.allocate("i", (vtx_count,))

    def sync_ui_state_to_blackboard(self):
        """将前端 Plug 属性同步到 Shape 的本地状态中"""

        def get_color(plug: om.MPlug, alpha=1.0):
            vec = plug.asFloatVector()
            return (vec.x, vec.y, vec.z, alpha)

        # 构造新状态
        new_state = RenderState(
            paintLayerIndex      = self.layer_plug.asInt(),
            paintInfluenceIndex  = self.influence_plug.asInt(),
            paintMask            = self.mask_plug.asBool(),
            render_mode          = self.renderMode_plug.asInt(),
            color_wire           = get_color(self.colorWire_plug, 1.0),
            color_point          = get_color(self.colorPoint_plug, 1.0),
            color_mask_remapA    = get_color(self.colorMaskRemapA_plug, 0.0),
            color_mask_remapB    = get_color(self.colorMaskRemapB_plug, 0.0),
            color_weights_remapA = get_color(self.colorWeightsRemapA_plug, 0.0),
            color_weights_remapB = get_color(self.colorWeightsRemapB_plug, 0.0),
            color_brush_remapA   = get_color(self.colorBrushRemapA_plug, 1.0),
            color_brush_remapB   = get_color(self.colorBrushRemapB_plug, 1.0),
        )

        # 只有在状态真正改变时才替换，触发后续可能的重绘逻辑
        if self.render_state != new_state:
            self.render_state = new_state

    def connectionBroken(self, plug, otherPlug, asSrc):
        if plug == self.deformMesh_plug:
            self._cached_cSkin = None
        return super().connectionBroken(plug, otherPlug, asSrc)

    def setDependentsDirty(self, plug, plugArray):
        attr = plug.attribute()
        if attr in (self.aInDeformMesh, self.aLayer, self.aMask, self.aInfluence):
            omr.MRenderer.setGeometryDrawDirty(self.mObj, True)
        return super().setDependentsDirty(plug, plugArray)

    def postEvaluation(self, context, evaluationNode, evalType):
        omr.MRenderer.setGeometryDrawDirty(self.mObj, True)
        super().postEvaluation(context, evaluationNode, evalType)

    def preEvaluation(self, context, evaluationNode):
        if evaluationNode.dirtyPlugExists(self.aInDeformMesh) or evaluationNode.dirtyPlugExists(self.aLayer) or evaluationNode.dirtyPlugExists(self.aInfluence) or evaluationNode.dirtyPlugExists(self.aMask):
            omr.MRenderer.setGeometryDrawDirty(self.mObj, True)

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

        # fmt:off
        WeightPreviewShape.aColorWire          = add_color("colorWire"         , "cwir", (0.0, 1.0, 1.0)) 
        WeightPreviewShape.aColorPoint         = add_color("colorPoint"        , "cpnt", (1.0, 0.0, 0.0))
        WeightPreviewShape.aColorMaskRemapA    = add_color("colorMaskRemapA"   , "cmra", (0.1, 0.1, 0.1))
        WeightPreviewShape.aColorMaskRemapB    = add_color("colorMaskRemapB"   , "cmrb", (0.1, 1.0, 0.1))
        WeightPreviewShape.aColorWeightsRemapA = add_color("colorWeightsRemapA", "cwra", (0.0, 0.0, 0.0))
        WeightPreviewShape.aColorWeightsRemapB = add_color("colorWeightsRemapB", "cwrb", (1.0, 1.0, 1.0))
        WeightPreviewShape.aColorBrushRemapA   = add_color("colorBrushRemapA"  , "cbra", (1.0, 0.0, 0.0))
        WeightPreviewShape.aColorBrushRemapB   = add_color("colorBrushRemapB"  , "cbrb", (1.0, 1.0, 0.0))
        # fmt:on

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
