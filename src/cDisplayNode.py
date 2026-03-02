from __future__ import annotations
import typing
import ctypes
import itertools

import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui
import maya.api.OpenMayaRender as omr

# --- 优化: 将紧密相关的上下文类直接定义在此文件中 ---
from . import cBoundingBoxCython
from . import cMemoryView
from . import _cRegistry
from . import cColorCython as cColor
from . import _profile
from . import cBrushCore

if typing.TYPE_CHECKING:
    import maya.OpenMaya as om1_api
    from . import cSkinDeform


def maya_useNewAPI():
    pass

NODE_NAME = "WeightPreviewShape"
NODE_ID = om.MTypeId(0x80005)
DRAW_CLASSIFICATION = "drawdb/geometry/WeightPreview"
DRAW_REGISTRAR = "WeightPreviewShapeRegistrar"


class MeshDataContext:
    __slots__ = (
        "vertex_count", "tri_indices_2D", "tri_to_face_map", "base_edge_indices",
        "adj_offsets", "adj_indices", "rawPoints_output", "mFnMesh_output",
    )

    def __init__(self) -> None:
        self.vertex_count: int = 0
        self.tri_indices_2D: cMemoryView.CMemoryManager = None
        self.tri_to_face_map: cMemoryView.CMemoryManager = None
        self.base_edge_indices: cMemoryView.CMemoryManager = None
        self.adj_offsets: cMemoryView.CMemoryManager = None
        self.adj_indices: cMemoryView.CMemoryManager = None
        self.rawPoints_output: cMemoryView.CMemoryManager = None
        self.mFnMesh_output: om.MFnMesh = None


class BrushDataContext:
    __slots__ = (
        "brush_settings", "brush_hit_state", "paintLayerIndex", "paintInfluenceIndex",
        "paintMask", "brush_epoch", "pool_node_epochs", "pool_dist", "pool_queue",
        "pool_in_queue", "pool_touched",
    )

    def __init__(self) -> None:
        self.brush_settings: cBrushCore.BrushSettings = cBrushCore.BrushSettings()
        self.brush_hit_state: cBrushCore.BrushHitState = None
        self.paintLayerIndex: int = -1
        self.paintInfluenceIndex: int = 0
        self.paintMask: bool = False
        self.brush_epoch: int = 1
        self.pool_node_epochs: cMemoryView.CMemoryManager = None
        self.pool_dist: cMemoryView.CMemoryManager = None
        self.pool_queue: cMemoryView.CMemoryManager = None
        self.pool_in_queue: cMemoryView.CMemoryManager = None
        self.pool_touched: cMemoryView.CMemoryManager = None


class DisplayDataContext:
    __slots__ = (
        "preview_shape_mObj", "color_wire", "color_point", "color_mask_remapA",
        "color_mask_remapB", "color_weights_remapA", "color_weights_remapB",
        "color_brush_remapA", "color_brush_remapB", "render_mode",
    )

    def __init__(self) -> None:
        self.preview_shape_mObj: om.MObject = None
        self.color_wire = (0.0, 1.0, 1.0, 1.0)
        self.color_point = (1.0, 0.0, 0.0, 1.0)
        self.color_mask_remapA = (0.1, 0.1, 0.1, 0.0)
        self.color_mask_remapB = (0.1, 1.0, 0.1, 0.0)
        self.color_weights_remapA = (0.0, 0.0, 0.0, 0.0)
        self.color_weights_remapB = (1.0, 1.0, 1.0, 0.0)
        self.color_brush_remapA = (1.0, 0.0, 0.0, 1.0)
        self.color_brush_remapB = (1.0, 1.0, 0.0, 1.0)
        self.render_mode: int = 0

class WeightGeometryOverride(omr.MPxGeometryOverride):
    # (此类的代码在此次重构中无需修改)
    def __init__(self, mObjectShape: om.MObject):
        super(WeightGeometryOverride, self).__init__(mObjectShape)
        self.mObject_shape = mObjectShape
        self.mFnDep_shape = om.MFnDependencyNode(mObjectShape)
        self.shape_node: WeightPreviewShape = self.mFnDep_shape.userNode()
        self._is_renderable = False
        self._indices_initialized = False
        self._last_topology_hash = None
        shader_mgr = omr.MRenderer.getShaderManager()
        self.cpv_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVSolidShader)
        self.wire_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVThickLineShader)
        self.point_shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVFatPointShader)

    def updateDG(self):
        self._is_renderable = False
        self.shape_node.prepare_render_data()
        if self.shape_node.is_renderable(): self._is_renderable = True

    def populateGeometry(self, requirements: omr.MGeometryRequirements, renderItems: omr.MRenderItemList, data: omr.MComponentData):
        if not self._is_renderable: return
        mesh_ctx, brush_ctx, display_ctx = self.shape_node.mesh_context, self.shape_node.brush_context, self.shape_node.display_context
        vertex_count, points_mgr = mesh_ctx.vertex_count, mesh_ctx.rawPoints_output
        for req in requirements.vertexRequirements():
            if req.semantic == omr.MGeometry.kPosition:
                if points_mgr and points_mgr.ptr:
                    vtx_buf = data.createVertexBuffer(req); vtx_addr = vtx_buf.acquire(vertex_count * 3, True)
                    if vtx_addr: stride = vertex_count * 12; ctypes.memmove(vtx_addr, points_mgr.ptr, stride); ctypes.memmove(vtx_addr + stride, points_mgr.ptr, stride); ctypes.memmove(vtx_addr + stride * 2, points_mgr.ptr, stride); vtx_buf.commit(vtx_addr)
            elif req.semantic == omr.MGeometry.kColor:
                color_buf = data.createVertexBuffer(req); color_addr = color_buf.acquire(vertex_count * 3, True)
                if color_addr:
                    color_view = cMemoryView.CMemoryManager.from_ptr(color_addr, "f", (vertex_count * 3, 4)).view
                    paint_target_weights = self.shape_node.get_paint_target_weights()
                    if paint_target_weights is not None:
                        if brush_ctx.paintMask: cColor.render_gradient(paint_target_weights, color_view[0:vertex_count], display_ctx.color_mask_remapA, display_ctx.color_mask_remapB)
                        elif display_ctx.render_mode == 1: cColor.render_gradient(paint_target_weights, color_view[0:vertex_count], display_ctx.color_weights_remapA, display_ctx.color_weights_remapB)
                        else: cColor.render_heatmap(paint_target_weights, color_view[0:vertex_count])
                    else: cColor.render_fill(color_view[0:vertex_count], (0.0, 0.0, 1.0, 1.0))
                    cColor.render_fill(color_view[vertex_count : 2 * vertex_count], display_ctx.color_wire)
                    hit_state = brush_ctx.brush_hit_state
                    if hit_state and hit_state.hit_count > 0: cColor.render_brush_gradient(color_view[2*vertex_count:3*vertex_count], hit_state.hit_indices_mgr.view, hit_state.hit_weights_mgr.view, hit_state.hit_count, display_ctx.color_brush_remapA, display_ctx.color_brush_remapB)
                    color_buf.commit(color_addr)
        current_topology_hash = self.shape_node.get_topology_hash()
        if self._last_topology_hash != current_topology_hash: self._indices_initialized = False; self._last_topology_hash = current_topology_hash
        for item in renderItems:
            item_name = item.name()
            if item_name == "WeightSolidItem" and mesh_ctx.tri_indices_2D:
                if not self._indices_initialized: mgr = mesh_ctx.tri_indices_2D; num_indices = mgr.view.nbytes // 4; i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32); i_addr = i_buf.acquire(num_indices, True); ctypes.memmove(i_addr, mgr.ptr, num_indices * 4); i_buf.commit(i_addr); item.associateWithIndexBuffer(i_buf)
            elif item_name == "WeightWireItem" and mesh_ctx.base_edge_indices:
                if not self._indices_initialized: mgr = mesh_ctx.base_edge_indices; num_indices = mgr.view.nbytes // 4; i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32); i_addr = i_buf.acquire(num_indices, True); cColor.offset_indices_direct(mgr.ptr, int(i_addr), num_indices, vertex_count); i_buf.commit(i_addr); item.associateWithIndexBuffer(i_buf)
            elif item_name == "BrushDebugPoints":
                hit_state = brush_ctx.brush_hit_state
                if hit_state and hit_state.hit_count > 0: i_buf = data.createIndexBuffer(omr.MGeometry.kUnsignedInt32); i_addr = i_buf.acquire(hit_state.hit_count, True); cColor.offset_indices_direct(hit_state.hit_indices_mgr.ptr, int(i_addr), hit_state.hit_count, 2 * vertex_count); i_buf.commit(i_addr); item.associateWithIndexBuffer(i_buf)
        self._indices_initialized = True

    def updateRenderItems(self, objPath, renderItems):
        self._setup_render_item(renderItems, "WeightSolidItem", omr.MGeometry.kTriangles, self.cpv_shader)
        self._setup_render_item(renderItems, "WeightWireItem", omr.MGeometry.kLines, self.wire_shader, omr.MRenderItem.sActiveWireDepthPriority)
        point_item = self._setup_render_item(renderItems, "BrushDebugPoints", omr.MGeometry.kPoints, self.point_shader, omr.MRenderItem.sActivePointDepthPriority)
        hit_state = self.shape_node.brush_context.brush_hit_state
        point_item.enable(hit_state is not None and hit_state.hit_count > 0)

    def _setup_render_item(self, renderItems, name, geom_type, shader, depth_priority=None) -> omr.MRenderItem:
        try: item = renderItems.get(name)
        except: item = omr.MRenderItem.create(name, omr.MRenderItem.MaterialSceneItem, geom_type); renderItems.append(item)
        item.setDrawMode(omr.MGeometry.kAll); item.setShader(shader)
        if depth_priority is not None: item.setDepthPriority(depth_priority)
        item.enable(True); return item
    def supportedDrawAPIs(self) -> int: return omr.MRenderer.kAllDevices
    @staticmethod
    def creator(obj: om.MObject) -> omr.MPxGeometryOverride: return WeightGeometryOverride(obj)

class WeightPreviewShape(om.MPxSurfaceShape):
    aLayer, aInfluence, aMask, aInDeformMesh = om.MObject(), om.MObject(), om.MObject(), om.MObject()

    def __init__(self):
        super(WeightPreviewShape, self).__init__()
        self._boundingBox = om.MBoundingBox()
        self.brush_context = BrushDataContext()
        self.display_context = DisplayDataContext()
        self.mesh_context = MeshDataContext()
        self._cached_cSkin_instance: "cSkinDeform.CythonSkinDeformer" = None
        self.skin_context: "cSkinDeform.SkinDeformerContext" = None
        self._renderable = False
        self._paint_target_weights_1d = None

    @property
    def cSkin(self) -> "cSkinDeform.CythonSkinDeformer" | None:
        if self._cached_cSkin_instance is None:
            if not self.deformMesh_plug.isConnected: return None
            plugs = self.deformMesh_plug.connectedTo(True, False); mObj_skin = plugs[0].node() if plugs else None
            if mObj_skin:
                instance = _cRegistry.SkinRegistry.get_instance_by_api2(mObj_skin)
                if instance: self._cached_cSkin_instance = instance; self.skin_context = instance.skin_context; self.display_context.preview_shape_mObj = self.thisMObject()
        return self._cached_cSkin_instance

    def connectionBroken(self, plug, otherPlug, asSrc):
        if plug == self.deformMesh_plug: self._cached_cSkin_instance = None; self.skin_context = None; self._renderable = False
        return super(WeightPreviewShape, self).connectionBroken(plug, otherPlug, asSrc)

    def prepare_render_data(self):
        self._renderable = False
        if not self.cSkin: return
        mesh_obj = self.deformMesh_plug.asMObject(); mFnMesh = om.MFnMesh(mesh_obj)
        if mFnMesh.numVertices() == 0: return
        vertex_count = mFnMesh.numVertices()
        self.mesh_context.rawPoints_output = cMemoryView.CMemoryManager.from_ptr(int(mFnMesh.getRawPoints()), "f", (vertex_count, 3))
        self.mesh_context.mFnMesh_output = mFnMesh
        if self.mesh_context.vertex_count != vertex_count: self.update_mesh_topology(mFnMesh, vertex_count)
        self.update_brush_memory_pools()
        self.sync_ui_state_to_context()
        self._paint_target_weights_1d = self._get_active_paint_weights_1d()
        self._renderable = True

    def update_mesh_topology(self, mFnMesh: om.MFnMesh, vertex_count: int):
        self.mesh_context.vertex_count = vertex_count
        tri_counts, tri_indices = mFnMesh.getTriangles(); num_tris = len(tri_indices) // 3
        self.mesh_context.tri_indices_2D = cMemoryView.CMemoryManager.from_list(list(tri_indices), "i").reshape((num_tris, 3))
        
        face_map_iterable = itertools.chain.from_iterable(itertools.repeat(face_id, count) for face_id, count in enumerate(tri_counts))
        self.mesh_context.tri_to_face_map = cMemoryView.CMemoryManager.from_list(list(face_map_iterable), "i")
        
        num_edges = mFnMesh.numEdges()
        self.mesh_context.base_edge_indices = cMemoryView.CMemoryManager.allocate("i", (num_edges * 2,))
        edge_view = self.mesh_context.base_edge_indices.view
        idx = 0
        for i in range(num_edges):
            edge_view[idx:idx+2] = mFnMesh.getEdgeVertices(i); idx += 2

        offsets_list = [0] * (vertex_count + 1); indices_list = []; current_offset = 0
        for i in range(vertex_count):
            neighbors = mFnMesh.getConnectedVertices(i); indices_list.extend(neighbors)
            offsets_list[i] = current_offset; current_offset += len(neighbors)
        offsets_list[vertex_count] = current_offset
        self.mesh_context.adj_offsets = cMemoryView.CMemoryManager.from_list(offsets_list, "i")
        self.mesh_context.adj_indices = cMemoryView.CMemoryManager.from_list(indices_list, "i")

    def _get_active_paint_weights_1d(self) -> cMemoryView.CMemoryManager | None:
        weights2D_mgr, target_idx, _ = self.active_paint_target
        if weights2D_mgr is not None and weights2D_mgr.view is not None:
            mv_2d = weights2D_mgr.view; cols = mv_2d.shape[1] if len(mv_2d.shape) > 1 else 1; safe_idx = max(0, min(target_idx, cols - 1))
            mv_1d_flat = mv_2d.cast("B").cast("f"); return mv_1d_flat[safe_idx::cols]
        return None

    @property
    def active_paint_target(self) -> typing.Tuple["cMemoryView.CMemoryManager", int, bool] | typing.Tuple[None, None, None]:
        if not self.skin_context or self.brush_context.paintLayerIndex not in self.skin_context.weightsLayer: return None, None, None
        active_layer = self.skin_context.weightsLayer[self.brush_context.paintLayerIndex]
        if self.brush_context.paintMask:
            if not active_layer.maskHandle or not active_layer.maskHandle.is_valid: return None, None, None
            return active_layer.maskHandle.memory.reshape((self.mesh_context.vertex_count, 1)), 0, True
        else:
            if not active_layer.weightsHandle or not active_layer.weightsHandle.is_valid: return None, None, None
            return active_layer.weightsHandle.memory.reshape((self.mesh_context.vertex_count, self.skin_context.influences_count)), self.brush_context.paintInfluenceIndex, False

    def sync_ui_state_to_context(self):
        if self.cSkin and self.brush_context: self.brush_context.paintLayerIndex = self.layer_plug.asInt(); self.brush_context.paintInfluenceIndex = self.influence_plug.asInt(); self.brush_context.paintMask = self.mask_plug.asBool()

    def update_brush_memory_pools(self):
        vertex_count = self.mesh_context.vertex_count
        if vertex_count == 0 or (self.brush_context.pool_node_epochs and self.brush_context.pool_node_epochs.view.shape[0] == vertex_count): return
        self.brush_context.pool_node_epochs = cMemoryView.CMemoryManager.allocate("i", (vertex_count,)); self.brush_context.pool_dist = cMemoryView.CMemoryManager.allocate("f", (vertex_count,)); self.brush_context.pool_queue = cMemoryView.CMemoryManager.allocate("i", (vertex_count,)); self.brush_context.pool_in_queue = cMemoryView.CMemoryManager.allocate("b", (vertex_count,)); self.brush_context.pool_touched = cMemoryView.CMemoryManager.allocate("i", (vertex_count,)); self.brush_context.pool_node_epochs.view[:] = 0

    def is_renderable(self) -> bool: return self._renderable
    def get_paint_target_weights(self) -> cMemoryView.CMemoryManager | None: return self._paint_target_weights_1d
    def get_topology_hash(self) -> int | None: return self.mesh_context.tri_indices_2D.ptr if self.mesh_context and self.mesh_context.tri_indices_2D else None

    def postConstructor(self):
        mObj = self.thisMObject(); self.layer_plug = om.MPlug(mObj, self.aLayer); self.mask_plug = om.MPlug(mObj, self.aMask); self.influence_plug = om.MPlug(mObj, self.aInfluence); self.deformMesh_plug = om.MPlug(mObj, self.aInDeformMesh)

    def setDependentsDirty(self, plug, plugArray):
        if plug.attribute() in (self.aInDeformMesh, self.aLayer, self.aMask, self.aInfluence): omr.MRenderer.setGeometryDrawDirty(self.thisMObject(), True)
        return super(WeightPreviewShape, self).setDependentsDirty(plug, plugArray)

    def preEvaluation(self, context, evaluationNode):
        if (evaluationNode.dirtyPlugExists(self.aInDeformMesh) or evaluationNode.dirtyPlugExists(self.aLayer) or evaluationNode.dirtyPlugExists(self.aInfluence) or evaluationNode.dirtyPlugExists(self.aMask)): omr.MRenderer.setGeometryDrawDirty(self.thisMObject(), True)
        super(WeightPreviewShape, self).preEvaluation(context, evaluationNode)

    def isBounded(self) -> bool: return True
    def boundingBox(self) -> om.MBoundingBox:
        if self.mesh_context and self.mesh_context.rawPoints_output: boxMin, boxMax = cBoundingBoxCython.compute_bbox_fast(self.mesh_context.rawPoints_output.view, self.mesh_context.vertex_count); self._boundingBox = om.MBoundingBox(om.MPoint(boxMin), om.MPoint(boxMax))
        return self._boundingBox

    @staticmethod
    def initialize():
        nAttr, tAttr = om.MFnNumericAttribute(), om.MFnTypedAttribute()
        WeightPreviewShape.aLayer = nAttr.create("layer", "lyr", om.MFnNumericData.kInt, 0); nAttr.storable = True; nAttr.channelBox = True; WeightPreviewShape.addAttribute(WeightPreviewShape.aLayer)
        WeightPreviewShape.aMask = nAttr.create("mask", "msk", om.MFnNumericData.kBoolean, False); nAttr.storable = True; nAttr.channelBox = True; WeightPreviewShape.addAttribute(WeightPreviewShape.aMask)
        WeightPreviewShape.aInfluence = nAttr.create("influence", "ifn", om.MFnNumericData.kInt, 0); nAttr.storable = True; nAttr.channelBox = True; WeightPreviewShape.addAttribute(WeightPreviewShape.aInfluence)
        WeightPreviewShape.aInDeformMesh = tAttr.create("inDeformMesh", "idm", om.MFnData.kMesh); tAttr.hidden = True; tAttr.storable = False; WeightPreviewShape.addAttribute(WeightPreviewShape.aInDeformMesh)
        
    @staticmethod
    def creator(): return WeightPreviewShape()

class WeightPreviewShapeUI(omui.MPxSurfaceShapeUI):
    def __init__(self): super(WeightPreviewShapeUI, self).__init__()
    @staticmethod
    def creator(): return WeightPreviewShapeUI()
