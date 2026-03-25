from __future__ import annotations
import ctypes

import maya.OpenMaya as om1  # type:ignore
import maya.OpenMayaMPx as ompx  # type:ignore

from . import _cRegistry
from . import cSkinDeformCython
from . import cTopologyCython as cTopology
from .cBufferManager import BufferManager
from .cWeightsManager import WeightsManager
from .cBrushCore2Cython import CoreBrushEngine

from ._cProfilerCython import MayaNativeProfiler, maya_profile

from typing import TYPE_CHECKING

# if TYPE_CHECKING:
from .cSkinContext import BrushHitContext


class MeshTopologyContext:
    """
    统一的数据结构 包含变形/渲染所需的所有网格数据。
    严格区分了动态的顶点位置与静态的拓扑/邻接表 方便 BufferManager 统一管理生命周期。
    """

    __slots__ = ("vertex_count",  # noqa: RUF023
                 "edge_count",
                 "polygon_count",
                 "triangle_count",
                 "vertex_positions",
                 "triangle_indices",
                 "edge_indices",
                 "quad_edge_indices",
                 "v2v_offsets",
                 "v2v_indices",
                 "v2f_offsets",
                 "v2f_indices")  # fmt:skip

    def __init__(self):
        self.vertex_count: int = 0
        self.edge_count: int = 0
        self.polygon_count: int = 0
        self.triangle_count: int = 0
        # 点位置 Buffer
        self.vertex_positions: BufferManager = None
        # 基础拓扑 Buffer
        self.triangle_indices: BufferManager = None
        self.edge_indices: BufferManager = None
        self.quad_edge_indices: BufferManager = None
        # v2v CSR Buffer
        self.v2v_offsets: BufferManager = None
        self.v2v_indices: BufferManager = None
        # v2f CSR Buffer
        self.v2f_offsets: BufferManager = None
        self.v2f_indices: BufferManager = None

    def clear(self):
        # fmt:off
        """可选的清理方法 用于显式释放底层的连续内存"""
        self.vertex_positions  = None
        self.triangle_indices  = None
        self.edge_indices      = None
        self.quad_edge_indices = None
        self.v2v_offsets       = None
        self.v2v_indices       = None
        self.v2f_offsets       = None
        self.v2f_indices       = None
        # fmt:on


class CythonSkinDeformer(ompx.MPxDeformerNode):
    __slots__ = (# 基础属性与 API 对象 (Base & API Objects) # noqa: RUF023
                 "hashCode",
                 "mObject",
                 "mFnDep",
                 "plug_refresh",
                 # 上下文与管理器 (Context & Managers)
                 "weights_manager",
                 "mesh_context",
                 "brush_context",
                 # 核心数据与缓存池 (Data & Buffers)
                 "influences_count",
                 "influences_locks_buffer",
                 # 脏标记体系 (Dirty Flags)
                 "isDirty",
                 "isDirty_weights",
                 "isDirty_geoMatrix",
                 "isDirty_inputGeometry",
                 "isDirty_bindPreMatrix",
                 "isDirty_influencesMatrix",
                 # 内部计算变量与内存视图 (Internal Vars & Views)
                 "_geo_matrix",
                 "_get_matrix_i",
                 "_geo_matrix_is_identity",
                 "_bindPreMatrix_buffer",
                 "_influencesMatrix_buffer",
                 "_rotateMatrix_buffer",
                 "_translateVector_buffer",
                 "_skinWeights")  # fmt:skip
    # fmt:off
    aGeomMatrix       = om1.MObject()
    aWeights          = om1.MObject()
    aLayerWeights     = om1.MObject()
    aLayerMask        = om1.MObject()
    aLayerEnabled     = om1.MObject()
    aLayerCompound    = om1.MObject()
    aInfluenceMatrix  = om1.MObject()
    aBindPreMatrix    = om1.MObject()
    aRefresh          = om1.MObject()
    # ---
    aCurrentPaintLayerIndex     = om1.MObject()
    aCurrentPaintInfluenceIndex = om1.MObject()
    aCurrentPaintMaskBool       = om1.MObject()
    # fmt:on

    def __init__(self):
        super().__init__()

        # fmt:off
        # --- maya
        self.hashCode       : int                   = None
        self.mObject        : om1.MObject           = None
        self.mFnDep         : om1.MFnDependencyNode = None
        self.plug_refresh   : om1.MPlug             = None

        # --- context
        self.weights_manager : WeightsManager      = None
        self.mesh_context    : MeshTopologyContext = None
        self.brush_context   : BrushHitContext     = None
        self.brush_engine    : CoreBrushEngine     = None

        # --- influences
        self.influences_count        : int           = 0
        self.influences_locks_buffer : BufferManager = None
        
        # --- current paint data
        self.layer_index      :int  = -1
        self.is_mask          :bool = False
        self.influences_count :int  = 0

        # --- dirty
        self.isDirty                  : bool = True
        self.isDirty_weights          : bool = True
        self.isDirty_geoMatrix        : bool = True
        self.isDirty_inputGeometry    : bool = True
        self.isDirty_bindPreMatrix    : bool = True
        self.isDirty_influencesMatrix : bool = True

        # --------------------
        self._geo_matrix             : om1.MMatrix = None
        self._get_matrix_i           : om1.MMatrix = None
        self._geo_matrix_is_identity : bool        = True

        self._bindPreMatrix_buffer       : BufferManager = None
        self._influencesMatrix_buffer    : BufferManager = None
        self._rotateMatrix_buffer        : BufferManager = None
        self._translateVector_buffer     : BufferManager = None

        self._skinWeights = None
        # fmt:on

    def postConstructor(self):
        """
        - 节点创建后 立刻通过api获取这些常用api对象 避免后续频繁调用api开销
        - 绑定节点实例 注册到全局 方便别的节点调用。
        """
        # fmt:off
        # --- maya
        self.mObject            = self.thisMObject()
        self.mFnDep             = om1.MFnDependencyNode(self.mObject)
        self.hashCode           = om1.MObjectHandle(self.mObject).hashCode()
        self.plug_refresh       = om1.MPlug(self.mObject, self.aRefresh)
        # --- context
        self.mesh_context       = MeshTopologyContext()
        self.brush_context      = BrushHitContext()
        self.weights_manager    = WeightsManager(self)
        # fmt:on

        _cRegistry.SkinRegistry.register(self.mObject, self)

    def setDirty(self):
        """
        - 用于笔刷调用 提醒 `Deform` 更新权重
        """
        self.plug_refresh.setInt(self.plug_refresh.asInt() + 1)
        self.isDirty_weights = True

    def setDependentsDirty(self, plug: om1.MPlug, dirtyPlugArray: om1.MPlugArray):
        """DG模式下脏数据标签 性能优化"""
        # --- weights
        if plug in (self.aWeights,
                    self.aLayerCompound,
                    self.aLayerMask,
                    self.aLayerWeights,
                    self.aLayerEnabled,
                    self.aRefresh):  # fmt:skip
            self.isDirty = True
            self.isDirty_weights = True
        # envelope
        elif plug == ompx.cvar.MPxGeometryFilter_envelope:
            self.isDirty = True
        # --- matrix
        elif plug == self.aInfluenceMatrix:
            self.isDirty = True
            self.isDirty_influencesMatrix = True
        # --- bind pre matrix
        elif plug == self.aBindPreMatrix:
            self.isDirty = True
            self.isDirty_bindPreMatrix = True
        # --- geo transform matrix
        elif plug == self.aGeomMatrix:
            self.isDirty = True
            self.isDirty_geoMatrix = True
        # --- input geometry
        elif plug == ompx.cvar.MPxGeometryFilter_inputGeom or \
             plug == ompx.cvar.MPxGeometryFilter_input:  # fmt:skip
            self.isDirty = True
            self.isDirty_inputGeometry = True

        return super().setDependentsDirty(plug, dirtyPlugArray)

    def preEvaluation(self, context: om1.MDGContext, evaluationNode: om1.MEvaluationNode):
        """并行模式下脏数据标签 性能优化"""
        if context.isNormal():
            # --- input geometry
            if (evaluationNode.dirtyPlugExists(ompx.cvar.MPxGeometryFilter_inputGeom)
                or evaluationNode.dirtyPlugExists(ompx.cvar.MPxGeometryFilter_input)
                or evaluationNode.dirtyPlugExists(ompx.cvar.MPxGeometryFilter_envelope)):  # fmt:skip
                self.isDirty = True
                self.isDirty_inputGeometry = True

            # --- matrix
            if evaluationNode.dirtyPlugExists(self.aInfluenceMatrix):
                self.isDirty = True
                self.isDirty_influencesMatrix = True

            # --- geo transform matrix
            if evaluationNode.dirtyPlugExists(self.aGeomMatrix):
                self.isDirty = True
                self.isDirty_geoMatrix = True

            # --- bind pre matrix
            if evaluationNode.dirtyPlugExists(self.aBindPreMatrix):
                self.isDirty = True
                self.isDirty_bindPreMatrix = True

            # --- weights
            if (   evaluationNode.dirtyPlugExists(self.aWeights)
                or evaluationNode.dirtyPlugExists(self.aLayerCompound)
                or evaluationNode.dirtyPlugExists(self.aLayerMask)
                or evaluationNode.dirtyPlugExists(self.aLayerWeights)
                or evaluationNode.dirtyPlugExists(self.aLayerEnabled)
                or evaluationNode.dirtyPlugExists(self.aRefresh)):  # fmt:skip
                self.isDirty = True
                self.isDirty_weights = True

        return super().preEvaluation(context, evaluationNode)

    def update_topology(self, mFnMesh: om1.MFnMesh):
        """
        提取并更新基础物理拓扑数据与 CSR 邻接表。

        仅在首次解算或模型拓扑 点数/面数 发生变化时执行
        为笔刷、渲染及平滑解算提供静态的拓扑数据结构。

        Args:
            mFnMesh (om1.MFnMesh): 输入的 Maya 网格函数集对象。

        Modifies:
            全面更新 self.mesh_context 中的所有计数器、基础索引和 CSR 邻接表缓存。
        """
        current_vertex_count = mFnMesh.numVertices()
        current_edge_count = mFnMesh.numEdges()
        current_polygon_count = mFnMesh.numPolygons()

        if self.brush_engine is not None and self.mesh_context.vertex_positions is not None:
            _buffer = self.mesh_context.vertex_positions.reshape((current_vertex_count, 3))
            self.brush_engine.update_vertex_positions(_buffer.view)

        # CHECK
        if (self.mesh_context.vertex_count == current_vertex_count 
            and self.mesh_context.edge_count == current_edge_count
            and self.mesh_context.polygon_count == current_polygon_count 
            and self.mesh_context.v2v_offsets is not None):  # fmt:skip
            # 通过检查 证明topology没有变化 直接跳过
            return

        # UPDATE TOPOLOGY
        self.mesh_context.vertex_count = current_vertex_count
        self.mesh_context.edge_count = current_edge_count
        self.mesh_context.polygon_count = current_polygon_count

        # --- Triangle Indices
        tri_counts = om1.MIntArray()
        tri_indices = om1.MIntArray()
        mFnMesh.getTriangles(tri_counts, tri_indices)
        tri_list = list(tri_indices)
        self.mesh_context.triangle_count = len(tri_list) // 3
        self.mesh_context.triangle_indices = BufferManager.from_list(tri_list, "i")

        # --- Quad Edge Indices
        quad_edge_list = [0] * (current_edge_count * 2)
        util = om1.MScriptUtil()
        ptr = util.asInt2Ptr()
        for i in range(current_edge_count):
            mFnMesh.getEdgeVertices(i, ptr)
            quad_edge_list[i * 2] = om1.MScriptUtil.getInt2ArrayItem(ptr, 0, 0)
            quad_edge_list[i * 2 + 1] = om1.MScriptUtil.getInt2ArrayItem(ptr, 0, 1)
        self.mesh_context.quad_edge_indices = BufferManager.from_list(quad_edge_list, "i")

        # --- Edge Indices
        unique_edges_ctypes = cTopology.compute_unique_edge_indices(self.mesh_context.triangle_indices.view)
        self.mesh_context.edge_indices = BufferManager.from_ctypes(unique_edges_ctypes)

        # --- Vertex-to-Vertex CSR
        v2v_offsets_ctypes, v2v_indices_ctypes = cTopology.build_v2v_adjacency(self.mesh_context.vertex_count, self.mesh_context.edge_indices.view)
        self.mesh_context.v2v_offsets = BufferManager.from_ctypes(v2v_offsets_ctypes)
        self.mesh_context.v2v_indices = BufferManager.from_ctypes(v2v_indices_ctypes)

        # --- Vertex-to-Face CSR
        v2f_offsets_ctypes, v2f_indices_ctypes = cTopology.build_v2f_adjacency(self.mesh_context.vertex_count, self.mesh_context.triangle_indices.view)
        self.mesh_context.v2f_offsets = BufferManager.from_ctypes(v2f_offsets_ctypes)
        self.mesh_context.v2f_indices = BufferManager.from_ctypes(v2f_indices_ctypes)
        # ====== Brush Core Engine
        # 临时申请一个 buffer, 方便我们创建coreBrushEngine
        self.mesh_context.vertex_positions = BufferManager.allocate("f", (self.mesh_context.vertex_count, 3))
        self.brush_engine = CoreBrushEngine(
            self.mesh_context.vertex_positions.view,
            self.mesh_context.triangle_indices.reshape((self.mesh_context.triangle_count, 3)).view,
            self.mesh_context.v2v_offsets.view,
            self.mesh_context.v2v_indices.view,
            self.mesh_context.v2f_offsets.view,
            self.mesh_context.v2f_indices.view,
        )

    def get_active_paint_weights(self) -> memoryview | None:
        """
        直接从 manager 提取当前需要绘制的的权重视图。

        Returns:
            tuple: (layer_index, is_mask, influences_index, weights_view)
                - layer_index (int): layer index
                - is_mask (bool): is mask
                - influences_index (int): influences index
                - weights_view (memoryview): weights view
        """

        if self.weights_manager is None:
            return None

        manager = self.weights_manager

        handle = manager.get_handle(self.layer_index, self.is_mask)
        if handle is None:
            return self.layer_index, self.is_mask, self.influences_index, None

        _, inf_count, _, weights_view = handle.parse_raw_weights()

        if inf_count <= 0 or not weights_view:
            return self.layer_index, self.is_mask, self.influences_index, None

        # 2. 计算安全偏移量
        safe_idx = max(0, min(self.influences_index, inf_count - 1))

        return self.layer_index, self.is_mask, self.influences_index, weights_view[safe_idx::inf_count]

    @maya_profile(0, "Compute")
    def compute(self, plug, dataBlock):
        """
        很蛋疼的是`DG`模式下 如果`.outputGeometry[i]`输出给多个模型
        每个模型求值都会触发一次 `Deform` 函数 非常消耗性能 尤其是在绘制权重的时候
        一个输出给 maya geometry 一个输出给 权重颜色显示模型 会导致deform函数执行两次。
        所以前面配置了`setDependentsDirty`标记 只有在input的数据改变的时候 才会触发 `Deform` 函数。
        后续获取蒙皮后的模型数据 可以用任意方法求值 不会造成多余的 `Deform` 函数调用 以节约资源。
        """
        if not self.isDirty:
            return None

        # 获取当前绘制 layer 数据
        self.layer_index = dataBlock.inputValue(self.aCurrentPaintLayerIndex).asInt()
        self.is_mask = dataBlock.inputValue(self.aCurrentPaintMaskBool).asBool()
        self.influences_index = dataBlock.inputValue(self.aCurrentPaintInfluenceIndex).asInt() if not self.is_mask else 0

        res = super().compute(plug, dataBlock)
        self.isDirty = False
        return res

    @maya_profile(0, "Deform")
    def deform(self, dataBlock: om1.MDataBlock, geoIter, localToWorldMatrix, multiIndex):  # noqa: ARG002
        envelope = dataBlock.inputValue(ompx.cvar.MPxGeometryFilter_envelope).asFloat()

        with MayaNativeProfiler("in-geo-object", 2):
            input_handle = dataBlock.outputArrayValue(ompx.cvar.MPxGeometryFilter_input)
            input_handle.jumpToElement(multiIndex)
            _input_geom_obj = input_handle.outputValue().child(ompx.cvar.MPxGeometryFilter_inputGeom)
            input_geom_obj = _input_geom_obj.asMesh()

        with MayaNativeProfiler("out-geo-object", 5):
            output_handle = dataBlock.outputArrayValue(ompx.cvar.MPxGeometryFilter_outputGeom)
            output_handle.jumpToElement(multiIndex)
            _output_geom_obj = output_handle.outputValue()
            output_geom_obj = _output_geom_obj.asMesh()

        if input_geom_obj.isNull() or output_geom_obj.isNull():
            return

        with MayaNativeProfiler("fnMesh-in", 3):
            mFnMesh_in = om1.MFnMesh(input_geom_obj)
            vertex_count = mFnMesh_in.numVertices()
            rawPoints_original = BufferManager.from_ptr(int(mFnMesh_in.getRawPoints()), "f", (vertex_count * 3,))
        with MayaNativeProfiler("update-topology", 3):
            self.update_topology(mFnMesh_in)

        with MayaNativeProfiler("out-fnMesh", 6):
            mFnMesh_out = om1.MFnMesh(output_geom_obj)
            self.mesh_context.vertex_positions = BufferManager.from_ptr(int(mFnMesh_out.getRawPoints()), "f", (vertex_count * 3,))

        with MayaNativeProfiler("influences allocate", 4):
            influences_handle = dataBlock.inputArrayValue(self.aInfluenceMatrix)
            influences_count = influences_handle.elementCount()
            if self.influences_count != influences_count:
                self.influences_count = influences_count
                self._influencesMatrix_buffer = BufferManager.allocate("d", (influences_count, 16))
                self._rotateMatrix_buffer = BufferManager.allocate("f", (influences_count, 9))
                self._translateVector_buffer = BufferManager.allocate("f", (influences_count, 3))
                for b in range(influences_count):
                    for i in range(16):
                        self._influencesMatrix_buffer.view[b, i] = 1.0 if (i % 5 == 0) else 0.0

        with MayaNativeProfiler("influences matrix", 5):
            if (self.isDirty_influencesMatrix
                and influences_count > 0):  # fmt:skip
                dest_base_addr = self._influencesMatrix_buffer.ptr
                for i in range(influences_count):
                    influences_handle.jumpToArrayElement(i)
                    influence_idx = influences_handle.elementIndex()
                    src_addr = int(influences_handle.inputValue().asMatrix().this)
                    dest_addr = dest_base_addr + (influence_idx * 128)
                    ctypes.memmove(dest_addr, src_addr, 128)
                self.isDirty_influencesMatrix = False

        with MayaNativeProfiler("influences bind matrix", 6):
            if self.isDirty_bindPreMatrix:
                bind_data_obj = dataBlock.inputValue(self.aBindPreMatrix).data()
                if not bind_data_obj.isNull():
                    fn_bind_array = om1.MFnMatrixArrayData(bind_data_obj)
                    bind_m_array = fn_bind_array.array()
                    if bind_m_array.length() > 0:
                        addr_base = int(bind_m_array[0].this)
                        self._bindPreMatrix_buffer = BufferManager.from_ptr(addr_base, "d", (bind_m_array.length(), 16))
                    self.isDirty_bindPreMatrix = False

        with MayaNativeProfiler("geo matrix", 7):
            if self.isDirty_geoMatrix:
                self._geo_matrix = dataBlock.inputValue(self.aGeomMatrix).asMatrix()
                self._get_matrix_i = self._geo_matrix.inverse()
                self._geo_matrix_is_identity = self._geo_matrix.isEquivalent(om1.MMatrix.identity)
                self.isDirty_geoMatrix = False

        with MayaNativeProfiler("Update Weights", 2):
            if self.isDirty_weights:
                self.weights_manager.sync_layer_cache(dataBlock)
                self.weights_manager.execute_deferred_tasks()

                self.isDirty_weights = True

        with MayaNativeProfiler("Cython Cal matrix", 1):
            _, _, _, self._skinWeights = self.weights_manager.get_handle(-1, 0).parse_raw_weights()
            if self._skinWeights is None:
                return

            cSkinDeformCython.compute_deform_matrices(
                int(self._geo_matrix.this),
                int(self._get_matrix_i.this),
                self._bindPreMatrix_buffer.view,
                self._influencesMatrix_buffer.view,
                self._rotateMatrix_buffer.view,
                self._translateVector_buffer.view,
                self._geo_matrix_is_identity,
            )

        with MayaNativeProfiler("Cython Cal skin", 3):
            cSkinDeformCython.run_skinning_core(
                rawPoints_original.view,
                self.mesh_context.vertex_positions.view,
                self._skinWeights,
                self._rotateMatrix_buffer.view,
                self._translateVector_buffer.view,
                envelope,
            )

    @classmethod
    def nodeInitializer(cls):
        nAttr = om1.MFnNumericAttribute()
        tAttr = om1.MFnTypedAttribute()
        mAttr = om1.MFnMatrixAttribute()
        cAttr = om1.MFnCompoundAttribute()

        # --- geo matrix
        CythonSkinDeformer.aGeomMatrix = mAttr.create("geomMatrix", "gm")
        mAttr.setHidden(True)
        mAttr.setKeyable(False)
        # --- bind pre matrix
        CythonSkinDeformer.aBindPreMatrix = tAttr.create("bindPreMatrixArray", "bpm", om1.MFnData.kMatrixArray)
        tAttr.setHidden(True)
        # --- influences matrix
        CythonSkinDeformer.aInfluenceMatrix = mAttr.create("matrix", "bm")
        mAttr.setArray(True)
        mAttr.setHidden(True)
        mAttr.setUsesArrayDataBuilder(True)
        # --- weights
        CythonSkinDeformer.aWeights = tAttr.create("cWeights", "cw", om1.MFnData.kVectorArray)
        tAttr.setHidden(True)
        # --- layer weights children
        CythonSkinDeformer.aLayerEnabled = nAttr.create("layerEnabled", "le", om1.MFnNumericData.kBoolean, False)
        nAttr.setHidden(True)
        CythonSkinDeformer.aLayerWeights = tAttr.create("layerWeights", "lw", om1.MFnData.kVectorArray)
        tAttr.setHidden(True)
        CythonSkinDeformer.aLayerMask = tAttr.create("layerMask", "lm", om1.MFnData.kVectorArray)
        tAttr.setHidden(True)
        # --- layer comp
        CythonSkinDeformer.aLayerCompound = cAttr.create("layers", "lays")
        cAttr.setArray(True)
        cAttr.setHidden(True)
        cAttr.setUsesArrayDataBuilder(True)
        cAttr.addChild(CythonSkinDeformer.aLayerEnabled)
        cAttr.addChild(CythonSkinDeformer.aLayerWeights)
        cAttr.addChild(CythonSkinDeformer.aLayerMask)
        # --- refresh
        CythonSkinDeformer.aRefresh = nAttr.create("cRefresh", "cr", om1.MFnNumericData.kInt, False)
        nAttr.setKeyable(False)
        nAttr.setStorable(False)
        nAttr.setCached(False)
        # --- Paint current weights
        CythonSkinDeformer.aCurrentPaintLayerIndex = nAttr.create("currentPaintLayer", "cpl", om1.MFnNumericData.kInt, -1)
        nAttr.setMin(-1)
        nAttr.setKeyable(True)
        CythonSkinDeformer.aCurrentPaintInfluenceIndex = nAttr.create("currentPaintInfluence", "cpi", om1.MFnNumericData.kInt, 0)
        nAttr.setMin(0)
        nAttr.setKeyable(True)
        CythonSkinDeformer.aCurrentPaintMaskBool = nAttr.create("currentPaintMask", "cpm", om1.MFnNumericData.kBoolean, False)
        nAttr.setKeyable(True)
        # ====================================
        for attr in  (CythonSkinDeformer.aGeomMatrix, 
                      CythonSkinDeformer.aBindPreMatrix, 
                      CythonSkinDeformer.aInfluenceMatrix, 
                      CythonSkinDeformer.aWeights, 
                      CythonSkinDeformer.aLayerCompound, 
                      CythonSkinDeformer.aRefresh,
                      CythonSkinDeformer.aCurrentPaintLayerIndex,
                      CythonSkinDeformer.aCurrentPaintInfluenceIndex,
                      CythonSkinDeformer.aCurrentPaintMaskBool):  # fmt:skip
            CythonSkinDeformer.addAttribute(attr)
            CythonSkinDeformer.attributeAffects(attr, ompx.cvar.MPxGeometryFilter_outputGeom)
