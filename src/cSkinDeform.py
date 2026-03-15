# encoding=utf-8

from __future__ import annotations
import ctypes

import maya.OpenMaya as om1  # type:ignore
import maya.OpenMayaMPx as ompx  # type:ignore

from . import cMemoryView
from .cWeightsManager import WeightsManager
from . import cSkinDeformCython
from . import _cRegistry

# from ._profile import MicroProfiler, DeepProfiler, maya_profile, MayaNativeProfiler
from ._cProfilerCython import MayaNativeProfiler, maya_profile


class CythonSkinDeformer(ompx.MPxDeformerNode):
    __slots__ = (
        # ==========================================
        # 🟢 公有数据 (Public API) - 供外部(Display/Brush)读取
        # ==========================================
        "vertex_count",  # 模型顶点数
        "rawPoints_output_mgr",  # 变形后的顶点坐标物理内存块
        "influences_count",  # 骨骼/影响物总数
        "influences_locks_mgr",  # 骨骼锁定状态
        "hashCode",  # 节点唯一哈希
        "mObject",  # 变形器自身对象
        "mFnDep",  # 依赖图函数集
        "plug_refresh",
        "weights_manager"
        # ==========================================
        # 🔴 私有数据 (Private) - 仅供 deform 内部计算使用
        # ==========================================
        "_weights_is_dirty",
        "_influencesMatrix_is_dirty",
        "_bindPreMatrix_is_dirty",
        "_geoMatrix_is_dirty",
        "_geo_matrix",
        "_get_matrix_i",
        "_geo_matrix_is_identity",
        "_influencesMatrix_mgr",
        "_bindPreMatrix_mgr",
        "_rotateMatrix_mgr",
        "_translateVector_mgr",
        "_skinWeights",
    )

    aGeomMatrix = om1.MObject()
    aWeights = om1.MObject()
    aLayerWeights = om1.MObject()
    aLayerMask = om1.MObject()
    aLayerEnabled = om1.MObject()
    aLayerCompound = om1.MObject()
    aInfluenceMatrix = om1.MObject()
    aBindPreMatrix = om1.MObject()
    aRefresh = om1.MObject()

    _memory_task_queue = None
    """
    任务队列，外部修改dataBlock中的内存数据容易出问题
    通过`dispatch_memory_task`函数，把外面修改数据的函数，提交到队列，
    在 `deform` 函数中执行。
    """

    def __init__(self):
        super(CythonSkinDeformer, self).__init__()

        # --- 初始化公有数据 ---
        self.vertex_count: int = 0
        self.rawPoints_output_mgr: "cMemoryView.CMemoryManager" = None
        self.influences_count: int = 0
        self.influences_locks_mgr: "cMemoryView.CMemoryManager" = None
        self.hashCode: int = None
        self.mObject: om1.MObject = None
        self.mFnDep: om1.MFnDependencyNode = None
        self.plug_refresh: om1.MPlug = None

        # --- 初始化私有数据 ---
        self._weights_is_dirty: bool = True
        self._influencesMatrix_is_dirty: bool = True
        self._bindPreMatrix_is_dirty: bool = True
        self._geoMatrix_is_dirty: bool = True

        self._geo_matrix = om1.MMatrix()
        self._get_matrix_i = om1.MMatrix()
        self._geo_matrix_is_identity = True

        self._influencesMatrix_mgr: "cMemoryView.CMemoryManager" = None
        self._bindPreMatrix_mgr: "cMemoryView.CMemoryManager" = None
        self._rotateMatrix_mgr: "cMemoryView.CMemoryManager" = None
        self._translateVector_mgr: "cMemoryView.CMemoryManager" = None

        self.weights_manager = None
        self._skinWeights = None
        """"""

    def setDirty(self):
        """
        - 用于笔刷调用，提醒Deform，更新权重
        """

        self.plug_refresh.setInt((self.plug_refresh.asInt() + 1))
        self._weights_is_dirty = True

    def postConstructor(self):
        self.mObject = self.thisMObject()
        self.mFnDep = om1.MFnDependencyNode(self.mObject)
        self.hashCode = om1.MObjectHandle(self.mObject).hashCode()
        self.plug_refresh = om1.MPlug(self.mObject, self.aRefresh)
        self.weights_manager = WeightsManager(self)

        _cRegistry.SkinRegistry.register(self.mObject, self)

    def setDependentsDirty(self, plug, dirtyPlugArray):
        weights_plugs = (self.aWeights, self.aLayerCompound, self.aLayerMask, self.aLayerWeights, self.aLayerEnabled, self.aRefresh)
        if plug in weights_plugs:
            self._weights_is_dirty = True
        elif plug == self.aInfluenceMatrix:
            self._influencesMatrix_is_dirty = True
        elif plug == self.aBindPreMatrix:
            self._bindPreMatrix_is_dirty = True
        elif plug == self.aGeomMatrix:
            self._geoMatrix_is_dirty = True
        return super(CythonSkinDeformer, self).setDependentsDirty(plug, dirtyPlugArray)

    def preEvaluation(self, context, evaluationNode):
        if context.isNormal():
            if evaluationNode.dirtyPlugExists(self.aGeomMatrix):
                self._geoMatrix_is_dirty = True
            if evaluationNode.dirtyPlugExists(self.aInfluenceMatrix):
                self._influencesMatrix_is_dirty = True
            if evaluationNode.dirtyPlugExists(self.aBindPreMatrix):
                self._bindPreMatrix_is_dirty = True
            if (
                evaluationNode.dirtyPlugExists(self.aWeights)
                or evaluationNode.dirtyPlugExists(self.aLayerCompound)
                or evaluationNode.dirtyPlugExists(self.aLayerMask)
                or evaluationNode.dirtyPlugExists(self.aLayerWeights)
                or evaluationNode.dirtyPlugExists(self.aLayerEnabled)
                or evaluationNode.dirtyPlugExists(self.aRefresh)
            ):
                self._weights_is_dirty = True
        return super(CythonSkinDeformer, self).preEvaluation(context, evaluationNode)

    @maya_profile(0, "Deform")
    def deform(self, dataBlock: om1.MDataBlock, geoIter, localToWorldMatrix, multiIndex):
        with MayaNativeProfiler("Envelop", 1):
            envelope = dataBlock.inputValue(ompx.cvar.MPxGeometryFilter_envelope).asFloat()
            if envelope == 0.0:
                return

        with MayaNativeProfiler("in-geo", 2):
            with MayaNativeProfiler("in-dataHandle", 3):
                input_handle = dataBlock.inputArrayValue(ompx.cvar.MPxGeometryFilter_input)
                input_handle.jumpToElement(multiIndex)
                _input_geom_obj = input_handle.inputValue().child(ompx.cvar.MPxGeometryFilter_inputGeom)
                with MayaNativeProfiler("in-asMesh", 4):
                    input_geom_obj = _input_geom_obj.asMesh()
        with MayaNativeProfiler("out-geo", 5):
            with MayaNativeProfiler("out-dataHandle", 5):
                output_handle = dataBlock.outputArrayValue(ompx.cvar.MPxGeometryFilter_outputGeom)
                output_handle.jumpToElement(multiIndex)
                _output_geom_obj = output_handle.outputValue()
                with MayaNativeProfiler("out-asMesh", 4):
                    output_geom_obj = _output_geom_obj.asMesh()

            if input_geom_obj.isNull() or output_geom_obj.isNull():
                return

        with MayaNativeProfiler("fnMesh", 3):
            with MayaNativeProfiler("in-fnMesh", 7):
                mFnMesh_in = om1.MFnMesh(input_geom_obj)
                self.vertex_count = mFnMesh_in.numVertices()
                with MayaNativeProfiler("in-buildBuffer", 5):
                    rawPoints_original_mgr = cMemoryView.CMemoryManager.from_ptr(int(mFnMesh_in.getRawPoints()), "f", (self.vertex_count * 3,))
            with MayaNativeProfiler("out-fnMesh", 6):
                mFnMesh_out = om1.MFnMesh(output_geom_obj)
                with MayaNativeProfiler("out-buildBuffer", 4):
                    self.rawPoints_output_mgr = cMemoryView.CMemoryManager.from_ptr(int(mFnMesh_out.getRawPoints()), "f", (self.vertex_count * 3,))

        with MayaNativeProfiler("influences allocate", 4):
            influences_handle = dataBlock.inputArrayValue(self.aInfluenceMatrix)
            influences_count = influences_handle.elementCount()
            if self.influences_count != influences_count:
                self.influences_count = influences_count
                self._influencesMatrix_mgr = cMemoryView.CMemoryManager.allocate("d", (influences_count, 16))
                self._rotateMatrix_mgr = cMemoryView.CMemoryManager.allocate("f", (influences_count, 9))
                self._translateVector_mgr = cMemoryView.CMemoryManager.allocate("f", (influences_count, 3))
                for b in range(influences_count):
                    for i in range(16):
                        self._influencesMatrix_mgr.view[b, i] = 1.0 if (i % 5 == 0) else 0.0

        with MayaNativeProfiler("influences matrix", 5):
            if self._influencesMatrix_is_dirty:
                if influences_count > 0:
                    dest_base_addr = self._influencesMatrix_mgr.ptr
                    for i in range(influences_count):
                        influences_handle.jumpToArrayElement(i)
                        influence_idx = influences_handle.elementIndex()
                        src_addr = int(influences_handle.inputValue().asMatrix().this)
                        dest_addr = dest_base_addr + (influence_idx * 128)
                        ctypes.memmove(dest_addr, src_addr, 128)
                    self._influencesMatrix_is_dirty = False

        with MayaNativeProfiler("influences bind matrix", 6):
            if self._bindPreMatrix_is_dirty:
                bind_data_obj = dataBlock.inputValue(self.aBindPreMatrix).data()
                if not bind_data_obj.isNull():
                    fn_bind_array = om1.MFnMatrixArrayData(bind_data_obj)
                    bind_m_array = fn_bind_array.array()
                    if bind_m_array.length() > 0:
                        addr_base = int(bind_m_array[0].this)
                        self._bindPreMatrix_mgr = cMemoryView.CMemoryManager.from_ptr(addr_base, "d", (bind_m_array.length(), 16))
                    self._bindPreMatrix_is_dirty = False

        with MayaNativeProfiler("geo matrix", 7):
            if self._geoMatrix_is_dirty:
                self._geo_matrix = dataBlock.inputValue(self.aGeomMatrix).asMatrix()
                self._get_matrix_i = self._geo_matrix.inverse()
                self._geo_matrix_is_identity = self._geo_matrix.isEquivalent(om1.MMatrix.identity)
                self._geoMatrix_is_dirty = False

        with MayaNativeProfiler("Update Weights", 2):
            if self._weights_is_dirty:
                # 刷新数据池
                self.weights_manager.update_data(dataBlock)
                self._weights_is_dirty = False

            _weightsView = self.weights_manager.get_raw_weights(-1, 0)
            if _weightsView is None:
                return
            _, _, _, self._skinWeights = self.weights_manager.parse_raw_weights(_weightsView)

        with MayaNativeProfiler("Cython Cal matrix", 1):
            cSkinDeformCython.compute_deform_matrices(
                int(self._geo_matrix.this),
                int(self._get_matrix_i.this),
                self._bindPreMatrix_mgr.view,
                self._influencesMatrix_mgr.view,
                self._rotateMatrix_mgr.view,
                self._translateVector_mgr.view,
                self._geo_matrix_is_identity,
            )

        with MayaNativeProfiler("Cython Cal skin", 3):
            cSkinDeformCython.run_skinning_core(
                rawPoints_original_mgr.view,
                self.rawPoints_output_mgr.view,
                self._skinWeights,
                self._rotateMatrix_mgr.view,
                self._translateVector_mgr.view,
                envelope,
            )

    @classmethod
    def nodeInitializer(cls):
        tAttr, mAttr, nAttr, cAttr = om1.MFnTypedAttribute(), om1.MFnMatrixAttribute(), om1.MFnNumericAttribute(), om1.MFnCompoundAttribute()
        cls.aGeomMatrix = mAttr.create("geomMatrix", "gm")
        mAttr.setHidden(True)
        mAttr.setKeyable(False)
        cls.aWeights = tAttr.create("cWeights", "cw", om1.MFnData.kMesh)
        tAttr.setHidden(True)
        cls.aInfluenceMatrix = mAttr.create("matrix", "bm")
        mAttr.setArray(True)
        mAttr.setHidden(True)
        mAttr.setUsesArrayDataBuilder(True)
        cls.aBindPreMatrix = tAttr.create("bindPreMatrixArray", "bpm", om1.MFnData.kMatrixArray)
        tAttr.setHidden(True)
        cls.aLayerWeights = tAttr.create("layerWeights", "lw", om1.MFnData.kMesh)
        tAttr.setHidden(True)
        cls.aLayerMask = tAttr.create("layerMask", "lm", om1.MFnData.kMesh)
        tAttr.setHidden(True)
        cls.aLayerEnabled = nAttr.create("layerEnabled", "le", om1.MFnNumericData.kBoolean, False)
        nAttr.setHidden(True)
        cls.aLayerCompound = cAttr.create("layers", "lays")
        cAttr.setArray(True)
        cAttr.setHidden(True)
        cAttr.setUsesArrayDataBuilder(True)
        cAttr.addChild(cls.aLayerEnabled)
        cAttr.addChild(cls.aLayerWeights)
        cAttr.addChild(cls.aLayerMask)

        cls.aRefresh = nAttr.create("cRefresh", "cr", om1.MFnNumericData.kInt, False)
        nAttr.setKeyable(False)  # 不要让它出现在通道盒里
        nAttr.setStorable(False)  # 💥 告诉 Maya 这个属性不需要存进文件
        nAttr.setCached(False)
        for attr in [cls.aGeomMatrix, cls.aWeights, cls.aInfluenceMatrix, cls.aBindPreMatrix, cls.aLayerCompound, cls.aRefresh]:
            cls.addAttribute(attr)
        outputGeom = ompx.cvar.MPxGeometryFilter_outputGeom
        for attr in [cls.aGeomMatrix, cls.aWeights, cls.aInfluenceMatrix, cls.aBindPreMatrix, cls.aLayerCompound, cls.aRefresh]:
            cls.attributeAffects(attr, outputGeom)
