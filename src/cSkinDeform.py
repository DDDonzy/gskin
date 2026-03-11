from __future__ import annotations
import ctypes

import maya.OpenMaya as om1  # type:ignore
import maya.OpenMayaMPx as ompx  # type:ignore

from . import cMemoryView
from .cWeightsManager2 import WeightsHandle, WeightsManager
from . import cSkinDeformCython
from . import _cRegistry

from ._profile import MicroProfiler


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
        "weights",
        "weights_handle"
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

        self.weights = None
        self.weights_handle = None
        self.weights_manager = None

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

    def deform(self, dataBlock: om1.MDataBlock, geoIter, localToWorldMatrix, multiIndex):
        envelope = dataBlock.inputValue(ompx.cvar.MPxGeometryFilter_envelope).asFloat()
        if envelope == 0.0:
            return

        input_handle = dataBlock.inputArrayValue(ompx.cvar.MPxGeometryFilter_input)
        input_handle.jumpToElement(multiIndex)
        input_geom_obj = input_handle.inputValue().child(ompx.cvar.MPxGeometryFilter_inputGeom).asMesh()

        output_handle = dataBlock.outputArrayValue(ompx.cvar.MPxGeometryFilter_outputGeom)
        output_handle.jumpToElement(multiIndex)
        output_geom_obj = output_handle.outputValue().asMesh()

        if input_geom_obj.isNull() or output_geom_obj.isNull():
            return

        mFnMesh_in = om1.MFnMesh(input_geom_obj)
        self.vertex_count = mFnMesh_in.numVertices()
        rawPoints_original_mgr = cMemoryView.CMemoryManager.from_ptr(int(mFnMesh_in.getRawPoints()), "f", (self.vertex_count * 3,))

        mFnMesh_out = om1.MFnMesh(output_geom_obj)
        self.rawPoints_output_mgr = cMemoryView.CMemoryManager.from_ptr(int(mFnMesh_out.getRawPoints()), "f", (self.vertex_count * 3,))

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

        if self._bindPreMatrix_is_dirty:
            bind_data_obj = dataBlock.inputValue(self.aBindPreMatrix).data()
            if not bind_data_obj.isNull():
                fn_bind_array = om1.MFnMatrixArrayData(bind_data_obj)
                bind_m_array = fn_bind_array.array()
                if bind_m_array.length() > 0:
                    addr_base = int(bind_m_array[0].this)
                    self._bindPreMatrix_mgr = cMemoryView.CMemoryManager.from_ptr(addr_base, "d", (bind_m_array.length(), 16))
                self._bindPreMatrix_is_dirty = False

        if self._geoMatrix_is_dirty:
            self._geo_matrix = dataBlock.inputValue(self.aGeomMatrix).asMatrix()
            self._get_matrix_i = self._geo_matrix.inverse()
            self._geo_matrix_is_identity = self._geo_matrix.isEquivalent(om1.MMatrix.identity)
            self._geoMatrix_is_dirty = False

        if self._weights_is_dirty:
            print("[Deform]: update weights")
            self.weights_manager.update_data(dataBlock)
            self.weights, _, _, _ = self.weights_manager.weights.get_weights()
            self._weights_is_dirty = False

        cSkinDeformCython.compute_deform_matrices(
            int(self._geo_matrix.this),
            int(self._get_matrix_i.this),
            self._bindPreMatrix_mgr.view,
            self._influencesMatrix_mgr.view,
            self._rotateMatrix_mgr.view,
            self._translateVector_mgr.view,
            self._geo_matrix_is_identity,
        )

        cSkinDeformCython.run_skinning_core(
            rawPoints_original_mgr.view,
            self.rawPoints_output_mgr.view,
            self.weights,
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
