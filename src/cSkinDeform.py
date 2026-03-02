from __future__ import annotations
import typing
import ctypes

import maya.OpenMaya as om1
import maya.OpenMayaMPx as ompx

from . import cMemoryView
from . import cWeightsHandle as CWH
from . import cSkinDeformCython
from . import _cRegistry

if typing.TYPE_CHECKING:
    from . import cWeightsHandle


class SkinDeformerContext:
    __slots__ = (
        "geo_matrix",
        "influences_count",
        "influences_locks_mgr",
        "_influencesMatrix_mgr",
        "_bindPreMatrix_mgr",
        "_rotateMatrix_mgr",
        "_translateVector_mgr",
        "weightsLayer",
        "hashCode",
        "mObject",
        "mFnDep",
    )

    def __init__(self) -> None:
        self.geo_matrix: om1.MMatrix = None
        self.influences_count: int = 0
        self.influences_locks_mgr: "cMemoryView.CMemoryManager" = None
        self._influencesMatrix_mgr: "cMemoryView.CMemoryManager" = None
        self._bindPreMatrix_mgr: "cMemoryView.CMemoryManager" = None
        self._rotateMatrix_mgr: "cMemoryView.CMemoryManager" = None
        self._translateVector_mgr: "cMemoryView.CMemoryManager" = None
        self.weightsLayer: typing.Dict[int, "cWeightsHandle.WeightsLayerData"] = {}
        self.hashCode: int = None
        self.mObject: om1.MObject = None
        self.mFnDep: om1.MFnDependencyNode = None


class CythonSkinDeformer(ompx.MPxDeformerNode):
    __slots__ = (
        "skin_context",
        "_weights_is_dirty",
        "_influencesMatrix_is_dirty",
        "_bindPreMatrix_is_dirty",
        "_geoMatrix_is_dirty",
        "_geo_matrix",
        "_get_matrix_i",
        "_geo_matrix_is_identity",
        "mObject",
    )

    aGeomMatrix = om1.MObject()
    aWeights = om1.MObject()
    aWeightsLayer = om1.MObject()
    aWeightsLayerMask = om1.MObject()
    aWeightsLayerEnabled = om1.MObject()
    aWeightsLayerCompound = om1.MObject()
    aInfluenceMatrix = om1.MObject()
    aBindPreMatrix = om1.MObject()

    def __init__(self):
        super(CythonSkinDeformer, self).__init__()
        self.skin_context = SkinDeformerContext()
        self._weights_is_dirty: bool = True
        self._influencesMatrix_is_dirty: bool = True
        self._bindPreMatrix_is_dirty: bool = True
        self._geoMatrix_is_dirty: bool = True
        self._geo_matrix = om1.MMatrix()
        self._get_matrix_i = om1.MMatrix()
        self._geo_matrix_is_identity = True

    def postConstructor(self):
        self.mObject = self.thisMObject()
        self.skin_context.mObject = self.mObject
        self.skin_context.mFnDep = om1.MFnDependencyNode(self.skin_context.mObject)
        self.skin_context.hashCode = om1.MObjectHandle(self.skin_context.mObject).hashCode()
        _cRegistry.SkinRegistry.register(self.mObject, self)

    def setDependentsDirty(self, plug, dirtyPlugArray):
        weights_plugs = (self.aWeights, self.aWeightsLayerCompound, self.aWeightsLayerMask, self.aWeightsLayer, self.aWeightsLayerEnabled)
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
            if evaluationNode.dirtyPlugExists(self.aWeights) or evaluationNode.dirtyPlugExists(self.aWeightsLayerCompound) or evaluationNode.dirtyPlugExists(self.aWeightsLayerMask) or evaluationNode.dirtyPlugExists(self.aWeightsLayer) or evaluationNode.dirtyPlugExists(self.aWeightsLayerEnabled):
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
        vertex_count = mFnMesh_in.numVertices()
        rawPoints_original_mgr = cMemoryView.CMemoryManager.from_ptr(int(mFnMesh_in.getRawPoints()), "f", (vertex_count * 3,))

        mFnMesh_out = om1.MFnMesh(output_geom_obj)
        rawPoints_output_mgr = cMemoryView.CMemoryManager.from_ptr(int(mFnMesh_out.getRawPoints()), "f", (vertex_count * 3,))

        influences_handle = dataBlock.inputArrayValue(self.aInfluenceMatrix)
        influences_count = influences_handle.elementCount()

        if self.skin_context.influences_count != influences_count:
            self.skin_context.influences_count = influences_count
            self.skin_context._influencesMatrix_mgr = cMemoryView.CMemoryManager.allocate("d", (influences_count, 16))
            self.skin_context._rotateMatrix_mgr = cMemoryView.CMemoryManager.allocate("f", (influences_count, 9))
            self.skin_context._translateVector_mgr = cMemoryView.CMemoryManager.allocate("f", (influences_count, 3))
            for b in range(influences_count):
                for i in range(16):
                    self.skin_context._influencesMatrix_mgr.view[b, i] = 1.0 if (i % 5 == 0) else 0.0

        if self._influencesMatrix_is_dirty:
            if influences_count > 0:
                dest_base_addr = self.skin_context._influencesMatrix_mgr.ptr
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
                    self.skin_context._bindPreMatrix_mgr = cMemoryView.CMemoryManager.from_ptr(addr_base, "d", (bind_m_array.length(), 16))
                self._bindPreMatrix_is_dirty = False

        if self._geoMatrix_is_dirty:
            self._geo_matrix = dataBlock.inputValue(self.aGeomMatrix).asMatrix()
            self._get_matrix_i = self._geo_matrix.inverse()
            self._geo_matrix_is_identity = self._geo_matrix.isEquivalent(om1.MMatrix.identity)
            self.skin_context.geo_matrix = self._geo_matrix
            self._geoMatrix_is_dirty = False

        if self._weights_is_dirty:
            self.skin_context.weightsLayer = self._get_weights_layers_data(dataBlock)
            self._weights_is_dirty = False

        if not self.skin_context.weightsLayer or not self.skin_context.weightsLayer[-1].weightsHandle.is_valid:
            return

        cSkinDeformCython.compute_deform_matrices(
            int(self._geo_matrix.this),
            int(self._get_matrix_i.this),
            self.skin_context._bindPreMatrix_mgr.view,
            self.skin_context._influencesMatrix_mgr.view,
            self.skin_context._rotateMatrix_mgr.view,
            self.skin_context._translateVector_mgr.view,
            self._geo_matrix_is_identity,
        )

        cSkinDeformCython.run_skinning_core(
            rawPoints_original_mgr.view,
            rawPoints_output_mgr.view,
            self.skin_context.weightsLayer[-1].weightsHandle.memory.view,
            self.skin_context._rotateMatrix_mgr.view,
            self.skin_context._translateVector_mgr.view,
            envelope,
        )

    def _get_weights_layers_data(self, dataBlock: om1.MDataBlock) -> dict[int, CWH.WeightsLayerData]:
        layer_data_dict = {}
        base_weights_val = dataBlock.inputValue(self.aWeights)
        base_weights_handle = CWH.WeightsHandle.from_data_handle(base_weights_val)
        layer_data_dict[-1] = CWH.WeightsLayerData(-1, True, base_weights_handle, None)
        layer_array_handle = dataBlock.inputArrayValue(self.aWeightsLayerCompound)
        for i in range(layer_array_handle.elementCount()):
            layer_array_handle.jumpToArrayElement(i)
            logical_idx = layer_array_handle.elementIndex()
            element_handle = layer_array_handle.inputValue()
            weights_handle = CWH.WeightsHandle.from_data_handle(element_handle.child(self.aWeightsLayer))
            mask_handle = CWH.WeightsHandle.from_data_handle(element_handle.child(self.aWeightsLayerMask))
            enabled = element_handle.child(self.aWeightsLayerEnabled).asBool()
            layer_data_dict[logical_idx] = CWH.WeightsLayerData(logical_idx, enabled, weights_handle, mask_handle)
        return layer_data_dict

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
        cls.aWeightsLayer = tAttr.create("cWeightsLayer", "cwl", om1.MFnData.kMesh)
        tAttr.setHidden(True)
        cls.aWeightsLayerMask = tAttr.create("cWeightsLayerMask", "cwlm", om1.MFnData.kMesh)
        tAttr.setHidden(True)
        cls.aWeightsLayerEnabled = nAttr.create("cWeightsLayerEnabled", "cwle", om1.MFnNumericData.kBoolean, False)
        nAttr.setHidden(True)
        cls.aWeightsLayerCompound = cAttr.create("cWeightsLayers", "cwls")
        cAttr.setArray(True)
        cAttr.setHidden(True)
        cAttr.setUsesArrayDataBuilder(True)
        cAttr.addChild(cls.aWeightsLayerEnabled)
        cAttr.addChild(cls.aWeightsLayer)
        cAttr.addChild(cls.aWeightsLayerMask)
        for attr in [cls.aGeomMatrix, cls.aWeights, cls.aInfluenceMatrix, cls.aBindPreMatrix, cls.aWeightsLayerCompound]:
            cls.addAttribute(attr)
        outputGeom = ompx.cvar.MPxGeometryFilter_outputGeom
        for attr in [cls.aGeomMatrix, cls.aWeights, cls.aInfluenceMatrix, cls.aBindPreMatrix, cls.aWeightsLayerCompound]:
            cls.attributeAffects(attr, outputGeom)
