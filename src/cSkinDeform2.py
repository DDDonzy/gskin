from __future__ import annotations
import ctypes

import maya.OpenMaya as OpenMaya  # type:ignore
import maya.OpenMayaMPx as OpenMayaMPx  # type:ignore

from . import _cRegistry
from . import cSkinDeformCython


class CythonSkinDeformer(OpenMayaMPx.MPxDeformerNode):
    # fmt:off
    aGeomMatrix          = OpenMaya.MObject()
    aBindPreMatrix       = OpenMaya.MObject()
    aInfluenceMatrix     = OpenMaya.MObject()

    aWeights             = OpenMaya.MObject()

    aLayer               = OpenMaya.MObject()
    aMaskWeights         = OpenMaya.MObject()
    aLockMasks           = OpenMaya.MObject()

    aLayerCompound       = OpenMaya.MObject()
    aLayerName           = OpenMaya.MObject()
    aLayerEnabled        = OpenMaya.MObject()
    aLayerWeights        = OpenMaya.MObject()
    aLayerLockInfluences = OpenMaya.MObject()

    aForceDirty          = OpenMaya.MObject()

    aCurrentPaintLayerIndex     = OpenMaya.MObject()
    aCurrentPaintInfluenceIndex = OpenMaya.MObject()
    aCurrentPaintMaskBool       = OpenMaya.MObject()

    aInputGeometry : OpenMaya.MObject  = OpenMayaMPx.cvar.MPxGeometryFilter_inputGeom
    aOutputGeometry : OpenMaya.MObject = OpenMayaMPx.cvar.MPxGeometryFilter_outputGeom
    # fmt:on

    def __init__(self):
        super().__init__()

        # fmt:off
        # --- maya
        self.hashCode       : int                        = None
        self.mObject        : OpenMaya.MObject           = None
        self.mFnDep         : OpenMaya.MFnDependencyNode = None
        self.plug_refresh   : OpenMaya.MPlug             = None


        # cache 缓存实例节点数据, 避免每次计算都调用api
        # --- current paint data
        self.influences_count : int           = 0

        self.layer_index      :int  = -1
        self.is_mask          :bool = False
        self.influences_count :int  = 0

        self._geo_matrix             : OpenMaya.MMatrix = None
        self._get_matrix_i           : OpenMaya.MMatrix = None
        self._geo_matrix_is_identity : bool             = True

        self._bindPreMatrix_buffer       : memoryview = None
        self._influencesMatrix_buffer    : memoryview = None
        self._rotateMatrix_buffer        : memoryview = None
        self._translateVector_buffer     : memoryview = None
        # dirty flag
        self.is_dirty = True

        # fmt:on

    def postConstructor(self):
        """
        - 节点创建后 立刻通过api获取这些常用api对象 避免后续频繁调用api开销
        - 绑定节点实例 注册到全局 方便别的节点调用。
        """
        # fmt:off
        self.mObject       = self.thisMObject()
        self.mFnDependNode = OpenMaya.MFnDependencyNode(self.mObject)
        self.hashCode      = OpenMaya.MObjectHandle(self.mObject).hashCode()
        # fmt:on

        _cRegistry.SkinRegistry.register(self.mObject, self)

    def set_dirty(self):
        """
        标记节点位脏, 在需要的时候Maya自行求值.

        通常情况需要搭配 `set_dirty` 先标记为脏, 再使用 `pull_output` 强行求值.

        此方法只能外部调用, 节点内部严禁调用.
        """
        self.is_dirty = True
        self.plug_refresh.setInt(self.plug_refresh.asInt() + 1)

    def pull_output(self):
        """
        强行向 Maya 索要输出数据 以触发 deform 求值 适用于笔刷修改权重后需要立刻更新模型的场景.

        通常情况需要搭配 `set_dirty` 先标记为脏, 再使用 `pull_output` 强行求值.
        此方法只能外部调用, 节点内部严禁调用.
        """
        self.set_dirty()
        mPlug: OpenMaya.MPlug = OpenMaya.MPlug(self.mObject, self.aOutputGeometry).elementByLogicalIndex(0)
        mPlug.asMObject()
        return True

    def setDependentsDirty(self, plug: OpenMaya.MPlug, dirtyPlugArray: OpenMaya.MPlugArray):
        """DG模式下脏数据标签 性能优化"""

        return super().setDependentsDirty(plug, dirtyPlugArray)

    def preEvaluation(self, context: OpenMaya.MDGContext, evaluationNode: OpenMaya.MEvaluationNode):
        """并行模式下脏数据标签 性能优化"""

        return super().preEvaluation(context, evaluationNode)

    def compute(self, plug, dataBlock):
        """
        很蛋疼的是`DG`模式下 如果`.outputGeometry[i]`输出给多个模型
        每个模型求值都会触发一次 `Deform` 函数 非常消耗性能 尤其是在绘制权重的时候
        一个输出给 maya geometry 一个输出给 权重颜色显示模型 会导致deform函数执行两次。
        所以前面配置了`setDependentsDirty`标记 只有在input的数据改变的时候 才会触发 `Deform` 函数。
        后续获取蒙皮后的模型数据 可以用任意方法求值 不会造成多余的 `Deform` 函数调用 以节约资源。
        """
        if self.is_dirty is False:
            return None

        res = super().compute(plug, dataBlock)
        # self.is_dirty = False
        print("compute")
        return res

    def deform(self, dataBlock: OpenMaya.MDataBlock, geoIter, localToWorldMatrix, multiIndex):
        print("deform")
        return

        self.isDirty_brushFastPreview = False

        self.envelope_value = dataBlock.inputValue(OpenMayaMPx.cvar.MPxGeometryFilter_envelope).asFloat()

        input_handle = dataBlock.inputArrayValue(OpenMayaMPx.cvar.MPxGeometryFilter_input)
        input_handle.jumpToElement(multiIndex)
        _input_geom_obj = input_handle.outputValue().child(OpenMayaMPx.cvar.MPxGeometryFilter_inputGeom)
        input_geom_obj = _input_geom_obj.asMesh()

        output_handle = dataBlock.outputArrayValue(OpenMayaMPx.cvar.MPxGeometryFilter_outputGeom)
        output_handle.jumpToElement(multiIndex)
        _output_geom_obj = output_handle.outputValue()
        output_geom_obj = _output_geom_obj.asMesh()

        if input_geom_obj.isNull() or output_geom_obj.isNull():
            return

        mFnMesh_out = OpenMaya.MFnMesh(output_geom_obj)
        vertex_count = mFnMesh_out.numVertices()
        self.mesh_context.vertex_positions = BufferManager.from_ptr(int(mFnMesh_out.getRawPoints()), "f", (vertex_count * 3,))

        self.update_topology(mFnMesh_out)

        if (self.isDirty_inputGeometry is True 
            or self.rawPoints_original is None 
            or self.rawPoints_original.shape[0] != vertex_count * 3):  # fmt:off
            if self.rawPoints_original is None or self.rawPoints_original.shape[0] != vertex_count * 3:
                # 没有原始数据 or topology 变了才申请内存
                self.rawPoints_original = BufferManager.allocate("f", (vertex_count * 3,))
            #  将原始数据复制到 original 内存中
            mFnMesh_in = OpenMaya.MFnMesh(input_geom_obj)
            ctypes.memmove(
                self.rawPoints_original.ptr,
                int(mFnMesh_in.getRawPoints()),
                vertex_count * 3 * ctypes.sizeof(ctypes.c_float),
            )
            self.isDirty_inputGeometry = False

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

        if self.isDirty_bindPreMatrix:
            bind_data_obj = dataBlock.inputValue(self.aBindPreMatrix).data()
            if not bind_data_obj.isNull():
                fn_bind_array = OpenMaya.MFnMatrixArrayData(bind_data_obj)
                bind_m_array = fn_bind_array.array()
                if bind_m_array.length() > 0:
                    addr_base = int(bind_m_array[0].this)
                    self._bindPreMatrix_buffer = BufferManager.from_ptr(addr_base, "d", (bind_m_array.length(), 16))
                self.isDirty_bindPreMatrix = False

        if self.isDirty_geoMatrix:
            self._geo_matrix = dataBlock.inputValue(self.aGeomMatrix).asMatrix()
            self._get_matrix_i = self._geo_matrix.inverse()
            self._geo_matrix_is_identity = self._geo_matrix.isEquivalent(OpenMaya.MMatrix.identity)
            self.isDirty_geoMatrix = False

        if self.isDirty_weights:
            # 更新权重layer数据
            self.weights_manager.sync_layer_cache(dataBlock)
            # 执行异步权重修改
            # 为了优化性能, 权重修改是放在deform内部进行, 通过dataBlock拿到rawPoint直接修改内存, 而非外部直接修改mPlug
            self.task_manager.execute_tasks()

            self.isDirty_weights = True

        _, _, _, self._skinWeights = self.weights_manager.get_weights_handle(-1).parse_raw_weights()
        if self._skinWeights is None:
            return

        cSkinDeformCython.cal_deform_matrices(
            int(self._geo_matrix.this),
            int(self._get_matrix_i.this),
            self._bindPreMatrix_buffer.view,
            self._influencesMatrix_buffer.view,
            self._rotateMatrix_buffer.view,
            self._translateVector_buffer.view,
            self._geo_matrix_is_identity,
        )

        cSkinDeformCython.run_skinning_core(
            self.rawPoints_original.view,
            self.mesh_context.vertex_positions.view,
            self._skinWeights,
            self._rotateMatrix_buffer.view,
            self._translateVector_buffer.view,
            self.envelope_value,
        )

    def fast_preview_deform(self, hit_indices: memoryview | None = None, hit_count: int = 0):
        """
        局部蒙皮算法, 专供笔刷调用
        根据笔刷的 hit_indices 和 hit_count 来局部计算蒙皮,
        不唤醒 deform 函数, 不触发maya dg, 直接通知渲染节点更新
        """

        if hit_count <= 0 or hit_indices is None:
            return

        # 检查上一帧的缓存是否真的存在避免还没经过 deform 就强行刷笔刷
        if (   self._skinWeights            is None 
            or self.rawPoints_original      is None 
            or self._rotateMatrix_buffer    is None 
            or self._translateVector_buffer is None):  # fmt:skip
            return

        valid_indices_view = hit_indices[:hit_count]

        # 局部极速蒙皮 复用上一帧的矩阵和源点
        cSkinDeformCython.run_partial_skinning_core(
            valid_indices_view,
            self.rawPoints_original.view,
            self.mesh_context.vertex_positions.view,
            self._skinWeights,
            self._rotateMatrix_buffer.view,
            self._translateVector_buffer.view,
            self.envelope_value,
        )
        # 更新脏标记
        self.isDirty_brushFastPreview = True

    @classmethod
    def nodeInitializer(cls):
        # fmt:off
        nAttr = OpenMaya.MFnNumericAttribute()
        tAttr = OpenMaya.MFnTypedAttribute()
        mAttr = OpenMaya.MFnMatrixAttribute()
        cAttr = OpenMaya.MFnCompoundAttribute()

        cls.aGeomMatrix      = mAttr.create("geomMatrix", "gm")
        cls.aBindPreMatrix   = tAttr.create("bindPreMatrixArray", "bpm", OpenMaya.MFnData.kMatrixArray)
        cls.aInfluenceMatrix = mAttr.create("matrix", "bm")
        mAttr.setArray(True)
        mAttr.setUsesArrayDataBuilder(True)
        # --- weights
        cls.aWeights             = tAttr.create("weightsData", "wd", OpenMaya.MFnData.kVectorArray)
        cls.aLayerCompound       = cAttr.create("layerData", "lds")
        cls.aLayerName           = tAttr.create("layerName", "ln", OpenMaya.MFnData.kString)
        cls.aLayerEnabled        = nAttr.create("layerEnabled", "le", OpenMaya.MFnNumericData.kBoolean, False)
        cls.aLayerWeights        = tAttr.create("layerWeightsData", "lwd", OpenMaya.MFnData.kVectorArray)
        cls.aLayerLockInfluences = tAttr.create("layerLockInfluences", "lli", OpenMaya.MFnData.kIntArray)
        cAttr.setArray(True)
        cAttr.setUsesArrayDataBuilder(True)
        cAttr.addChild(cls.aLayerName)
        cAttr.addChild(cls.aLayerEnabled)
        cAttr.addChild(cls.aLayerWeights)
        cAttr.addChild(cls.aLayerLockInfluences)
        # --- layer compound
        cls.aLayer       = cAttr.create("layers", "lyd")
        cls.aMaskWeights = tAttr.create("layersMaskData", "lmd", OpenMaya.MFnData.kVectorArray)
        cls.aLockMasks   = tAttr.create("layersLockMask", "llm", OpenMaya.MFnData.kIntArray)
        cAttr.addChild(cls.aMaskWeights)
        cAttr.addChild(cls.aLockMasks)
        cAttr.addChild(cls.aLayerCompound)

        # --- refresh
        cls.aForceDirty = nAttr.create("forceDirty", "di", OpenMaya.MFnNumericData.kInt, False)
        nAttr.setChannelBox(True)
        nAttr.setStorable(False)
        # --- Paint current weights
        cls.aCurrentPaintLayerIndex = nAttr.create("currentPaintLayer", "cpl", OpenMaya.MFnNumericData.kInt, -1)
        nAttr.setMin(-1)
        nAttr.setKeyable(True)
        cls.aCurrentPaintInfluenceIndex = nAttr.create("currentPaintInfluence", "cpi", OpenMaya.MFnNumericData.kInt, 0)
        nAttr.setMin(0)
        nAttr.setKeyable(True)
        cls.aCurrentPaintMaskBool = nAttr.create("currentPaintMask", "cpm", OpenMaya.MFnNumericData.kBoolean, False)
        nAttr.setKeyable(True)
        # ====================================
        for attr in  (cls.aGeomMatrix, 
                      cls.aBindPreMatrix, 
                      cls.aInfluenceMatrix, 

                      cls.aWeights,
                      cls.aLayer, 

                      cls.aForceDirty, 
                      
                      cls.aCurrentPaintLayerIndex,
                      cls.aCurrentPaintInfluenceIndex,
                      cls.aCurrentPaintMaskBool):  # fmt:skip
            cls.addAttribute(attr)
            cls.attributeAffects(attr, cls.aOutputGeometry)


        # fmt:on
