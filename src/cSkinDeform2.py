from __future__ import annotations

import ctypes
from tkinter.filedialog import Open

from gskin.src import MWeightsHandle
import maya.OpenMaya as OpenMaya  # type:ignore
import maya.OpenMayaMPx as OpenMayaMPx  # type:ignore

from . import _cRegistry
from . import cSkinDeformCython
from . import cDirtyEvent
from .MTopologyContext import TopologyContext


class CSkinContext:
    # fmt:off
    input_mFnMesh : OpenMaya.MFnMesh
    output_mFnMesh: OpenMaya.MFnMesh

    current_paint_layer_index    : int
    current_paint_influence_index: int
    current_paint_mask_bool      : bool

    out_position : memoryview
    orig_position: memoryview
    skin_weights : MWeightsHandle.MWeightsHandle

    geo_matrix            : memoryview
    geo_matrix_i          : memoryview
    geo_matrix_is_identity: bool

    bind_pre_matrix  : memoryview
    influences_matrix: memoryview
    rotate_matrix    : memoryview
    translate_vector : memoryview

    topology: TopologyContext
    # fmt:on

    def __init__(self):
        # fmt:off
        self.current_paint_layer_index     = -1
        self.current_paint_influence_index = 0
        self.current_paint_mask_bool       = False

        self.skin_weights  = None

        self.geo_matrix             = None
        self.geo_matrix_i           = None
        self.geo_matrix_is_identity = True

        self.bind_pre_matrix   = None
        self.influences_matrix = None
        self.rotate_matrix     = None
        self.translate_vector  = None

        self.input_mesh = TopologyContext()
        self.output_mesh = TopologyContext()
        # fmt:on


class CSkinDeform(OpenMayaMPx.MPxDeformerNode):
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

    aInput :OpenMaya.MObject = OpenMayaMPx.cvar.MPxGeometryFilter_input
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

        self.ctx = CSkinContext()

        self.init_dirtyEvent()
        # fmt:on

    def init_dirtyEvent(self):
        # dirty flag
        # fmt:off
        self.event_update_mesh              = cDirtyEvent.DirtyEventHandler(self.aInputGeometry, self._update_mesh)
        self.event_update_influences_matrix = cDirtyEvent.DirtyEventHandler(self.aInfluenceMatrix, self._update_influences_matrix)
        self.event_update_bind_pre_matrix   = cDirtyEvent.DirtyEventHandler(self.aBindPreMatrix, self._update_bind_pre_matrix)
        self.event_update_geo_matrix        = cDirtyEvent.DirtyEventHandler(self.aGeomMatrix, self._update_geo_matrix)
        self.event_update_deform_matrix     = cDirtyEvent.DirtyEventHandler((self.aInfluenceMatrix, self.aBindPreMatrix, self.aGeomMatrix),
                                                                              self._update_deform_matrices,)  # fmt:skip
        self.event_update_paint_information = cDirtyEvent.DirtyEventHandler((self.aCurrentPaintLayerIndex, self.aCurrentPaintInfluenceIndex, self.aCurrentPaintMaskBool),
                                                                              self._update_paint_information)  # fmt:skip
        self.event_update_weights           = cDirtyEvent.DirtyEventHandler(self.aWeights, self._update_weights)
        # fmt:on

    def setDependentsDirty(self, plug: OpenMaya.MPlug, dirtyPlugArray: OpenMaya.MPlugArray):
        """DG模式下脏数据标签 性能优化"""
        self.event_update_mesh.sync_from_plug(plug)
        self.event_update_influences_matrix.sync_from_plug(plug)
        self.event_update_bind_pre_matrix.sync_from_plug(plug)
        self.event_update_geo_matrix.sync_from_plug(plug)
        self.event_update_deform_matrix.sync_from_plug(plug)
        self.event_update_paint_information.sync_from_plug(plug)
        self.event_update_weights.sync_from_plug(plug)
        return super().setDependentsDirty(plug, dirtyPlugArray)

    def preEvaluation(self, context: OpenMaya.MDGContext, evaluationNode: OpenMaya.MEvaluationNode):
        """并行模式下脏数据标签 性能优化"""
        self.event_update_mesh.sync_from_evaluation(evaluationNode)
        self.event_update_influences_matrix.sync_from_evaluation(evaluationNode)
        self.event_update_bind_pre_matrix.sync_from_evaluation(evaluationNode)
        self.event_update_geo_matrix.sync_from_evaluation(evaluationNode)
        self.event_update_deform_matrix.sync_from_evaluation(evaluationNode)
        self.event_update_paint_information.sync_from_evaluation(evaluationNode)
        self.event_update_weights.sync_from_evaluation(evaluationNode)
        return super().preEvaluation(context, evaluationNode)

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

    def _update_mesh(self, dataBlock: OpenMaya.MDataBlock, multiIndex: int):
        """
        Updata:
            - `self.ctx.input_mesh`
            - `self.ctx.output_mesh`
        """
        ctx = self.ctx

        input_handle = dataBlock.inputArrayValue(self.aInput)
        input_handle.jumpToElement(multiIndex)
        input_geom_obj = input_handle.outputValue().child(self.aInputGeometry).asMesh()
        ctx.input_mesh.update_fnMesh(OpenMaya.MFnMesh(input_geom_obj))

        output_handle = dataBlock.outputArrayValue(self.aOutputGeometry)
        output_handle.jumpToElement(multiIndex)
        output_geom_obj = output_handle.outputValue().asMesh()
        ctx.output_mesh.update_fnMesh(OpenMaya.MFnMesh(output_geom_obj))

        ctx.input_mesh.update_position()
        ctx.output_mesh.update_position()
        ctx.output_mesh.update_topology()
        print("update_input_mesh")

    def _update_influences_matrix(self, dataBlock: OpenMaya.MDataBlock):
        """
        Update:
            - `self.ctx.influences_matrix`
        """
        ctx = self.ctx

        influence_handle: OpenMaya.MDataHandle = dataBlock.inputArrayValue(self.aInfluenceMatrix)
        num_influences = influence_handle.elementCount()
        if num_influences == 0:
            return
        # 申请内存池
        # TODO 需要优化, 避免每帧都申请新内存池, 尽量复用
        _c = (ctypes.c_double * (num_influences * 16))()
        ctx.influences_matrix = memoryview(_c).cast("B").cast("d", (num_influences, 16))
        # 把maya数据填充到内存池
        influences_matrix_address = ctypes.addressof(ctx.influences_matrix.obj)
        for i in range(num_influences):
            influence_handle.jumpToArrayElement(i)
            i_matrix_address = int(influence_handle.inputValue().asMatrix().this)
            dst_address = (i * 128) + influences_matrix_address  # 128 = (4*4) * ctypes.sizeof(ctypes.c_double)
            ctypes.memmove(dst_address, i_matrix_address, 128)  # 128 = (4*4) * ctypes.sizeof(ctypes.c_double)
        print(f"update_influences_matrix: {ctx.influences_matrix.tolist()}")

    def _update_bind_pre_matrix(self, dataBlock: OpenMaya.MDataBlock):
        """
        Update:
            - `self.ctx.bind_pre_matrix`
        """
        ctx = self.ctx

        bind_pre_matrix_handle: OpenMaya.MDataHandle = dataBlock.inputValue(self.aBindPreMatrix)
        array_mObject = bind_pre_matrix_handle.data()
        if array_mObject.isNull():
            return
        fn_matrix = OpenMaya.MFnMatrixArrayData(array_mObject)
        array: OpenMaya.MMatrixArray = fn_matrix.array()
        length = array.length()
        if length == 0:
            return
        array_address = int(array[0].this)
        # maya 提供连续内存地址, 咱就不自己申请了, 直接使用 maya 内存地址
        _c = (ctypes.c_double * (length * 16)).from_address(array_address)
        ctx.bind_pre_matrix = memoryview(_c).cast("B").cast("d", (length, 16))
        print(f"update_bind_pre_matrix: {ctx.bind_pre_matrix.tolist()}")

    def _update_geo_matrix(self, dataBlock: OpenMaya.MDataBlock):
        """
        Update:
            - `self.ctx.geo_matrix`
            - `self.ctx.geo_matrix_i`
            - `self.ctx.geo_matrix_is_identity`
        """
        geo_matrix_handle: OpenMaya.MDataHandle = dataBlock.inputValue(self.aGeomMatrix)
        geo_matrix: OpenMaya.MMatrix = geo_matrix_handle.asMatrix()
        geo_matrix_i: OpenMaya.MMatrix = geo_matrix.inverse()

        _c = (ctypes.c_double * 16).from_address(int(geo_matrix.this))
        self.ctx.geo_matrix = memoryview(_c).cast("B").cast("d")

        _c = (ctypes.c_double * 16).from_address(int(geo_matrix_i.this))
        self.ctx.geo_matrix_i = memoryview(_c).cast("B").cast("d")

        self.ctx.geo_matrix_is_identity = geo_matrix.isEquivalent(OpenMaya.MMatrix.identity)
        print(f"update_geo_matrix: {self.ctx.geo_matrix.tolist()}")

    def _update_paint_information(self, dataBlock: OpenMaya.MDataBlock):
        """
        Update:
            - `self.ctx.current_paint_layer_index`
            - `self.ctx.current_paint_influence_index`
            - `self.ctx.current_paint_mask_bool`
        """
        ctx = self.ctx

        ctx.current_paint_layer_index = dataBlock.inputValue(self.aCurrentPaintLayerIndex).asInt()
        ctx.current_paint_influence_index = dataBlock.inputValue(self.aCurrentPaintInfluenceIndex).asInt()
        ctx.current_paint_mask_bool = dataBlock.inputValue(self.aCurrentPaintMaskBool).asBool()

    def _update_weights(self, dataBlock: OpenMaya.MDataBlock):
        """
        Update:
            - `self.ctx.skin_weights`
        """
        ctx = self.ctx

        weights_handle = dataBlock.inputValue(self.aWeights)
        ctx.skin_weights = MWeightsHandle.MWeightsHandle(weights_handle, ctx.input_mesh.num_vertices)

        if ctx.skin_weights.is_initialized:
            print(f"update_weights: {ctx.skin_weights.tolist()}")
        else:
            print("update_weights: None")

    def _update_deform_matrices(self):
        """
        Update:
            - `self.ctx.rotate_matrix`
            - `self.ctx.translate_vector`
        """
        ctx = self.ctx

        if ctx.influences_matrix is None or ctx.bind_pre_matrix is None:
            OpenMaya.MGlobal.displayWarning("influences_matrix or bind_pre_matrix is None!")
            return
        num_influences = ctx.bind_pre_matrix.shape[0]
        num_bind_pre_matrix = ctx.bind_pre_matrix.shape[0]

        if num_influences < 1 or num_bind_pre_matrix < 1:
            OpenMaya.MGlobal.displayWarning("num_influences <1 or num_bind_pre_matrix <1!")
            return
        if num_influences != num_bind_pre_matrix:
            OpenMaya.MGlobal.displayWarning("num_influences != num_bind_pre_matrix!")
            return
        # 申请内存
        # TODO 后续优化, 不要每一帧都申请新内存
        # 旋转矩阵是3*3,位移向量1*3
        _c = (ctypes.c_float * (9 * num_influences))()
        ctx.rotate_matrix = memoryview(_c).cast("B").cast("f", (num_influences, 9))
        _c = (ctypes.c_float * (3 * num_influences))()
        ctx.translate_vector = memoryview(_c).cast("B").cast("f", (num_influences, 3))

        cSkinDeformCython.cal_deform_matrices(
            ctx.rotate_matrix,
            ctx.translate_vector,
            ctx.influences_matrix,
            ctx.bind_pre_matrix,
            ctx.geo_matrix,
            ctx.geo_matrix_i,
            ctx.geo_matrix_is_identity,
        )
        print(f"update_deform_matrices: {ctx.translate_vector.tolist()}")

    def compute(self, plug, dataBlock):
        """
        很蛋疼的是`DG`模式下 如果`.outputGeometry[i]`输出给多个模型
        每个模型求值都会触发一次 `Deform` 函数 非常消耗性能 尤其是在绘制权重的时候
        一个输出给 maya geometry 一个输出给 权重颜色显示模型 会导致deform函数执行两次。
        所以前面配置了`setDependentsDirty`标记 只有在input的数据改变的时候 才会触发 `Deform` 函数。
        后续获取蒙皮后的模型数据 可以用任意方法求值 不会造成多余的 `Deform` 函数调用 以节约资源。
        """
        # if self.is_dirty is False:
        #     return None
        print("compute")
        res = super().compute(plug, dataBlock)
        return res

    def deform(self, dataBlock: OpenMaya.MDataBlock, geoIter, localToWorldMatrix, multiIndex):
        print("deform")
        self.event_update_mesh.execute(dataBlock, multiIndex)
        self.event_update_influences_matrix.execute(dataBlock)
        self.event_update_bind_pre_matrix.execute(dataBlock)
        self.event_update_geo_matrix.execute(dataBlock)
        self.event_update_paint_information.execute(dataBlock)
        self.event_update_weights.execute(dataBlock)
        self.event_update_deform_matrix.execute()

        return

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
        mAttr.setDisconnectBehavior(OpenMaya.MFnAttribute.kDelete)
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


class FnCSkinDeform:
    __slots__ = ("instance",)
    instance: CSkinDeform

    def __init__(self, cSkin_instance: CSkinDeform):
        self.instance = cSkin_instance

    @classmethod
    def from_string(cls, input_string: str):
        instance = _cRegistry.SkinRegistry.get_instance_by_string(input_string)
        return cls(instance)

    @classmethod
    def from_object_api1(cls, mObject: OpenMaya.MObject):
        instance = _cRegistry.SkinRegistry.get_instance_by_api1(mObject)
        return cls(instance)

    @classmethod
    def from_object_api2(cls, mObject: OpenMaya.MObject):
        instance = _cRegistry.SkinRegistry.get_instance_by_api2(mObject)
        return cls(instance)

    def fast_preview_deform(self, vertex_indices: memoryview | None = None):
        """
        局部蒙皮算法, 专供笔刷调用
        根据笔刷的 hit_indices 来局部计算蒙皮,
        不唤醒 deform 函数, 不触发maya dg, 直接通知渲染节点更新
        """

    def set_dirty(self):
        """
        标记节点位脏, 在需要的时候Maya自行求值.

        通常情况需要搭配 `set_dirty` 先标记为脏, 再使用 `pull_output` 强行求值.

        此方法只能外部调用, 节点内部严禁调用.
        """
        plug = OpenMaya.MPlug(self.instance.mObject, self.instance.aForceDirty)
        plug.setInt(plug.asInt() + 1)
        return True

    def pull_output(self):
        """
        强行向 Maya 索要输出数据 以触发 deform 求值 适用于笔刷修改权重后需要立刻更新模型的场景.

        通常情况需要搭配 `set_dirty` 先标记为脏, 再使用 `pull_output` 强行求值.

        此方法只能外部调用, 节点内部严禁调用.
        """
        mPlug: OpenMaya.MPlug = OpenMaya.MPlug(self.instance.mObject, self.instance.aOutputGeometry).elementByLogicalIndex(0)
        return mPlug.asMObject()

    def set_bind_pre_matrix(self, bind_pre_matrix_array: OpenMaya.MMatrixArray | memoryview | list):
        """
        TODO 想办法优化，最好可以直接支持UNDO REDO
        """

        if isinstance(bind_pre_matrix_array, memoryview):
            array = OpenMaya.MMatrixArray()
            length = bind_pre_matrix_array.shape[0]
            array.setLength(length)
            address = int(array[0].this)
            src_address = ctypes.addressof(bind_pre_matrix_array.obj)
            ctypes.memmove(address, src_address, length * 128)  # 128 = (4*4) * ctypes.sizeof(ctypes.c_double)

        else:
            array = bind_pre_matrix_array

        mObject = OpenMaya.MFnMatrixArrayData().create(array)
        plug = OpenMaya.MPlug(self.instance.mObject, self.instance.aBindPreMatrix)
        plug.setMObject(mObject)
        return True

    def transfer_bind_pre_matrix_from_skinCluster(self, skinCluster_name: str):
        """ """
