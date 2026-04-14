from __future__ import annotations

import ctypes
import weakref


from maya import cmds, mel
import maya.api.OpenMaya as OpenMaya2  # type:ignore
import maya.api.OpenMayaAnim as OpenMayaAnim2  # type:ignore
import maya.OpenMaya as OpenMaya  # type:ignore
import maya.OpenMayaMPx as OpenMayaMPx  # type:ignore

from . import cSkinDeformCython
from .MProfiler import MProfiler
from .MRegistry import MRegistry
from .MDirtyEvent import DirtyEvent
from .MWeightsHandle import MWeightsHandle
from .MTopologyContext import TopologyContext


__all__ = ("FnCSkinDeform",)


class CSkinContext:
    __slots__ = (
        "envelope",
        "current_paint_layer_index",
        "current_paint_influence_index",
        "current_paint_mask_bool",
        "skin_weights",
        "geo_matrix",
        "geo_matrix_i",
        "geo_matrix_is_identity",
        "bind_pre_matrix",
        "influences_matrix",
        "rotate_matrix",
        "translate_vector",
        "input_mesh",
        "output_mesh",
    )

    # fmt:off
    envelope: float

    current_paint_layer_index    : int
    current_paint_influence_index: int
    current_paint_mask_bool      : bool

    skin_weights : MWeightsHandle

    geo_matrix            : memoryview
    geo_matrix_i          : memoryview
    geo_matrix_is_identity: bool

    bind_pre_matrix  : memoryview
    influences_matrix: memoryview
    rotate_matrix    : memoryview
    translate_vector : memoryview

    input_mesh : TopologyContext
    output_mesh : TopologyContext

    # fmt:on

    def __init__(self):
        # fmt:off
        self.envelope = 1.0

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
    NODE_TYPE = r"cSkinDeformer"
    NODE_ID = OpenMaya.MTypeId(0x00080033)

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

    aInput          :OpenMaya.MObject  = OpenMayaMPx.cvar.MPxGeometryFilter_input
    aInputGeometry  : OpenMaya.MObject = OpenMayaMPx.cvar.MPxGeometryFilter_inputGeom
    aOutputGeometry : OpenMaya.MObject = OpenMayaMPx.cvar.MPxGeometryFilter_outputGeom
    aEnvelope       : OpenMaya.MObject = OpenMayaMPx.cvar.MPxGeometryFilter_envelope
    # fmt:on

    def __init__(self):
        super().__init__()

        # fmt:off
        # --- maya
        self.hashCode       : int                        = None
        self.mObject        : OpenMaya.MObject           = None
        self.mFnDependNode  : OpenMaya.MFnDependencyNode = None
        self.plug_refresh   : OpenMaya.MPlug             = None

        self.ctx = CSkinContext()
        self.layer_manager = WeightsLayerManager()

        self.init_dirtyEvent()

        # fmt:on

    def init_dirtyEvent(self):
        # dirty flag
        # fmt:off
        self.event_envelope                 = DirtyEvent(self.aEnvelope, 
                                                                            self._update_envelope)
        self.event_update_mesh              = DirtyEvent(self.aInputGeometry, 
                                                                            self._update_mesh)
        self.event_update_influences_matrix = DirtyEvent(self.aInfluenceMatrix, 
                                                                            self._update_influences_matrix)
        self.event_update_bind_pre_matrix   = DirtyEvent(self.aBindPreMatrix, 
                                                                            self._update_bind_pre_matrix)
        self.event_update_geo_matrix        = DirtyEvent(self.aGeomMatrix, 
                                                                            self._update_geo_matrix)
        self.event_update_deform_matrix     = DirtyEvent((self.aInfluenceMatrix,
                                                                             self.aBindPreMatrix,
                                                                             self.aGeomMatrix),
                                                                            self._update_deform_matrices,)  # fmt:skip
        self.event_update_paint_information = DirtyEvent((self.aCurrentPaintLayerIndex,
                                                                             self.aCurrentPaintInfluenceIndex,
                                                                             self.aCurrentPaintMaskBool),
                                                                            self._update_paint_information)  # fmt:skip
        self.event_update_weights           = DirtyEvent((self.aWeights,
                                                                             self.aMaskWeights,
                                                                             self.aLockMasks), 
                                                                             self._update_weights)
        self.event_update_layer_manager     = DirtyEvent((self.aMaskWeights,
                                                                             self.aLockMasks,
                                                                             self.aLayerCompound,
                                                                             self.aLayerName,
                                                                             self.aLayerEnabled,
                                                                             self.aLayerWeights,
                                                                             self.aLayerLockInfluences),
                                                                            self._update_layer_manager)  # fmt:skip

        # fmt:on

    def setDependentsDirty(self, plug: OpenMaya.MPlug, dirtyPlugArray: OpenMaya.MPlugArray):
        """DG模式下脏数据标签 性能优化"""
        self.event_envelope.sync_from_plug(plug)
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
        self.event_envelope.sync_from_evaluation(evaluationNode)
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
        Update:
            - `self.mObject`
            - `self.mFnDependNode`
            - `self.hashCode`
        """
        # fmt:off
        self.mObject       = self.thisMObject()
        self.mFnDependNode = OpenMaya.MFnDependencyNode(self.mObject)
        # fmt:on
        MRegistry.register(self)

    def _update_envelope(self, dataBlock: OpenMaya.MDataBlock):
        """
        Update:
            - `self.ctx.envelope`
        """
        self.ctx.envelope = dataBlock.inputValue(self.aEnvelope).asFloat()

    def _update_mesh(self, dataBlock: OpenMaya.MDataBlock, multiIndex: int):
        """
        Updata:
            - `self.ctx.input_mesh`
            - `self.ctx.output_mesh`
        """
        ctx = self.ctx
        # input/orig meshes
        # HACK: 直接使用 outputArrayValue 读取输入网格以跳过冗余的 DG 检查, 如果遇到奇怪的BUG, 改回 inputArrayValue
        input_array_handle: OpenMaya.MArrayDataHandle = dataBlock.outputArrayValue(self.aInput)
        input_array_handle.jumpToElement(multiIndex)
        input_geom_obj = input_array_handle.outputValue().child(self.aInputGeometry).asMesh()
        ctx.input_mesh.update_fnMesh(OpenMaya.MFnMesh(input_geom_obj))
        ctx.input_mesh.update_position()
        # output meshes
        output_array_handle: OpenMaya.MArrayDataHandle = dataBlock.outputArrayValue(self.aOutputGeometry)
        output_array_handle.jumpToElement(multiIndex)
        output_geom_obj = output_array_handle.outputValue().asMesh()
        ctx.output_mesh.update_fnMesh(OpenMaya.MFnMesh(output_geom_obj))
        ctx.output_mesh.update_position()
        # topology
        # TODO 需要优化避免每次都更新topology
        ctx.output_mesh.update_topology()
        # print("update_input_mesh")

    def _update_influences_matrix(self, dataBlock: OpenMaya.MDataBlock):
        """
        Update:
            - `self.ctx.influences_matrix`
        """
        ctx = self.ctx

        influence_handle: OpenMaya.MArrayDataHandle = dataBlock.inputArrayValue(self.aInfluenceMatrix)
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
        # print(f"update_influences_matrix: {ctx.influences_matrix.tolist()}")

    def _update_bind_pre_matrix(self, dataBlock: OpenMaya.MDataBlock):
        """
        Update:
            - `self.ctx.bind_pre_matrix`
        """
        ctx = self.ctx

        bindPreMatrix_handle: OpenMaya.MArrayDataHandle = dataBlock.inputArrayValue(self.aBindPreMatrix)
        num_influences = bindPreMatrix_handle.elementCount()
        if num_influences == 0:
            return
        # 申请内存池
        # TODO 需要优化, 避免每帧都申请新内存池, 尽量复用
        _c = (ctypes.c_double * (num_influences * 16))()
        ctx.bind_pre_matrix = memoryview(_c).cast("B").cast("d", (num_influences, 16))
        # 把maya数据填充到内存池
        bindPreMatrix_address = ctypes.addressof(ctx.bind_pre_matrix.obj)
        for i in range(num_influences):
            bindPreMatrix_handle.jumpToArrayElement(i)
            i_matrix_address = int(bindPreMatrix_handle.inputValue().asMatrix().this)
            dst_address = (i * 128) + bindPreMatrix_address  # 128 = (4*4) * ctypes.sizeof(ctypes.c_double)
            ctypes.memmove(dst_address, i_matrix_address, 128)  # 128 = (4*4) * ctypes.sizeof(ctypes.c_double)
        # print(f"update_influences_matrix: {ctx.influences_matrix.tolist()}")

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
        # print(f"update_geo_matrix: {self.ctx.geo_matrix.tolist()}")

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
        ctx.skin_weights = MWeightsHandle(weights_handle)

    def _update_deform_matrices(self):
        """
        Update:
            - `self.ctx.rotate_matrix`
            - `self.ctx.translate_vector`
        """
        ctx = self.ctx

        # 校验 influences_matrix 和 bind_pre_matrix 真实存在
        if ctx.influences_matrix is None or ctx.bind_pre_matrix is None:
            OpenMaya.MGlobal.displayWarning("influences_matrix or bind_pre_matrix is None!")
            return
        num_influences = ctx.bind_pre_matrix.shape[0]
        num_bind_pre_matrix = ctx.bind_pre_matrix.shape[0]
        # 校验 num_influences 和 num_bind_pre_matrix 不为空
        if num_influences < 1 or num_bind_pre_matrix < 1:
            OpenMaya.MGlobal.displayWarning("num_influences <1 or num_bind_pre_matrix <1!")
            return
        # 校验 num_influences 和 num_bind_pre_matrix 数量一致
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
        # print(f"update_deform_matrices: {ctx.translate_vector.tolist()}")

    def _run_skinning(self):
        ctx = self.ctx
        num_influences = ctx.skin_weights.num_influences
        num_vertices = ctx.skin_weights.num_vertices
        # 校验数据
        if (ctx.output_mesh.position is None or
            ctx.input_mesh.position  is None or
            ctx.skin_weights         is None or
            ctx.rotate_matrix        is None or
            ctx.translate_vector     is None):  # fmt:skip
            OpenMaya.MGlobal.displayWarning("data error!")
            return
        if (ctx.output_mesh.position.shape[0] != num_vertices * 3 or 
            ctx.skin_weights.view.shape[0]    != num_influences * num_vertices or 
            ctx.rotate_matrix.shape[0]        != num_influences or 
            ctx.translate_vector.shape[0]     != num_influences):  # fmt:skip
            OpenMaya.MGlobal.displayWarning("data error!")
            return

        cSkinDeformCython.run_skinning_core(
            ctx.output_mesh.position,
            ctx.input_mesh.position,
            ctx.skin_weights.view,
            ctx.rotate_matrix,
            ctx.translate_vector,
            ctx.envelope,
        )

    def _update_layer_manager(self, dataBlock: OpenMaya.MDataBlock):
        """
        Update:
            - `self.layer_manager`
        """
        self.layer_manager.update_from_dataBlock(dataBlock)
        # print(self.layer_manager)

    def compute(self, plug, dataBlock: OpenMaya.MDataBlock):
        """
        很蛋疼的是`DG`模式下 如果`.outputGeometry[i]`输出给多个模型
        每个模型求值都会触发一次 `Deform` 函数 非常消耗性能 尤其是在绘制权重的时候
        一个输出给 maya geometry 一个输出给 权重颜色显示模型 会导致deform函数执行两次。
        所以前面配置了`setDependentsDirty`标记 只有在input的数据改变的时候 才会触发 `Deform` 函数。
        后续获取蒙皮后的模型数据 可以用任意方法求值 不会造成多余的 `Deform` 函数调用 以节约资源。
        """
        # 涉及并行模式内存地址问题, 强制每帧更新
        self.event_update_mesh.set_dirty(True)

        return super().compute(plug, dataBlock)

    @MProfiler(color=9)
    def deform(self, dataBlock: OpenMaya.MDataBlock, geoIter, localToWorldMatrix, multiIndex):
        # print("deform")
        with MProfiler("event_envelope", color=10):
            self.event_envelope.execute(dataBlock)
        with MProfiler("event_update_mesh", color=11):
            self.event_update_mesh.execute(dataBlock, multiIndex)
        with MProfiler("event_update_influences_matrix", color=12):
            self.event_update_influences_matrix.execute(dataBlock)
        with MProfiler("event_update_bind_pre_matrix", color=13):
            self.event_update_bind_pre_matrix.execute(dataBlock)
        with MProfiler("event_update_geo_matrix", color=14):
            self.event_update_geo_matrix.execute(dataBlock)
        with MProfiler("event_update_paint_information", color=15):
            self.event_update_paint_information.execute(dataBlock)
        with MProfiler("event_update_weights", color=16):
            self.event_update_weights.execute(dataBlock)
        with MProfiler("event_update_layer_manager", color=17):
            self.event_update_layer_manager.execute(dataBlock)
        with MProfiler("event_update_deform_matrix", color=18):
            self.event_update_deform_matrix.execute()
        with MProfiler("run_skinning", color=19):
            self._run_skinning()
        return

    @classmethod
    def nodeInitializer(cls):
        # fmt:off
        nAttr = OpenMaya.MFnNumericAttribute()
        tAttr = OpenMaya.MFnTypedAttribute()
        mAttr = OpenMaya.MFnMatrixAttribute()
        cAttr = OpenMaya.MFnCompoundAttribute()

        cls.aGeomMatrix      = mAttr.create("geomMatrix", "gm")
        cls.aBindPreMatrix   = mAttr.create("bindPreMatrix", "bpm")
        mAttr.setArray(True)
        mAttr.setUsesArrayDataBuilder(True)
        mAttr.setDisconnectBehavior(OpenMaya.MFnAttribute.kDelete)
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

        for attr in (cls.aLayerCompound,
                     cls.aLayerName,
                     cls.aLayerEnabled,
                     cls.aLayerWeights,
                     cls.aLayerLockInfluences,
                     cls.aMaskWeights,
                     cls.aLockMasks):
            cls.attributeAffects(attr, cls.aOutputGeometry)
        # fmt:on

    @classmethod
    def nodeCreator(cls):
        return OpenMayaMPx.asMPxPtr(CSkinDeform())


class WeightLayerItem:
    __slots__ = (
        "name",
        "enabled",
        "weights",
        "lock_influences",
        "item_mDataHandle",
    )
    item_mDataHandle: OpenMaya.MDataHandle

    name: str
    enabled: bool
    weights: MWeightsHandle
    lock_influences: list | None | OpenMaya.MIntArray

    def __init__(
        self,
        dataHandle: OpenMaya.MDataHandle,
    ):
        self.item_mDataHandle = dataHandle

        self.name = dataHandle.child(CSkinDeform.aLayerName).asString()
        self.enabled = dataHandle.child(CSkinDeform.aLayerEnabled).asBool()
        self.weights = MWeightsHandle(dataHandle.child(CSkinDeform.aLayerWeights))
        self.lock_influences = WeightsLayerManager.get_MIntArray(dataHandle.child(CSkinDeform.aLayerLockInfluences))

    def __repr__(self) -> str:
        # 1. 安全处理 lock_influences (可能是 None, list, 或 MIntArray)
        if self.lock_influences is None:
            locks_info = "None"
        elif hasattr(self.lock_influences, "length"):  # MIntArray 鸭子类型检测
            locks_info = f"<MIntArray length={self.lock_influences.length()}>"
        elif isinstance(self.lock_influences, list):
            locks_info = f"[{len(self.lock_influences)} items]"
        else:
            locks_info = type(self.lock_influences).__name__

        # 2. 安全处理 weights (极大概率是 MWeightsHandle 或 list)
        if isinstance(self.weights, list):
            weights_info = f"[{len(self.weights)} items]"
        elif self.weights is None:
            weights_info = "None"
        else:
            # 如果 MWeightsHandle 自己写了很好的 __repr__，这里会直接调用
            # 如果没写，至少打印出它的类名，而不会引发海量数据输出
            weights_info = f"<{self.weights.__class__.__name__}>"

        return f"<{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled}, weights={weights_info}, lock_influences={locks_info})>"


class WeightsLayerManager:
    """
    CSkinDeform 权重Layer管理器.

    提供节点内部和节点外部两种更新模式
    """

    __slots__ = (
        "mDependNode",
        "node_name",
        "mask_weights",
        "mask_weights_lock",
        "_items_dict",
    )

    node_name: str

    mask_weights: MWeightsHandle
    mask_weights_lock: list | OpenMaya.MIntArray | None
    _items_dict: dict[int, WeightLayerItem]

    def __init__(self, node_name=None):
        self.node_name = node_name
        self.mDependNode = None
        self.mask_weights = None
        self.mask_weights_lock = None
        self._items_dict = {}

        if self.node_name:
            sel = OpenMaya.MSelectionList()
            try:
                sel.add(self.node_name)
            except RuntimeError as e:
                raise RuntimeError(f"'{self.node_name}' is not a valid node") from e

            self.node_name = self.node_name
            obj = OpenMaya.MObject()
            sel.getDependNode(0, obj)
            self.update_from_dependNode(obj)
            self.mDependNode = obj

    @staticmethod
    def get_MIntArray(dataHandle: OpenMaya.MDataHandle):
        """
        获取 MIntArray
        Return:
            array (OpenMaya.MIntArray | None): 返回 `MIntArray`
        """
        obj: OpenMaya.MObject = dataHandle.data()
        if obj.isNull():
            return None
        int_array = OpenMaya.MFnIntArrayData(obj).array()
        if int_array.length() == 0:
            return None
        return int_array

    def update(self):
        """
        更新 `Layer` 信息组
        仅供节点外部调用, 如果是节点内部, 请使用 `update_from_dataBlock`
        """
        self.update_from_dependNode(self.mDependNode)

    def update_from_dataBlock(self, dataBlock: OpenMaya.MDataBlock):
        """
        从节点内部 DataBlock 更新 Layer 信息组.
        Args:
            dataBlock (OpenMaya.MDataBlock): 输入节点内部 `DataBlock`
        Update:
            - `self.mask_weights`
            - `self.mask_weights_lock`
            - `self._items_dict`
        """
        self.mask_weights = MWeightsHandle(dataBlock.outputValue(CSkinDeform.aMaskWeights))
        self.mask_weights_lock = self.get_MIntArray(dataBlock.outputValue(CSkinDeform.aLockMasks))
        arrayDataHandle = dataBlock.outputArrayValue(CSkinDeform.aLayerCompound)
        self._update_items_from_arrayDataHandle(arrayDataHandle)

    def update_from_dependNode(self, depend_node: OpenMaya.MObject):
        """
        从根据`CSkinDeform`的`Attributes Object` 查找 `MPlug` 更新 Layer 信息组.
        等同 `update_from_string`
        Args:
            depend_node (OpenMaya.MObject): 输入`CSkinDeform`节点实例的 `MObject`
        Update:
            - `self.mask_weights`
            - `self.mask_weights_lock`
            - `self._items_dict`
        """
        compound_plug = OpenMaya.MPlug(depend_node, CSkinDeform.aLayerCompound)
        mask_plug = OpenMaya.MPlug(depend_node, CSkinDeform.aMaskWeights)
        lock_plug = OpenMaya.MPlug(depend_node, CSkinDeform.aLockMasks)
        self.mask_weights = MWeightsHandle(mask_plug.asMDataHandle())
        self.mask_weights_lock = self.get_MIntArray(lock_plug.asMDataHandle())

        arrayDataHandle = OpenMaya.MArrayDataHandle(compound_plug.asMDataHandle())
        self._update_items_from_arrayDataHandle(arrayDataHandle)

    def _update_items_from_arrayDataHandle(self, arrayDataHandle: OpenMaya.MArrayDataHandle):
        """
        私有方法, 根据 MDataBlock/MPlug 获取的 `MArrayDataHandle` 更新 Layer 信息组.
        DataBlock/MPlug 迭代查找ArrayHandle方法相同, 这里就提为单独函数.
        Args:
            arrayDataHandle (OpenMaya.MArrayDataHandle): 输入 `MArrayDataHandle`
        Update:
            - `self._items_dict`
        """
        self._items_dict.clear()
        count = arrayDataHandle.elementCount()
        for i in range(count):
            arrayDataHandle.jumpToArrayElement(i)
            logical_index = arrayDataHandle.elementIndex()
            dataHandle: OpenMaya.MDataHandle = arrayDataHandle.outputValue()
            item = WeightLayerItem(dataHandle)
            self._items_dict[logical_index] = item
        return True

    def get_layer(self, layer_index: int) -> WeightLayerItem | None:
        return self._items_dict.get(layer_index, None)

    def __repr__(self) -> str:
        res = (f"<{self.__class__.__name__}>\n"
               f"    Mask Weights: {self.mask_weights}\n\n"
               f"    Mask Weights Lock: {self.mask_weights_lock}\n\n")  # fmt:skip
        for i, v in self._items_dict.items():
            res += f"    {i} : " + repr(v) + "\n"
        return res


class FnCSkinDeform:
    __slots__ = (
        "_instance_ref",
        "node_name",
    )
    instance: CSkinDeform
    node_name: str

    def __init__(self, cSkin_instance: CSkinDeform):
        # 强制将传入的实例转为 proxy
        # 如果传进来的已经是 proxy, weakref.proxy 会安全放行
        if isinstance(cSkin_instance, weakref.ProxyTypes):
            self._instance_ref = cSkin_instance
        else:
            self._instance_ref = weakref.proxy(cSkin_instance)

        self.node_name = self._instance_ref.mFnDependNode.name()

    @property
    def instance(self) -> CSkinDeform:
        return self._instance_ref

    @classmethod
    def from_string(cls, input_string: str):
        instance = MRegistry.get_instance(input_string)
        return cls(instance)

    @classmethod
    def from_mObject(cls, mObject: OpenMaya.MObject | OpenMaya2.MObject):

        instance = MRegistry.get_instance(mObject)
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

    def ensure_node(self):
        """校验节点是否已绑定且在场景中真实存在"""
        if not self.node_name:
            raise ValueError("node_name is None, need update from string first.")
        if not cmds.objExists(self.node_name):
            raise ValueError(f"'{self.node_name}' is not a valid node.")
        return True

    def add_layer(self):
        """
        添加 Layer
        节点内部严禁调用
        """
        self.ensure_node()
        indices = cmds.getAttr(f"{self.node_name}.layers.layerData", mi=1)
        i = indices[-1] + 1 if indices else 0
        cmds.setAttr(f"{self.node_name}.layers.layerData[{i}].layerName", "new_layer", type="string")
        cmds.setAttr(f"{self.node_name}.layers.layerData[{i}].layerEnabled", 1)
        return True

    def delete_layer(self, index: int):
        """
        删除 Layer
        节点内部严禁调用
        """
        self.ensure_node()
        cmds.removeMultiInstance(f"{self.node_name}.layers.layerData[{index}]", b=True)
        return True

    def set_weights(self, weights, num_influences):
        """
        设置权重数据
        节点内部严禁调用
        TODO 暂时不支持 undo redo
        Args:
            weights (list|array.array|memoryview): 权重数据列表
        """
        weights_plug = OpenMaya.MPlug(self.instance.mObject, self.instance.aWeights)
        weights_handle: MWeightsHandle = MWeightsHandle.from_mPlug(weights_plug)
        weights_handle.set_weights(weights, num_influences)
        weights_handle.set_to_mPlug(weights_plug)
        return True

    def set_weights_from_skinCluster(self, name: str):
        """
        从 skinCluster 中获取并设置权重数据
        Args:
            name (str): 输入模型名称/蒙皮节点名称
        """
        weights, num_influences = self._get_skinCluster_weights(name)
        self.set_weights(weights, num_influences)
        return True

    def set_mask_weights(self, weights: list, num_layers):
        """
        设置Mask数据
        节点内部严禁调用
        Args:
            weights (list|array.array|memoryview): 权重数据列表
        """
        weights_plug = OpenMaya.MPlug(self.instance.mObject, self.instance.aMaskWeights)
        weights_handle: MWeightsHandle = MWeightsHandle.from_mPlug(weights_plug)
        weights_handle.set_weights(weights, num_layers)
        weights_handle.set_to_mPlug(weights_plug)
        return True

    def set_mask_weights_lock(self, lock_list: list):
        self.ensure_node()
        cmds.setAttr(f"{self.node_name}.layers.layersLockMask", lock_list, type="Int32Array")
        return True

    def set_layer_name(self, layer_index: int, _name: str):
        """
        设置 Layer 名称
        节点内部严禁调用
        """
        self.ensure_node()
        cmds.setAttr(f"{self.node_name}.layers.layerData[{layer_index}].layerName", _name, type="string")
        return True

    def set_layer_enabled(self, layer_index: int, enabled: bool):
        """
        设置 Layer 启用状态
        节点内部严禁调用
        """
        self.ensure_node()
        cmds.setAttr(f"{self.node_name}.layers.layerData[{layer_index}].layerEnabled", enabled)
        return True

    def set_layer_weights(self, layer_index, weights, num_influences):
        """
        设置 Layer 权重数据
        节点内部严禁调用
        """
        attr = f"{self.node_name}.layers.layerData[{layer_index}].layerWeightsData"
        weights_handle: MWeightsHandle = MWeightsHandle.from_string(attr)
        weights_handle.set_weights(weights, num_influences)
        weights_handle.set_to_string(attr)
        return True

    def set_layer_lock_influences(self, layer_index: int, lock_list: list):
        """
        设置 Layer 权重锁定状态
        节点内部严禁调用
        """
        self.ensure_node()
        cmds.setAttr(f"{self.node_name}.layers.layerData[{layer_index}].layerLockInfluences", lock_list, type="Int32Array")
        return True

    def get_layer_name(self, layer_index: int):
        """
        获取 Layer 名称
        节点内部严禁调用
        """
        self.ensure_node()
        return cmds.getAttr(f"{self.node_name}.layers.layerData[{layer_index}].layerName")

    def get_layer_enabled(self, layer_index: int):
        """
        获取 Layer 启用状态
        节点内部严禁调用
        """
        self.ensure_node()
        return cmds.getAttr(f"{self.node_name}.layers.layerData[{layer_index}].layerEnabled")

    def get_layer_lock_influences(self, layer_index: int):
        """
        获取 Layer 权重锁定状态
        节点内部严禁调用
        """
        self.ensure_node()
        return cmds.getAttr(f"{self.node_name}.layers.layerData[{layer_index}].layerLockInfluences")

    @staticmethod
    def _get_skinCluster_weights(name):
        """
        获取 maya 自身 `skinCluster` 节点的蒙皮权重

        Args:
            name (str): 输入模型名称/蒙皮节点名称
        Return:
            weights (OpenMaya2.MDoubleArray): 权重
            num_influences (int): influences数量
        """
        # 如果输入的是模型名字, 可以通过 ‘findRelatedSkinCluster’ 找到skinCluster
        name = mel.eval(f'findRelatedSkinCluster("{name}")') or name
        # get mesh
        mesh = cmds.skinCluster(name, q=1, g=1)[0]
        sel: OpenMaya2.MSelectionList = OpenMaya2.MGlobal.getSelectionListByName(name)
        sel.add(mesh)
        mObj_skinCluster = sel.getDependNode(0)
        fn_skinCluster: OpenMayaAnim2.MFnSkinCluster = OpenMayaAnim2.MFnSkinCluster(mObj_skinCluster)
        mesh_dag = sel.getDagPath(1)

        weights, num_influences = fn_skinCluster.getWeights(mesh_dag, OpenMaya2.MObject())

        return weights, num_influences

    @staticmethod
    def create_cSkinDeform_from_skinCluster(skinCluster: str):
        """
        根据 `skinCluster` 创建 `cSkinDeform` 实例.
        并且设置 `matrix`, `bindPreMatrix`, `weights`.
        TODO maya api 获取权重, 再转为list传递给MWeightsHandle很慢需要优化

        Args:
            skinCluster (str): 输入 skinCluster 名称 或者 mesh 名称
        Return:
            name (str): cSkinDeform 名称
            instance (CSkinDeform): cSkinDeform 实例
        """

        skinCluster = mel.eval(f'findRelatedSkinCluster("{skinCluster}")') or skinCluster
        mesh = cmds.skinCluster(skinCluster, q=1, g=1)[0]
        cSkin = cmds.deformer(mesh, type=CSkinDeform.NODE_TYPE)[0]

        cSkin_instance = FnCSkinDeform.from_string(cSkin)

        cmds.connectAttr(f"{skinCluster}.matrix", f"{cSkin}.matrix")
        cmds.connectAttr(f"{skinCluster}.bindPreMatrix", f"{cSkin}.bindPreMatrix")

        weights, num_influences = FnCSkinDeform._get_skinCluster_weights(skinCluster)

        cSkin_instance.set_weights(weights, num_influences)
        cmds.setAttr(f"{skinCluster}.envelope", 0.0)
        return cSkin, cSkin_instance
