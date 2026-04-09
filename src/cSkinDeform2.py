from __future__ import annotations
import ctypes

import maya.OpenMaya as OpenMaya  # type:ignore
import maya.OpenMayaMPx as OpenMayaMPx  # type:ignore

from . import _cRegistry
from . import cSkinDeformCython
from . import cDirtyManager


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

        self.init_dirtyEvent()
        # fmt:on

    def init_dirtyEvent(self):
        # dirty flag
        self.is_dirty = True
        self.dirtyEvent = cDirtyManager.DirtyEventManager()

        self.dirtyEvent.add_handler(
            cDirtyManager.DirtyEventHandler(self.aForceDirty, self._print_v),
        )

    def setDependentsDirty(self, plug: OpenMaya.MPlug, dirtyPlugArray: OpenMaya.MPlugArray):
        """DG模式下脏数据标签 性能优化"""

        self.dirtyEvent.sync_from_plug(plug)
        return super().setDependentsDirty(plug, dirtyPlugArray)

    def preEvaluation(self, context: OpenMaya.MDGContext, evaluationNode: OpenMaya.MEvaluationNode):
        """并行模式下脏数据标签 性能优化"""

        self.dirtyEvent.sync_from_evaluation(evaluationNode)
        return super().preEvaluation(context, evaluationNode)

    def _print_v(self,dataBlock):
        print(self.dataBlock.inputValue(self.aForceDirty).asInt())

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

    def pull_output(self):
        """
        强行向 Maya 索要输出数据 以触发 deform 求值 适用于笔刷修改权重后需要立刻更新模型的场景.

        通常情况需要搭配 `set_dirty` 先标记为脏, 再使用 `pull_output` 强行求值.

        此方法只能外部调用, 节点内部严禁调用.
        """
        mPlug: OpenMaya.MPlug = OpenMaya.MPlug(self.mObject, self.aOutputGeometry).elementByLogicalIndex(0)
        mPlug.asMObject()
        return True

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

        res = super().compute(plug, dataBlock)
        # self.is_dirty = False
        print("compute")
        self.dataBlock = dataBlock
        self.dirtyEvent.execute(dataBlock)
        return res

    def deform(self, dataBlock: OpenMaya.MDataBlock, geoIter, localToWorldMatrix, multiIndex):
        print("deform")

        return

    def fast_preview_deform(self, hit_indices: memoryview | None = None, hit_count: int = 0):
        """
        局部蒙皮算法, 专供笔刷调用
        根据笔刷的 hit_indices 和 hit_count 来局部计算蒙皮,
        不唤醒 deform 函数, 不触发maya dg, 直接通知渲染节点更新
        """

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
