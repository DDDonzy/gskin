import sys
import maya.OpenMaya as om1
import maya.OpenMayaMPx as mpx
from gskin.src.MFloatArrayProxy import MFloatArrayProxy


# 节点名称和 ID
NODE_NAME = "MemTestNode"
NODE_ID = om1.MTypeId(0x87001)  # 自定义测试 ID


class MemTestNode(mpx.MPxNode):
    # 属性对象
    aCustomData = om1.MObject()
    aCustomDataP = om1.MObject()
    aRefresh = om1.MObject()
    aDummy = om1.MObject()

    def __init__(self):
        mpx.MPxNode.__init__(self)

    def compute(self, plug, dataBlock):

        # 1. 获取输入触发值 (refresh)
        refresh_val = dataBlock.inputValue(MemTestNode.aRefresh).asBool()

        try:
            dataHandle = dataBlock.inputValue(MemTestNode.aCustomDataP)
            data = dataHandle.data()
            print(hex(int(om1.MFnPointArrayData(data).array()[0].this)))
        except:
            pass

        try:
            dataHandle = dataBlock.inputValue(MemTestNode.aCustomData)
            data = dataHandle.data()
            print(hex(int(om1.MFnVectorArrayData(data).array()[0].this)))
        except:
            pass

        h_dummy = dataBlock.outputValue(MemTestNode.aDummy)
        h_dummy.setFloat(1.0 if refresh_val else 0.0)

        dataBlock.setClean(plug)


def nodeCreator():
    return mpx.asMPxPtr(MemTestNode())


def nodeInitializer():
    nAttr = om1.MFnNumericAttribute()
    tAttr = om1.MFnTypedAttribute()

    # 1. 创建 MVectorArray 属性
    # 使用 om1.MFnData.kVectorArray 正确指定类型
    MemTestNode.aCustomData = tAttr.create(
        "customData",
        "cd",
        om1.MFnData.kVectorArray,
        om1.MFnVectorArrayData().create(om1.MVectorArray()),
    )
    tAttr.setStorable(True)
    tAttr.setKeyable(False)

    MemTestNode.aCustomDataP = tAttr.create("customDataP", "cdp", om1.MFnData.kPointArray)
    tAttr.setStorable(True)
    tAttr.setKeyable(False)

    # 2. Refresh 属性

    MemTestNode.aRefresh = nAttr.create("refresh", "ref", om1.MFnNumericData.kBoolean, False)
    nAttr.setKeyable(True)
    nAttr.setStorable(True)

    # 3. Dummy 输出属性
    MemTestNode.aDummy = nAttr.create("dummy", "dum", om1.MFnNumericData.kFloat, 0.0)
    nAttr.setWritable(False)
    nAttr.setStorable(False)

    # 添加并设置依赖
    MemTestNode.addAttribute(MemTestNode.aCustomData)
    MemTestNode.addAttribute(MemTestNode.aCustomDataP)
    MemTestNode.addAttribute(MemTestNode.aRefresh)
    MemTestNode.addAttribute(MemTestNode.aDummy)

    MemTestNode.attributeAffects(MemTestNode.aRefresh, MemTestNode.aDummy)
    MemTestNode.attributeAffects(MemTestNode.aCustomData, MemTestNode.aDummy)
    MemTestNode.attributeAffects(MemTestNode.aCustomDataP, MemTestNode.aDummy)


# 注册插件的标配函数
def initializePlugin(mobject):
    mplugin = mpx.MFnPlugin(mobject, "YourName", "1.0", "Any")
    try:
        mplugin.registerNode(NODE_NAME, NODE_ID, nodeCreator, nodeInitializer)
    except:
        sys.stderr.write("Failed to register node: %s\n" % NODE_NAME)


def uninitializePlugin(mobject):
    mplugin = mpx.MFnPlugin(mobject)
    try:
        mplugin.deregisterNode(NODE_ID)
    except:
        sys.stderr.write("Failed to deregister node: %s\n" % NODE_NAME)
