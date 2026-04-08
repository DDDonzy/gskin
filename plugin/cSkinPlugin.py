import maya.api.OpenMaya as om
import maya.OpenMaya as om1
import maya.OpenMayaMPx as ompx

# 鉴于 cSkinDeform 节点本身是基于旧版 API (OM1) 开发的，
# 它的插件加载器必须使用旧版的 MFnPlugin (ompx.MFnPlugin)。
# 这是一个为了兼容性而必须保留的特例。
from gskin.src import cSkinDeform2 as cSkinDeform

NODE_NAME = "cSkinDeformer"
# MTypeId 必须使用旧版 API 的对象
NODE_ID = om1.MTypeId(0x00080033)


def nodeCreator():
    # creator 必须返回一个由 asMPxPtr 包装的指针
    return ompx.asMPxPtr(cSkinDeform.CythonSkinDeformer())


def initializePlugin(mObj):
    # 使用旧版的 MFnPlugin
    mPlugin = ompx.MFnPlugin(mObj, "Donzy", "1.0", "Any")
    try:
        mPlugin.registerNode(
            NODE_NAME,
            NODE_ID,
            nodeCreator,
            cSkinDeform.CythonSkinDeformer.nodeInitializer,
            ompx.MPxNode.kDeformerNode,
        )
        om.MGlobal.displayInfo(f"{NODE_NAME} (OM1 Node) loaded successfully.")
    except:
        om.MGlobal.displayError(f"Failed to register node: {NODE_NAME}")
        raise


def uninitializePlugin(mObj):
    # 使用旧版的 MFnPlugin
    mPlugin = ompx.MFnPlugin(mObj)
    try:
        mPlugin.deregisterNode(NODE_ID)
        om.MGlobal.displayInfo(f"{NODE_NAME} (OM1 Node) unloaded successfully.")
    except:
        om.MGlobal.displayError(f"Failed to deregister node: {NODE_NAME}")
        raise
