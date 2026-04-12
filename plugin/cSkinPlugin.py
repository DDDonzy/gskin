import maya.api.OpenMaya as om
import maya.OpenMayaMPx as ompx

import gskin.src.cSkinDeform2 as CSkinDeformModules
CSkinDeform = CSkinDeformModules.CSkinDeform


def initializePlugin(mObj):
    # 使用旧版的 MFnPlugin
    mPlugin = ompx.MFnPlugin(mObj, "Donzy", "1.0", "Any")
    try:
        mPlugin.registerNode(
            CSkinDeform.NODE_TYPE,
            CSkinDeform.NODE_ID,
            CSkinDeform.nodeCreator,
            CSkinDeform.nodeInitializer,
            ompx.MPxNode.kDeformerNode,
        )
        om.MGlobal.displayInfo(f"{CSkinDeform.NODE_TYPE} (OM1 Node) loaded successfully.")
    except:
        om.MGlobal.displayError(f"Failed to register node: {CSkinDeform.NODE_TYPE}")
        raise


def uninitializePlugin(mObj):
    # 使用旧版的 MFnPlugin
    mPlugin = ompx.MFnPlugin(mObj)
    try:
        mPlugin.deregisterNode(CSkinDeform.NODE_ID)
        om.MGlobal.displayInfo(f"{CSkinDeform.NODE_TYPE} (OM1 Node) unloaded successfully.")
    except:
        om.MGlobal.displayError(f"Failed to deregister node: {CSkinDeform.NODE_TYPE}")
        raise
