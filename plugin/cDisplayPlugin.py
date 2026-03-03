import maya.api.OpenMaya as om
import maya.api.OpenMayaRender as omr

# 统一使用相对路径和模块导入

from gskin.src import cDisplayNode as cDisplay

# 定义插件元数据
PLUGIN_NAME = "cDisplayNodePlugin"
NODE_NAME = "WeightPreviewShape"
NODE_ID = om.MTypeId(0x80005)

DRAW_CLASSIFICATION = "drawdb/geometry/WeightPreview"
DRAW_REGISTRAR_ID = "WeightPreviewShapeRegistrar"

def maya_useNewAPI():
    pass

def initializePlugin(mobject):
    """Initializes the plug-in.
    Registers the custom node and its UI override.
    """
    plugin = om.MFnPlugin(mobject, "Donzy", "1.0", "Any")
    try:
        plugin.registerShape(
            NODE_NAME,
            NODE_ID,
            cDisplay.WeightPreviewShape.creator,
            cDisplay.WeightPreviewShape.initialize,
            cDisplay.WeightPreviewShapeUI.creator,
            DRAW_CLASSIFICATION
        )
        om.MGlobal.displayInfo(f"{NODE_NAME} shape node loaded successfully.")

        omr.MDrawRegistry.registerGeometryOverrideCreator(
            DRAW_CLASSIFICATION,
            DRAW_REGISTRAR_ID,
            cDisplay.WeightGeometryOverride.creator
        )
        om.MGlobal.displayInfo(f"{NODE_NAME} draw override registered successfully.")

    except Exception as e:
        om.MGlobal.displayError(f"Failed to initialize plugin {PLUGIN_NAME}: {e}")
        raise

def uninitializePlugin(mobject):
    """Uninitializes the plug-in.
    Deregisters the node and its UI override.
    """
    plugin = om.MFnPlugin(mobject)
    try:
        # 遵循更安全的卸载顺序: 先注销绘制覆盖, 再注销节点
        omr.MDrawRegistry.deregisterGeometryOverrideCreator(
            DRAW_CLASSIFICATION,
            DRAW_REGISTRAR_ID
        )
        om.MGlobal.displayInfo(f"{NODE_NAME} draw override deregistered successfully.")

        plugin.deregisterNode(NODE_ID)
        om.MGlobal.displayInfo(f"{NODE_NAME} shape node unloaded successfully.")
        
    except Exception as e:
        om.MGlobal.displayError(f"Failed to uninitialize plugin {PLUGIN_NAME}: {e}")
        raise
