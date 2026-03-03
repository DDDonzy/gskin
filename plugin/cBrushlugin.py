import maya.api.OpenMaya as om

# 统一使用相对路径和模块导入
from gskin.src.cBrush import WeightBrushContextCmd

# 定义插件元数据
PLUGIN_NAME = "cBrushContextPlugin"

def maya_useNewAPI():
    """告诉Maya我们要使用API 2.0"""
    pass

def initializePlugin(mobject):
    """Initializes the plug-in."""
    plugin = om.MFnPlugin(mobject, "Donzy", "1.0", "Any")
    try:
        plugin.registerContextCommand(
            WeightBrushContextCmd.COMMAND_NAME,
            WeightBrushContextCmd.creator,
        )
        om.MGlobal.displayInfo(f"{WeightBrushContextCmd.COMMAND_NAME} context loaded successfully.")
    except:
        om.MGlobal.displayError(f"Failed to register context: {WeightBrushContextCmd.COMMAND_NAME}")
        raise

def uninitializePlugin(mobject):
    """Uninitializes the plug-in."""
    plugin = om.MFnPlugin(mobject)
    try:
        plugin.deregisterContextCommand(WeightBrushContextCmd.COMMAND_NAME)
        om.MGlobal.displayInfo(f"{WeightBrushContextCmd.COMMAND_NAME} context unloaded successfully.")
    except:
        om.MGlobal.displayError(f"Failed to deregister context: {WeightBrushContextCmd.COMMAND_NAME}")
        raise
