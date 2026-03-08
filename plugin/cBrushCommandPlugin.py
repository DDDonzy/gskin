import maya.api.OpenMaya as om

# 导入你的命令类 (路径根据你的实际目录结构调整)
from gskin.src.cBrushCommand import CallbackCmd


def maya_useNewAPI():
    """告诉Maya我们要使用API 2.0"""
    pass


def initializePlugin(mobject):
    mplugin = om.MFnPlugin(mobject)
    try:
        mplugin.registerCommand(CallbackCmd.COMMAND_NAME, CallbackCmd.creator)
    except Exception as e:
        om.MGlobal.displayError(f"无法注册命令: {CallbackCmd.COMMAND_NAME}\n{e}")


def uninitializePlugin(mobject):
    mplugin = om.MFnPlugin(mobject)

    try:
        mplugin.deregisterCommand(CallbackCmd.COMMAND_NAME)
    except Exception as e:
        om.MGlobal.displayError(f"无法注销命令: {CallbackCmd.COMMAND_NAME}\n{e}")
