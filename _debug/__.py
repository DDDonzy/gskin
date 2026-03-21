from importlib import reload
import re

import maya.cmds as cmds
import maya.api.OpenMaya as om2
import maya.api.OpenMayaAnim as oma2
from gskin.src import cWeightsManager as wm
from gskin.src import cBufferManager as cb

import gskin._debug.gskinReload as gskinReload
from gskin._debug.convert import get_skinWeights, convert_skin_to_cSkin
from gskin.src._cRegistry import SkinRegistry
import numpy as np


maya_file = r"C:/Users/ext.dxu/Desktop/ng_test.ma"


gskinReload.reload_modules_in_path()
gskinReload.reload_all_plugins()

# test file
cmds.file(maya_file, o=1, f=1)


cmds.refresh()
# convert
sk_node = "skinCluster1"
shape = cmds.skinCluster(sk_node, q=1, g=1)[0]
cSkin = convert_skin_to_cSkin(sk_node)


cmds.refresh()

# set weights
maya_weights, _ = get_skinWeights(sk_node)
vertex_count = cmds.polyEvaluate(shape, vertex=True)
influence_indices = cmds.getAttr(f"{sk_node}.matrix", mi=1)
# --- 修正后的调用代码 ---
manager = wm.WeightsManager.from_node("cSkinDeformer1")

# 1. 正常调用重建 (这步会完成内存里的 0-Copy 覆写)
manager.init_handle_data(-1, 0, vertex_count, len(influence_indices), influence_indices, list(maya_weights))

# 2. 🌟 关键：手动执行固化同步！
# 从 manager 拿到对应的 handle
handle = manager.get_handle(-1, 0)

# 3. 强制固化到 MPlug (将内存里的 mObject_data 拍进插座)
handle.commit()

# 4. 踢醒 Maya 刷新视口
manager.updateDG()