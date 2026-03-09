from importlib import reload
import gskin.src.cWeightsManager as wm
from maya.api import OpenMaya as om

reload(wm)

cSkin = "cSkinDeformer1"

manager = wm.WeightsManager.from_string(cSkin)
manager.layers

manager.add_layer()