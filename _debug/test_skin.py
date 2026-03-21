from importlib import reload

import maya.cmds as cmds
import maya.api.OpenMaya as om2
import maya.api.OpenMayaAnim as oma2
from gskin.src import cWeightsManager as wm

import gskin._debug.gskinReload as gskinReload
from gskin._debug.convert import get_skinWeights, convert_skin_to_cSkin


maya_file = r"C:/Users/Donzy/Desktop/ng_test.ma"


gskinReload.reload_modules_in_path()
gskinReload.reload_all_plugins()

# test file
cmds.file(maya_file, o=1, f=1)
    
# convert
sk_node = "skinCluster1"
shape = cmds.skinCluster(sk_node, q=1, g=1)[0]
cSkin = convert_skin_to_cSkin(sk_node)




# set weights
maya_weights, _ = get_skinWeights(sk_node)
vertex_count = cmds.polyEvaluate(shape, vertex=True)
influence_indices = cmds.getAttr(f"{sk_node}.matrix", mi=1)
manager = wm.WeightsManager.from_node(f"{'cSkinDeformer1'}")
manager.init_handle_data(-1,0,vertex_count,2, influence_indices, list(maya_weights))





# display


cmds.createNode("WeightPreviewShape")
cmds.connectAttr("cSkinDeformer1.outputGeometry[0]", "WeightPreviewShape1.inDeformMesh")
cmds.setAttr("WeightPreviewShape1.layer", -1)


cmds.setToolTo("selectSuperContext")
if cmds.contextInfo("cBrush", exists=True):
    cmds.deleteUI("cBrush", toolContext=True)


cmds.select("pCube1")

if not cmds.contextInfo("cBrush", exists=True):
    cmds.cBrushCtx("cBrush")


cmds.setToolTo("cBrush")
