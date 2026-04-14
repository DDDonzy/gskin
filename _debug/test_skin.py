import maya.cmds as cmds
import gskin._debug.gskinReload as gskinReload

from gskin.src.MRegistry import MRegistry
from gskin.src.cSkinDeform2 import FnCSkinDeform



maya_file = r"C:/Users/ext.dxu/Desktop/ng_test.ma"



gskinReload.reload_modules_in_path()
gskinReload.reload_all_plugins()

# test file
cmds.file(maya_file, o=1, f=1)

FnCSkinDeform.create_cSkinDeform_from_skinCluster("skinCluster1")