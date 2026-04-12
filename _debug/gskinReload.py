import os
import sys
from pathlib import Path
from importlib import reload

import maya.cmds as cmds
import maya.api.OpenMaya as om


def reload_all_plugins(root_dir=Path(r"E:\d_maya\gskin\plugin")):
    cmds.file(new=True, force=True)

    plugin_paths = list(root_dir.glob("*Plugin.py"))

    if not plugin_paths:
        om.MGlobal.displayWarning(f"在 {root_dir} 下未找到任何 *Plugin.py 文件！")
        return

    # 遍历卸载
    print("=================== UNLOAD ===========================")
    for plugin in plugin_paths:
        if cmds.pluginInfo(plugin.name, q=True, loaded=True):
            try:
                cmds.unloadPlugin(plugin.name)
                print(f"SUCCESS UNLOAD: {plugin.name}")
            except Exception:
                om.MGlobal.displayError("FAILED UNLOAD: {plugin.name}")
    print("==================== LOAD ============================")
    # 遍历加载
    for plugin in plugin_paths:
        try:
            cmds.loadPlugin(str(plugin))
            print(f"SUCCESS LOAD: {plugin.name}")
        except Exception as e:
            om.MGlobal.displayError(f"FAILED LOAD: {plugin.name} - {e}")
    print("====================================================")


def reload_modules_in_path(target_dir=r"E:\d_maya\gskin\src"):
    """
    自动重载指定路径下的所有已加载的 Python 模块。

    :param target_dir: 目标文件夹的绝对路径 (例如: r"E:\d_maya\z_np")
    :return: 成功重载的模块名称列表
    """
    # 1. 路径标准化（非常重要！消除反斜杠 \ 和斜杠 /，以及大小写的差异）
    target_dir = os.path.normpath(os.path.abspath(target_dir))
    # 在 Windows 上统一转换为小写进行比对，防止 C: 和 c: 的差异
    target_dir_lower = os.path.normcase(target_dir)

    reloaded_list = []

    # 2. 遍历当前内存中所有已加载的模块
    # 注意：必须用 list() 包裹，防止在遍历字典时发生尺寸改变的 RuntimeError
    for mod_name, mod in list(sys.modules.items()):
        # 跳过 None 和没有 __file__ 属性的内建模块 (如 sys, builtins)
        if mod is None or not hasattr(mod, "__file__") or mod.__file__ is None:
            continue
        if mod_name == "gskin.src._cRegistry":
            continue

        # 3. 获取模块的文件路径并标准化
        mod_file_path = os.path.normpath(os.path.abspath(mod.__file__))
        mod_file_lower = os.path.normcase(mod_file_path)

        # 4. 判断该模块的物理路径是否在我们的目标文件夹内
        if mod_file_lower.startswith(target_dir_lower):
            try:
                # 5. 执行重载
                reload(mod)
                reloaded_list.append(mod_name)
                print(f"[Reload Success] {mod_name}")
            except Exception as e:
                # 如果代码有语法错误，重载会失败，捕获错误以防中断整个循环
                print(f"[Reload Failed] {mod_name} -> Error: {e}")

    print(f"\n--- Total Reloaded: {len(reloaded_list)} modules ---")
    return reloaded_list


if __name__ == "__main__":
    reload_modules_in_path()
    reload_all_plugins()
