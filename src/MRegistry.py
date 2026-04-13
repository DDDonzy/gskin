from __future__ import annotations

import sys
import typing
import weakref

from maya import OpenMaya as OpenMaya  # type:ignore
from maya.api import OpenMaya as OpenMaya2  # type:ignore

if typing.TYPE_CHECKING:
    from maya import OpenMayaMPx  # type:ignore


__all__ = ("MRegistry",)


class MRegistry:
    """
    Maya 节点开发中可以通过 `MFnDependencyNode.userNode` 获取到实例节点的对象

    可以使用这个实例对象直接访问节点属性, 获取各种缓存数据, 修改节点内部状态

    由于 api1.0 的节点实例, 在 api2.0 中无法直接通过 userNode 获取, 反之亦然

    所以用全局注册的极端方法, 抛弃userNode方式

    在节点创建时全局注册此节点, 并且绑定节点的实例

    Steps:
        register
            1. 在节点的 postConstructor 中获取 thisObject
            2. 利用 MObjectHandle 获取唯一哈希值
            3. 注册到此类的 _storage 字典中 {hash_code: python_instance}
        get_instances
            1. 传入节点的 MObject
            2. 利用 MObjectHandle 获取唯一哈希值
            3. _storage[hash_code] 获取实例
    """

    if not hasattr(sys, "_cSkinRegistry_storage"):
        sys._cSkinRegistry_storage = weakref.WeakValueDictionary()
    _storage = sys._cSkinRegistry_storage

    @staticmethod
    def get_hash(mObject: OpenMaya.MObject | OpenMaya2.MObject):
        """
        获取 MObject 的唯一哈希值
        Args:
            mObject (MObject): MObject 对象
        Returns:
            int: 哈希值
        """
        MObjectHandle = OpenMaya.MObjectHandle if isinstance(mObject, OpenMaya.MObject) else OpenMaya2.MObjectHandle

        if mObject.isNull():
            raise ValueError("MObject is null.")

        handle = MObjectHandle(mObject)

        if not handle.isAlive():
            raise ValueError("MObject is not alive/valid.")

        return handle.hashCode()

    @classmethod
    def register(cls, python_instance: OpenMayaMPx.MPxNode | OpenMaya2.MPxNode):
        """
        注册节点实例
        Args:
            python_instance (MPxNode): 节点开发定义的 Python 实例
        """

        mObject = python_instance.thisMObject()
        cls._storage[cls.get_hash(mObject)] = python_instance
        return True

    @classmethod
    def get_instance(cls, node_input: str | OpenMaya.MObject | OpenMaya2.MObject):
        """
        通过 MObject 获取实例
        Args:
            input (str|MObject): MObject 对象 或 节点名称
        Returns:
            instance (MPxNode): 节点开发定义的 Python 实例
        """
        if isinstance(node_input, str):
            mObject = cls._get_mObject_by_string(node_input)
        elif isinstance(node_input, (OpenMaya.MObject, OpenMaya2.MObject)):
            mObject = node_input

        hash_code = cls.get_hash(mObject)

        return cls._storage.get(hash_code, None)

    @staticmethod
    def _get_mObject_by_string(input_string: str) -> OpenMaya2.MObject:
        """
        通过字符串获取 Maya.api.OpenMaya.Object
        Args:
            input_string (str): 节点名称
        Return:
            mObject (MObject): Maya.api.OpenMaya.Object 对象
        """
        try:
            sel: OpenMaya2.MSelectionList = OpenMaya2.MGlobal.getSelectionListByName(input_string)
        except RuntimeError:
            raise RuntimeError(f"Object '{input_string}' does not exist") from None
        return sel.getDependNode(0)
