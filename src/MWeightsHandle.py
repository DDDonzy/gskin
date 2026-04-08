import array

import maya.OpenMaya as OpenMaya  # type: ignore

from gskin.src.MFloatArrayProxy import MFloatArrayProxy


class MWeightsHandle(MFloatArrayProxy):
    __slots__ = (
        "num_vertices",
        "num_influences",
    )
    num_vertices: int
    num_influences: int

    def __init__(self, dataHandle, num_vertices):
        """
        Args:
            dataHandle (OpenMaya.MDataHandle): MDataHandle 可以来自 `MPlug.asMDataHandle` 或节点内部 `MDataBlock.inputValue`
            num_vertices (int): 模型顶点数量
        """

        if num_vertices <= 0:
            raise ValueError(f"num_vertices must be greater than 0, got {num_vertices}.")

        super().__init__(dataHandle)

        if self.length % num_vertices != 0:
            raise ValueError(f"Data length mismatch! Total float count ({self.length}) is not exactly divisible by num_vertices ({num_vertices}). Cannot determine a valid num_influences.")

        if self.is_initialized:
            self.num_vertices = num_vertices
            self.num_influences = self.length // self.num_vertices
        else:
            self.num_vertices = 0
            self.num_influences = 0

    def set_array(self, *args, **kwargs):
        raise NotImplementedError("This method is disabled, Please use set_weights.")

    def set_weights(self, weights):
        """
        设置权重数据
        Args:
            weights (list|array.array|memoryview): 权重数据列表
        Update:
            `- self.array`
        """
        if len(weights) != self.length:
            raise ValueError(f"Weights length mismatch. Expected {self.length}, got {len(weights)}.")
        try:
            self.view[:] = weights
        except (TypeError, ValueError):
            self.view[:] = array.array("f", weights)

    def get_influence_weights(self, influence_index: int) -> list:
        """
        获取指定骨骼的权重, 返回的是原始数据的拷贝

        Args:
            influence_index (int): 骨骼索引
        Return:
            weights (list): 权重数据列表
        """
        return self.get_influence_weights_raw(influence_index).tolist()

    def get_influence_weights_raw(self, influence_index: int) -> memoryview:
        """
        获取指定骨骼的权重, 返回的是原始数据的视图

        Args:
            influence_index (int): 骨骼索引
        Return:
            weights (memoryview): 权重数据视图
        """
        if (  influence_index < 0 
           or influence_index >= self.num_influences):  # fmt:skip
            raise IndexError(f"Influence index {influence_index} out of range. Must be between 0 and {self.num_influences - 1}.")

        return self.view[influence_index :: self.num_influences]

    def get_weights(self) -> list:
        """
        获取权重, 返回的是原始数据的拷贝

        Return:
            weights (list): 权重数据列表
        """
        return self.get_weights_raw().tolist()

    def get_weights_raw(self) -> memoryview:
        """
        获取权重, 返回的是原始数据的视图

        Return:
            weights (memoryview): 权重数据视图
        """
        return self.view

    def remap_influences(
        self,
        source_influence_indices: list,
        target_influence_indices: list,
    ):
        """
        此函数会根据输入的`source_influence_indices` 和 `target_influence_indices` 对权重数据重构

        添加骨骼/删除骨骼, 用这个函数对权重重构, 以确保不会丢失权重数据且正确

        新增骨骼默认权重为 0.0

        Args:
            source_influence_indices: 当前句柄中对应的骨骼标识列表 (可以是逻辑索引 int, 也可以是骨骼名字 str)
            target_influence_indices: 期望的目标骨骼标识列表

        Update:
            - `self.length`
            - `self.array`
            - `self.view`
            - `self.address`
            - `self._srcAddress`
        """
        old_num_influences = self.num_influences
        new_num_influences = len(target_influence_indices)
        num_vertices = self.num_vertices

        # 确保源长度和当前数据列数对应
        if len(source_influence_indices) != old_num_influences:
            raise ValueError(f"Length of source_influence_indices ({len(source_influence_indices)}) must match current num_influences ({old_num_influences}).")

        # 映射字典
        source_map = {influence_index: i for i, influence_index in enumerate(source_influence_indices)}
        # 开辟全新的, 默认全为 0.0 的内存块
        new_data = array.array("f", [0.0]) * (num_vertices * new_num_influences)
        # 遍历目标列表, 按需搬运数据
        for i, target_influence_index in enumerate(target_influence_indices):
            old_influence_index = source_map.get(target_influence_index)

            if old_influence_index is not None:
                # 这根骨骼在原来的列表里存在,保留或挪动位置, 将旧列数据 整列 复制到新列
                new_data[i::new_num_influences] = array.array("f", self.view[old_influence_index::old_num_influences])
            else:
                # 这是一根全新的骨骼,什么都不用做因为 new_data 就是 0.0
                pass

        # 调整底层物理内存大小
        self.resize(len(new_data))

        # 新重组的数据并更新状态
        self.view[:] = new_data
        self.num_influences = new_num_influences

    @classmethod
    def from_mPlug(cls, plug: OpenMaya.MPlug, num_vertices: int) -> MFloatArrayProxy:
        """
        传入 MPlug 获取实例
        - 此方法获取的实例, 修改数据不会实时反馈到 Maya, 修改完后需要显示的调用 set 方法通知 Maya 更新数据
        """
        return cls(plug.asMDataHandle(), num_vertices)

    @classmethod
    def from_string(cls, input_string: str, num_vertices: int) -> MFloatArrayProxy:
        """
        传入字符串获取实例
        - 此方法获取的实例, 修改数据不会实时反馈到 Maya, 修改完后需要显示的调用 set 方法通知 Maya 更新数据
        """
        sel = OpenMaya.MSelectionList()
        sel.add(input_string)
        plug = OpenMaya.MPlug()
        sel.getPlug(0, plug)
        return cls.from_mPlug(plug, num_vertices)

    def __repr__(self) -> str:
        res = super().__repr__()
        if self.is_initialized:
            return (f"{res}\n"
                    f"{' '*4}num_vertices  : {self.num_vertices}\n"
                    f"{' '*4}num_influences: {self.num_influences}\n")  # fmt:skip
        return res
