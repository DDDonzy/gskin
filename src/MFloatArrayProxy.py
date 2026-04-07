from __future__ import annotations

import ctypes

import maya.OpenMaya as OpenMaya  # type: ignore

__all__ = ["MFloatArrayProxy"]


class MFloatArrayProxy:
    """
    将 `MVectorArray` 封装成 `FloatArray` 的处理器.

    Maya 节点属性不直接支持连续的 `FloatArray` 类型。

    本类通过内存映射, 将双精度 `MVectorArray` (8-byte double) 重新解释为单精度 `Float` (4-byte)

    一个 `MVector` 包含 3 个 Double (24字节), 可承载 6 个 Float。

    为了保证可以写入任意数量的 `Float`, 把 `MVectorArray` 解析位两部分, 具体内存布局如下：
    ```
    -----------------------------------------------------------------------------------------
    |  Header[0] (size_t)  |  Header[1] (size_t)  |    Header[2]   |      DATA (c_float)    |
    |----------------------|----------------------|----------------|------------------------|
    |      float count     |      data bytes      |      None      |    Actual Float Data   |
    |        8 bytes       |        8 bytes       |    8 bytes     |      data_byte_size    |
    | <------------------- 24-byte Header (1st MVector) ---------> | <------ Payload ------>|
    -----------------------------------------------------------------------------------------
    ```

    """

    _DOUBLE_SIZE = ctypes.sizeof(ctypes.c_double)
    _FLOAT_SIZE = ctypes.sizeof(ctypes.c_float)
    _MVECTOR_SIZE = 3 * _DOUBLE_SIZE
    _HEADER_SIZE = 3 * _DOUBLE_SIZE

    __slots__ = (
        "mDataHandle",
        "length",
        "view",
        "array",
        "address",
        "_src_address",
        # func
    )

    mDataHandle: OpenMaya.MDataHandle
    array: OpenMaya.MVectorArray
    view: memoryview
    length: int
    address: int
    _src_address: int

    def __init__(self, dataHandle: OpenMaya.MDataHandle):
        """
        Args:
            dataHandle (OpenMaya.MDataHandle): MDataHandle 可以来自 `MPlug.asMDataHandle` 或节点内部 `MDataBlock.inputValue`
        """
        self.mDataHandle = dataHandle
        self.length = 0
        self.view = None
        self.array = None
        self.address = 0
        self._src_address = 0

        self.init_array()

    def init_array(self):
        """
        从 `MDataHandle` 解析 `MVectorArray`
        然后将 `MVectorArray` 处理成 `float` 视图

        Updates:
            - `self.length`
            - `self.array`
            - `self.view`
            - `self.address`
            - `self._srcAddress`
        """

        # vector array object
        mObject: OpenMaya.MObject = self.mDataHandle.data()
        if mObject.isNull():
            # data handle 可能是空的
            print("MDataHandle's data is null.")
            return

        # vector array
        self.array: OpenMaya.MVectorArray = OpenMaya.MFnVectorArrayData(mObject).array()
        vAry_length = self.array.length()
        if vAry_length == 0:
            # vector array 可能是空的
            print("MVectorArray is empty.")
            return

        byte_capacity = MFloatArrayProxy._DOUBLE_SIZE * 3 * vAry_length  # vector array byte 大小
        _src_address = int(self.array[0].this)  # 第一个元素内存地址
        data_address = _src_address + MFloatArrayProxy._HEADER_SIZE  # 跳过 header

        header = ctypes.cast(_src_address, ctypes.POINTER(ctypes.c_size_t))  # 获取 header
        self.length = header[0]  # 数组大小 - float 数量
        data_byte_size = header[1]  # 不包含header的数据的大小 byte

        # 排除有问题数据
        if data_byte_size > byte_capacity - MFloatArrayProxy._HEADER_SIZE:
            raise ValueError("Data size exceeds capacity.")
        if data_byte_size % 4 != 0:
            raise ValueError("Data byte size must be a multiple of 4.")

        #
        full_view = memoryview((ctypes.c_char * data_byte_size).from_address(data_address)).cast("B").cast("f")
        self.view = full_view[: self.length]
        self._src_address = _src_address
        self.address = data_address

    def resize(self, float_count: int):
        """
        调整底层物理内存大小, 更新 Header, 并重新映射视图

        Updates:
            - `self.length`
            - `self.array`
            - `self.view`
            - `self.address`
            - `self._srcAddress`
        """
        if float_count < 0:
            raise ValueError("Float count cannot be negative.")

        old_float_count = self.length  # 记录调整前的长度

        data_byte_size = float_count * MFloatArrayProxy._FLOAT_SIZE
        # 需求bytes大小 = header(24) + 数据bytes
        total_needed_bytes = self._HEADER_SIZE + data_byte_size

        # 即使 float_count 为 0, 也会分配 1 个 Vector 给 Header
        needed_vector_count = (total_needed_bytes + self._MVECTOR_SIZE - 1) // self._MVECTOR_SIZE

        if self.array is None:  # 如果还未分配数组
            self.array = OpenMaya.MVectorArray()
            mObject = OpenMaya.MFnVectorArrayData().create(self.array)
            self.mDataHandle.setMObject(mObject)

        current_vector_count = self.array.length()
        if current_vector_count != needed_vector_count:  # 只在需要的时候扩容
            self.array.setLength(needed_vector_count)

        # 扩容后 底层内存地址可能变,重新获取
        _src_address = int(self.array[0].this)

        # 重新映射视图
        data_address = _src_address + MFloatArrayProxy._HEADER_SIZE

        # 默认值填充
        if float_count > old_float_count:
            added_float_count = float_count - old_float_count  # resize 多出来的元素数量
            bytes_to_clear = added_float_count * MFloatArrayProxy._FLOAT_SIZE
            clear_start_address = data_address + (old_float_count * self._FLOAT_SIZE)
            ctypes.memset(clear_start_address, 0, bytes_to_clear)

        # Header
        header = ctypes.cast(_src_address, ctypes.POINTER(ctypes.c_size_t))
        header[0] = float_count
        header[1] = data_byte_size

        raw_buffer = (ctypes.c_char * data_byte_size).from_address(data_address)

        # 更新实例状态
        self.length = float_count
        self.view = memoryview(raw_buffer).cast("B").cast("f")
        self._src_address = _src_address
        self.address = data_address

    @classmethod
    def from_mPlug(cls, plug: OpenMaya.MPlug) -> MFloatArrayProxy:
        """
        传入 MPlug 获取实例
        - 此方法获取的实例, 修改数据不会实时反馈到 Maya节点, 修改完后需要显示的调用 set 方法通知 Maya节点 更新数据
        """
        return cls(plug.asMDataHandle())

    @classmethod
    def from_string(cls, input_string: str) -> MFloatArrayProxy:
        """
        传入字符串获取实例
        - 此方法获取的实例, 修改数据不会实时反馈到 Maya节点, 修改完后需要显示的调用 set 方法通知 Maya节点 更新数据
        """
        sel = OpenMaya.MSelectionList()
        sel.add(input_string)
        plug = OpenMaya.MPlug()
        sel.getPlug(0, plug)
        return cls.from_mPlug(plug)

    def set_to_mPlug(self, plug: OpenMaya.MPlug, copy: bool = False):
        """
        将实例传递给用户输入的MPlug, 以通知Maya节点更新数据
        Args:
            plug (OpenMaya.MPlug): Maya MPlug
            copy (bool): True = 拷贝, False = 引用
        """
        if self.array is None:
            return False

        if copy:
            data_obj = OpenMaya.MFnVectorArrayData().create(self.array)
            plug.setMObject(data_obj)
        else:
            plug.setMDataHandle(self.mDataHandle)

        return True

    def set_to_string(self, input_string: str, copy: bool = False):
        """
        将实例传递给用户输入的字符串, 以通知Maya更新数据
        Args:
            input_string (str): 属性名
            copy (bool): True = 拷贝, False = 引用
        """
        sel = OpenMaya.MSelectionList()
        sel.add(input_string)
        plug = OpenMaya.MPlug()
        sel.getPlug(0, plug)
        return self.set_to_mPlug(plug, copy)

    @property
    def __array_interface__(self):
        """
        让底层库(如 Numpy, Cython)将此实例直接视为 C 连续数组。
        """
        if self.view is None:
            raise ValueError("Handle is not initialized.")

        return {
            "shape": (self.length,),  # 数组一维长度
            "typestr": "<f4",  # 小端序 32位浮点数 (float32)
            "data": (self.address, False),  # (内存地址, 是否只读)
            "version": 3,
        }

    def __getitem__(self, key):
        """支持 handle[i] 和切片 handle[start:end]"""
        return self.view[key]

    def __setitem__(self, key, value):
        """支持 handle[i] = 1.0"""
        self.view[key] = value

    def __len__(self):
        """支持 len(handle)"""
        return self.length

    def __iter__(self):
        """支持 for val in handle"""
        return iter(self.view)

    def tobytes(self):
        """导出为 bytes"""
        return self.view.tobytes()

    def tolist(self):
        """导出为 Python 原生 list"""
        return self.view.tolist()

    def __repr__(self) -> str:
        if self.array is None or self.view is None:
            # 如果还未分配内存
            return f"<{self.__class__.__name__} [Uninitialized]>"

        # 核心信息提取
        hex_addr = hex(self.address)
        float_count = self.length

        # 换算物理内存占用 (KB)
        byte_size = float_count * self._FLOAT_SIZE
        mb_size = byte_size / (1024)

        # 数据预览
        # 为了防止 300 万个数据直接 print 导致 Maya 卡死, 我们只截取头尾
        preview_str = ""
        if float_count == 0:
            preview_str = "[]"
        elif float_count <= 7:
            # 数据量小, 全部显示, 保留 5 位有效数字
            preview_str = "[" + ", ".join(f"{x:.5f}" for x in self.view) + "]"
        else:
            # 数据量大掐头去尾显示
            head = ", ".join(f"{self.view[i]:.5f}" for i in range(5))
            tail = ", ".join(f"{self.view[i]:.5f}" for i in range(float_count - 2, float_count))
            preview_str = f"[{head}, ......, {tail}]"

        return (f"<{self.__class__.__name__} at {hex_addr} | "
                f"size: {float_count} floats ({mb_size:.2f} KB)>\n"
                f"{' '*4}{preview_str}")  # fmt:skip
