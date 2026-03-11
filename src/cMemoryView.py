import ctypes
import array


class CMemoryManager:
    """
    一个高效的、零拷贝的内存管理器。
    它负责 C 级别连续内存的申请、生命周期管理和 Pythonic 视图生成。
    核心目标是为 Cython/C++ 模块提供安全、高效的裸指针访问，同时防止内存被 Python GC 意外回收。
    """

    # --- 优化: 使用 __slots__ 替代 __dict__，提升属性访问速度并降低内存占用 ---
    __slots__ = ("_cache", "view", "ptr", "format_char", "shape")

    _CTYPES_MAP = {
        "d": ctypes.c_double,
        "f": ctypes.c_float,
        "i": ctypes.c_int32,
        "I": ctypes.c_uint32,
        "b": ctypes.c_int8,
        "B": ctypes.c_uint8,
    }

    def __init__(self):
        """初始化一个空的内存管理器实例。"""
        self._cache = None
        self.view: memoryview = None
        self.ptr: int = 0
        self.format_char: str = ""
        self.shape: tuple = ()

    @staticmethod
    def allocate(format_char: str, shape: tuple):
        instance = CMemoryManager()
        if format_char not in CMemoryManager._CTYPES_MAP:
            raise ValueError(f"Unsupported format character: {format_char}")

        total_elements = 1
        for dim in shape:
            total_elements *= dim

        if total_elements <= 0:
            instance._cache = array.array(format_char)
            instance.ptr, _ = instance._cache.buffer_info()
            instance.format_char = format_char
            instance.shape = shape
            instance.view = memoryview(instance._cache).cast(format_char)
            return instance

        ctype_base = CMemoryManager._CTYPES_MAP[format_char]
        instance._cache = (ctype_base * total_elements)()
        instance.ptr = ctypes.addressof(instance._cache)
        instance.format_char = format_char
        instance.shape = shape
        instance.view = memoryview(instance._cache).cast("B").cast(format_char, shape=shape)
        return instance

    @staticmethod
    def from_list(data_list: list, format_char: str = "f", shape: tuple = None):
        instance = CMemoryManager()

        safe_list = data_list if data_list else []

        instance._cache = array.array(format_char, safe_list)
        instance.ptr, _ = instance._cache.buffer_info()
        instance.format_char = format_char

        final_shape = shape if shape is not None else (len(safe_list),)
        instance.shape = final_shape
        instance.view = memoryview(instance._cache).cast("B").cast(format_char, shape=final_shape)

        return instance

    @staticmethod
    def from_ptr(address: int, format_char: str, shape: tuple):
        """
        从已存在的内存地址创建一个只读的内存视图管理器。
        注意: 此方法不管理内存的生命周期，调用者需确保该内存地址持续有效。
        """
        instance = CMemoryManager()
        if not address or not shape:
            return instance

        total_elements = 1
        for dim in shape:
            total_elements *= dim
        if total_elements <= 0:
            return instance

        # 1. 计算内存大小和 C 类型
        ctype_base = CMemoryManager._CTYPES_MAP[format_char]

        # 2. 从裸指针创建 ctypes 数组 (无拷贝)
        c_array_type = ctype_base * total_elements
        instance._cache = c_array_type.from_address(address)

        # 3. 记录基础信息
        instance.ptr = address
        instance.format_char = format_char
        instance.shape = shape

        # 4. 生成零拷贝视图
        # 💥 优化: ctypes 对象原生支持 memoryview 协议，无需二次转换
        instance.view = memoryview(instance._cache).cast("B").cast(format_char, shape=shape)

        return instance

    def reshape(self, new_shape: tuple) -> "CMemoryManager":
        """
        在不改变底层数据的情况下，返回一个具有新维度的内存管理器实例。
        这是一个零拷贝操作。
        """
        # 创建一个新的实例来持有新的视图，但共享底层的 _cache
        new_instance = CMemoryManager()
        new_instance._cache = self._cache  # 共享引用，防止 GC
        new_instance.ptr = self.ptr
        new_instance.format_char = self.format_char
        new_instance.shape = new_shape
        new_instance.view = self.view.cast("B").cast(self.format_char, shape=new_shape)
        return new_instance

    def fill(self, value):
        """
        [极速填充] 将当前 Manager 掌管的所有内存区域填充为指定值。
        如果是局部清理，请通过 from_ptr 生成 Sub-Manager 后再调用此方法。
        """
        if not self.ptr or self.view is None:
            return

        total_bytes = self.view.nbytes
        if total_bytes == 0:
            return

        # --- 路径 1: 纯零填充 (直接使用底层的 memset，物理极限速度) ---
        if value == 0 or value == 0.0:
            ctypes.memset(self.ptr, 0, total_bytes)
            return

        # --- 路径 2: 非零填充 (利用对数级倍增复制) ---
        element_size = self.view.itemsize
        total_elements = total_bytes // element_size

        # 给第 0 个元素赋上种子值
        flat_view = self.view.cast("B").cast(self.format_char)
        flat_view[0] = value

        # 爆发式倍增 (1变2, 2变4, 4变8...)
        filled = 1
        while filled < total_elements:
            to_fill = min(filled, total_elements - filled)
            ctypes.memmove(self.ptr + filled * element_size, self.ptr, to_fill * element_size)
            filled += to_fill

    def __repr__(self) -> str:
        return f"<CMemoryManager ptr=0x{self.ptr:X} format='{self.format_char}' shape={self.shape}>"
