import ctypes


__all__ = [
    "BufferManager",
]


class BufferManager:
    """
    一个高效的、零拷贝的内存管理器。
    它负责 C 级别连续内存的申请、生命周期管理和 Pythonic 视图生成。
    核心目标是为 Cython/C++ 模块提供安全、高效的裸指针访问，同时防止内存被 Python GC 意外回收。
    """

    # --- 优化: 使用 __slots__ 替代 __dict__，提升属性访问速度并降低内存占用 ---
    __slots__ = ("ctypes", "view", "ptr", "format_char", "shape")

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
        self.ctypes = None
        self.view: memoryview = None
        self.ptr: int = 0
        self.format_char: str = ""
        self.shape: tuple = ()

    @staticmethod
    def auto(data, format_char: str = "f", shape: tuple = None):
        """
        [万能路由函数]
        根据输入对象类型自动选择最优的构建路径：
        1. 如果是 BufferManager 实例：直接返回。
        2. 如果是 int：视为内存地址，调用 from_ptr (零拷贝)。
        3. 如果支持 Buffer 协议 (array, cython memoryview, bytes)：调用 from_buffer (严格零拷贝)。
        4. 如果是 list/tuple 或 Maya API 1.0 数组：调用 from_list (显式拷贝)。
        """
        if data is None:
            return BufferManager()

        # --- 路径 1: 实例直接穿透 (最快) ---
        if isinstance(data, BufferManager):
            return data

        # --- 路径 2: 裸指针 (零拷贝) ---
        if isinstance(data, int):
            return BufferManager.from_ptr(data, format_char, shape)

        # --- 路径 3: 显式拷贝路径 (通过类型检测分流，避免 try...except 的开销) ---
        # 优先拦截不支持 Buffer 协议且需要拷贝的类型（如 list 或 Maya API 1.0 对象）
        data_type = type(data)
        if issubclass(data_type, (list, tuple)) or "maya.OpenMaya" in str(data_type):
            return BufferManager.from_list(data, format_char, shape)

        # --- 路径 4: Buffer 协议桥接 (零拷贝) ---
        # 适配 array.array, numpy, cython views 等。
        # 因为绝大多数高性能数据都支持 buffer，进入此路径时发生异常的概率极低。
        try:
            return BufferManager.from_buffer(data, format_char, shape)
        except (TypeError, ValueError):
            pass

        raise TypeError(f"BufferManager.auto 无法转换输入类型: {data_type}")

    @staticmethod
    def allocate(format_char: str, shape: tuple):
        instance = BufferManager()
        if format_char not in BufferManager._CTYPES_MAP:
            raise ValueError(f"Unsupported format character: {format_char}")

        total_elements = 1
        for dim in shape:
            total_elements *= dim

        if total_elements <= 0:
            ctype_base = BufferManager._CTYPES_MAP[format_char]
            instance.ctypes = (ctype_base * 0)()
            instance.ptr = ctypes.addressof(instance.ctypes)
            instance.format_char = format_char
            instance.shape = shape
            instance.view = memoryview(instance.ctypes).cast("B").cast(format_char, shape=shape)
            return instance

        ctype_base = BufferManager._CTYPES_MAP[format_char]
        instance.ctypes = (ctype_base * total_elements)()
        instance.ptr = ctypes.addressof(instance.ctypes)
        instance.format_char = format_char
        instance.shape = shape
        instance.view = memoryview(instance.ctypes).cast("B").cast(format_char, shape=shape)
        return instance

    @staticmethod
    def from_list(data_list: list, format_char: str = "f", shape: tuple = None):
        safe_list = data_list if data_list else []
        count = len(safe_list)
        final_shape = shape if shape is not None else (count,)

        # 统一使用 allocate，确保 ctypes 永远是 ctypes 对象
        instance = BufferManager.allocate(format_char, final_shape)
        if count > 0:
            # ctypes 数组支持高效的切片批量赋值
            instance.ctypes[:] = safe_list
        return instance

    @staticmethod
    def from_buffer(data, format_char: str, shape: tuple = None):
        """
        [严格零拷贝] 将支持 Buffer 协议的对象 (如 array.array 或 Cython 返回的数组)
        包装为 BufferManager。它会确保底层的 ctypes 是 ctypes 对象，
        从而兼容 ctypes.addressof()，同时与原数据共享内存。

        注意：如果 data 不支持可写 buffer 协议，将抛出 TypeError。
        """
        if data is None:
            return BufferManager()

        # 1. 探测字节大小
        mv = memoryview(data)
        ctype_base = BufferManager._CTYPES_MAP[format_char]
        item_size = ctypes.sizeof(ctype_base)
        num_elements = mv.nbytes // item_size

        instance = BufferManager()
        instance.format_char = format_char
        instance.shape = shape if shape is not None else (num_elements,)

        # 2. 核心魔法：严格执行零拷贝包装。身份转换为原生的 ctypes 数组。
        instance.ctypes = (ctype_base * num_elements).from_buffer(data)

        instance.ptr = ctypes.addressof(instance.ctypes)
        instance.view = memoryview(instance.ctypes).cast("B").cast(format_char, shape=instance.shape)
        return instance

    @staticmethod
    def from_ptr(address: int, format_char: str, shape: tuple):
        """
        从已存在的内存地址创建一个只读的内存视图管理器。
        注意: 此方法不管理内存的生命周期，调用者需确保该内存地址持续有效。
        """
        instance = BufferManager()
        if not address or not shape:
            return instance

        total_elements = 1
        for dim in shape:
            total_elements *= dim
        if total_elements <= 0:
            return instance

        # 1. 计算内存大小和 C 类型
        ctype_base = BufferManager._CTYPES_MAP[format_char]

        # 2. 从裸指针创建 ctypes 数组 (无拷贝)
        c_array_type = ctype_base * total_elements
        instance.ctypes = c_array_type.from_address(address)

        # 3. 记录基础信息
        instance.ptr = address
        instance.format_char = format_char
        instance.shape = shape

        # 4. 生成零拷贝视图
        # 💥 优化: ctypes 对象原生支持 memoryview 协议，无需二次转换
        instance.view = memoryview(instance.ctypes).cast("B").cast(format_char, shape=shape)

        return instance

    @staticmethod
    def from_ctypes(ctypes_array, format_char: str = "i", shape: tuple = None):
        """
        [终极 0 拷贝接收器]
        直接接管由 Cython/C 极速运算后返回的原生 ctypes 连续内存数组。
        完全没有内存拷贝，直接提取底层裸指针和内存视图，并将生命周期安全地绑定到本实例。

        Args:
            ctypes_array: 原生的 ctypes 数组实例 (例如 (c_int * 10)())
            format_char (str): 目标类型码，默认为 "i" (32位整型)
            shape (tuple): 形状。如果为 None，则自动使用一维数组的长度
        """
        instance = BufferManager()
        if ctypes_array is None:
            return instance

        # 1. 直接接管 ctypes 对象，这极其重要！
        # 只要 BufferManager 活着，这块 C 内存就不会被 Python GC 回收
        instance.ctypes = ctypes_array

        # 2. 直接获取底层物理内存的裸指针 (极速)
        instance.ptr = ctypes.addressof(ctypes_array)
        instance.format_char = format_char

        # 3. 自动推断 Shape
        instance.shape = shape if shape is not None else (len(ctypes_array),)

        # 4. 生成统一标准的 memoryview 视图
        # 采用你源码中经典的 "B" (Bytes) 中转强转法，确保维度绝对安全
        instance.view = memoryview(ctypes_array).cast("B").cast(format_char, shape=instance.shape)

        return instance

    def reshape(self, new_shape: tuple) -> "BufferManager":
        """
        在不改变底层数据的情况下，返回一个具有新维度的内存管理器实例。
        这是一个零拷贝操作。
        """
        # 创建一个新的实例来持有新的视图，但共享底层的 ctypes 对象引用
        new_instance = BufferManager()
        new_instance.ctypes = self.ctypes  # 共享引用，防止 GC
        new_instance.ptr = self.ptr
        new_instance.format_char = self.format_char
        new_instance.shape = new_shape
        new_instance.view = self.view.cast("B").cast(self.format_char, shape=new_shape)
        return new_instance

    def slice(self, start: int, end: int = None) -> "BufferManager":
        """
        [极速零拷贝切片]
        自动计算物理地址偏移，返回一个拥有独立精确 ptr 的子 Manager。
        完美兼容 ctypes.memset 和 fill() 等绝对地址操作。
        """
        if self.view is None or not self.ptr:
            return BufferManager()

        # 假设主要操作的是一维展平的视图
        total_elements = self.shape[0] if self.shape else len(self.view)

        if end is None:
            end = total_elements

        # 支持 Python 的负数索引机制 (例如 -1 代表最后一个)
        if start < 0:
            start += total_elements
        if end < 0:
            end += total_elements

        # 边界安全钳制
        start = max(0, min(start, total_elements))
        end = max(0, min(end, total_elements))

        count = end - start
        if count <= 0:
            return BufferManager()

        # 💥 核心魔法：让底层物理指针跟着切片一起移动！
        item_size = self.view.itemsize
        new_ptr = self.ptr + (start * item_size)

        # 返回一个带有精确指针偏移的全新 Manager
        return BufferManager.from_ptr(new_ptr, self.format_char, (count,))

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

    @property
    def nbytes(self) -> int:
        """获取缓冲区总字节数"""
        return self.view.nbytes if self.view else 0

    def copy_to(self, dest_ptr: int):
        """
        [极速拷贝] 将当前缓冲区的内容拷贝到指定的内存地址。
        常用于将数据从 CPU 内存同步到 GPU 映射的地址。
        """
        if not self.ptr or not dest_ptr:
            return
        sz = self.view.nbytes
        if sz > 0:
            ctypes.memmove(dest_ptr, self.ptr, sz)

    def __repr__(self) -> str:
        return f"<CMemoryManager ptr=0x{self.ptr:X} format='{self.format_char}' shape={self.shape}>"
