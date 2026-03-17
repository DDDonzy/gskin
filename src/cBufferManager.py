import ctypes
import array


__all__ = [
    "BufferManager",
    "ensure_bytes",
    "ensure_memoryview",
]


def ensure_memoryview(data, typecode="f"):
    """
    将输入对象转换为 memoryview。

    Args:
        data: 输入对象 (list, tuple, memoryview, array.array, bytes 等)
        typecode (str): 如果输入是 list/tuple，转换时使用的 C 类型码。
                        'f' 为 float (32-bit), 'd' 为 double (64-bit), 'i' 为 int (32-bit)

    Returns:
        memoryview: 对象的内存视图
    """
    # 1. 尝试直接获取视图 (针对已支持 buffer 协议的对象: memoryview, array, numpy, bytes)
    try:
        return memoryview(data)
    except TypeError:
        pass

    # 2. 如果是 list 或 tuple，先转为连续内存的 array.array
    if isinstance(data, (list, tuple)):
        # 注意：这里会发生一次 $O(N)$ 的内存拷贝和类型转换
        return memoryview(array.array(typecode, data))

    # 3. 抛出不支持的异常
    raise TypeError(f"无法将类型 {type(data).__name__} 转换为 memoryview。支持的类型包括: list, tuple, memoryview, array.array, bytes, bytearray 等。")


def ensure_bytes(data, typecode="f") -> bytes:
    """
    [终极数据洗白器]
    将任何输入 (list, tuple, memoryview, array) 强制转化为不可变的 C 字节流。
    专供 Undo 队列做只读备份使用，彻底阻断 Maya 内存崩溃。
    """
    # 1. 已经是 bytes 了，直接放行 (比如 Redo 再次推入队列时)
    if isinstance(data, bytes):
        return data

    # 2. 如果是 memoryview，瞬间 C 级别内存快照
    if isinstance(data, memoryview):
        return data.tobytes()

    # 3. 如果是 array.array，同样瞬间快照
    if isinstance(data, array.array):
        return data.tobytes()

    # 4. 如果是 list 或 tuple，必须先转为连续内存，再抽成 bytes
    if isinstance(data, (list, tuple)):
        return array.array(typecode, data).tobytes()

    raise TypeError(f"无法将 {type(data)} 转换为安全的 bytes 备份。")


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
