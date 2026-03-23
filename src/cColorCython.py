
# ==============================================================================
# 🎨 cColor.py - 纯 Python 语法版 (支持 Cython 极限编译)
# ==============================================================================
import cython
import typing

# --- 核心优化 定义静态颜色梯度表 ---
# 格式  (权重上限, (R, G, B))
# 表中每一项定义了一个颜色“停靠点”。权重值会在此表定义的颜色之间进行平滑的线性插值。
# 这个表可以轻松地进行扩展和修改 以实现不同的热力图效果。
GRADIENT_TABLE: typing.List[typing.Tuple[cython.float, typing.Tuple[cython.float, cython.float, cython.float]]] = [  # noqa: UP006
    (0.0,   (0.0, 0.0, 1.0)),   # 权重 0.00 -> 蓝
    (0.4,   (0.0, 1.0, 0.0)),   # 权重 0.25 -> 绿
    (0.6,   (1.0, 1.0, 0.0)),   # 权重 0.50 -> 黄
    (0.75,  (1.0, 0.5, 0.0)),   # 权重 0.75 -> 橙
    (1.0,   (1.0, 0.0, 0.0)),   # 权重 1.00 -> 红
]

# --- Cython 3.0 纯C静态数据区 ---
# 使用 cython.declare 在.py文件中声明C级别的数据结构
# 这是一个5x4的C浮点数组 用于存储上述梯度表 以便在nogil环境中高速访问
# --- Cython 3.0 纯C静态数据区 ---
# 使用 cython.declare 在.py文件中声明C级别的数据结构
# 这是一个5x4的C浮点数组 用于存储上述梯度表 以便在nogil环境中高速访问
gradient_c_array: cython.float[5][4] = cython.declare(cython.float[5][4])
# 一个C布尔标志 确保数组只被初始化一次
gradient_initialized: cython.bint = cython.declare(cython.bint)
gradient_initialized = False


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def render_heatmap(
    weights_1d: cython.float[:],
    color_view: cython.float[:, :],
):
    '''优雅、高效、数据驱动的热力图渲染器 (线性插值版)'''
    # 声明全局C变量
    global gradient_c_array, gradient_initialized

    # --- 首次调用时 将Python的梯度表转译为C数组 ---
    # 这个块只执行一次 且在GIL环境下执行
    if not gradient_initialized:
        TABLE_SIZE_CONST = 5
        for i in range(TABLE_SIZE_CONST):
            w_py, (r_py, g_py, b_py) = GRADIENT_TABLE[i]
            gradient_c_array[i][0] = w_py
            gradient_c_array[i][1] = r_py
            gradient_c_array[i][2] = g_py
            gradient_c_array[i][3] = b_py
        gradient_initialized = True

    # --- C级别的局部变量声明 (使用Python注解语法) ---
    N: cython.int = color_view.shape[0]
    TABLE_SIZE: cython.int = 5
    i: cython.int
    j: cython.int
    w: cython.float
    r: cython.float
    g: cython.float
    b: cython.float
    t: cython.float
    start_w: cython.float
    end_w: cython.float
    start_r: cython.float
    start_g: cython.float
    start_b: cython.float
    end_r: cython.float
    end_g: cython.float
    end_b: cython.float

    with cython.nogil:
        for i in range(N):
            w = weights_1d[i]

            # --- 优化分支 1: 处理权重溢出或无效的情况 ---
            if w >= 0.9999999:
                r, g, b = 1.0, 1.0, 1.0
            elif w <= 0.0000001:
                r, g, b = 0.0, 0.0, 0.0
            
            # --- 优化分支 2: 在C数组中查找并进行线性插值 ---
            else:
                for j in range(TABLE_SIZE - 1):
                    start_w = gradient_c_array[j][0]
                    start_r = gradient_c_array[j][1]
                    start_g = gradient_c_array[j][2]
                    start_b = gradient_c_array[j][3]

                    end_w = gradient_c_array[j + 1][0]
                    end_r = gradient_c_array[j + 1][1]
                    end_g = gradient_c_array[j + 1][2]
                    end_b = gradient_c_array[j + 1][3]
                    
                    if start_w <= w < end_w:
                        t = (w - start_w) / (end_w - start_w)
                        r = start_r + t * (end_r - start_r)
                        g = start_g + t * (end_g - start_g)
                        b = start_b + t * (end_b - start_b)
                        break

            color_view[i, 0] = r
            color_view[i, 1] = g
            color_view[i, 2] = b
            color_view[i, 3] = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
def render_gradient(weights_1d: cython.float[:], color_view: cython.float[:, :], color_a: tuple, color_b: tuple):
    '''通用双色插值器 (纯 Python 语法)'''
    N: cython.int = color_view.shape[0]
    i: cython.int
    w: cython.float

    bg_r: cython.float = color_a[0]
    bg_g: cython.float = color_a[1]
    bg_b: cython.float = color_a[2]
    bg_a: cython.float = color_a[3]

    fg_r: cython.float = color_b[0]
    fg_g: cython.float = color_b[1]
    fg_b: cython.float = color_b[2]
    fg_a: cython.float = color_b[3]

    with cython.nogil:
        for i in range(N):
            w = weights_1d[i]
            color_view[i, 0] = bg_r + w * (fg_r - bg_r)
            color_view[i, 1] = bg_g + w * (fg_g - bg_g)
            color_view[i, 2] = bg_b + w * (fg_b - bg_b)
            color_view[i, 3] = bg_a + w * (fg_a - bg_a)


@cython.boundscheck(False)
@cython.wraparound(False)
def render_fill(color_view: cython.float[:, :], color: tuple):
    '''纯色填充器 (纯 Python 语法)'''
    N: cython.int = color_view.shape[0]
    i: cython.int

    r: cython.float = color[0]
    g: cython.float = color[1]
    b: cython.float = color[2]
    a: cython.float = color[3]

    with cython.nogil:
        for i in range(N):
            color_view[i, 0] = r
            color_view[i, 1] = g
            color_view[i, 2] = b
            color_view[i, 3] = a


@cython.boundscheck(False)
@cython.wraparound(False)
def render_brush_gradient(
    color_view: cython.float[:, :],
    hit_indices: cython.int[:],
    hit_weights: cython.float[:],
    hit_count: cython.int,
    color_a: tuple,
    color_b: tuple,
):
    '''散点渐变器 专门用于通过顶点ID精准映射笔刷衰减颜色'''
    i: cython.int
    v_idx: cython.int
    w: cython.float

    bg_r: cython.float = color_b[0]
    bg_g: cython.float = color_b[1]
    bg_b: cython.float = color_b[2]
    bg_a: cython.float = color_b[3]
    fg_r: cython.float = color_a[0]
    fg_g: cython.float = color_a[1]
    fg_b: cython.float = color_a[2]
    fg_a: cython.float = color_a[3]

    with cython.nogil:
        # 核心 只循环 hit_count 次 绝不多算一点
        for i in range(hit_count):
            v_idx = hit_indices[i]
            w = hit_weights[i]

            color_view[v_idx, 0] = bg_r + w * (fg_r - bg_r)
            color_view[v_idx, 1] = bg_g + w * (fg_g - bg_g)
            color_view[v_idx, 2] = bg_b + w * (fg_b - bg_b)
            color_view[v_idx, 3] = bg_a + w * (fg_a - bg_a)


# =====================================================================
# 3. 渲染管线辅助函数 (直接向 Maya VP2 缓冲区写入数据)
# =====================================================================
@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nogil
def _offset_indices_direct(
    src_ptr: cython.p_uint,
    dst_ptr: cython.p_uint,
    count: cython.int,
    offset: cython.uint,
) -> cython.void:
    i: cython.int
    for i in range(count):
        dst_ptr[i] = src_ptr[i] + offset


def offset_indices_direct(
    src_addr: cython.Py_ssize_t,
    dst_addr: cython.Py_ssize_t,
    count: cython.int,
    offset: cython.int,
):
    '''Python 包装器 接收整型地址 强转为 unsigned int 指针'''
    src_ptr = cython.cast(cython.p_uint, src_addr)
    dst_ptr = cython.cast(cython.p_uint, dst_addr)

    _offset_indices_direct(src_ptr, dst_ptr, count, cython.cast(cython.uint, offset))
