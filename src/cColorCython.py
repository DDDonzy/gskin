# ==============================================================================
# 🎨 cColor.py - 纯 Python 语法版 (支持 Cython 极限编译)
# ==============================================================================
import cython

# --- 核心优化：定义静态颜色梯度表 ---
# 格式： (权重上限, (R, G, B))
# 表中每一项定义了一个颜色“停靠点”。权重值会在此表定义的颜色之间进行平滑的线性插值。
# 这个表可以轻松地进行扩展和修改，以实现不同的热力图效果。
GRADIENT_TABLE: typing.List[typing.Tuple[cython.float, typing.Tuple[cython.float, cython.float, cython.float]]] = [
    (0.0,   (0.0, 0.0, 0.0)),   # 权重 0.0 -> 纯黑
    (0.25,  (0.0, 0.0, 1.0)),   # 权重 0.25 -> 纯蓝
    (0.5,   (0.0, 1.0, 1.0)),   # 权重 0.5 -> 青色
    (0.75,  (1.0, 1.0, 0.0)),   # 权重 0.75 -> 黄色
    (1.0,   (1.0, 0.0, 0.0)),   # 权重 1.0 -> 纯红
]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def render_heatmap(
    weights_1d: cython.float[:],
    color_view: cython.float[:, :],
):
    """优雅、高效、数据驱动的热力图渲染器 (线性插值版)"""
    N: cython.int = color_view.shape[0]
    TABLE_SIZE: cython.int = len(GRADIENT_TABLE)
    i: cython.int
    j: cython.int
    w: cython.float
    r: cython.float
    g: cython.float
    b: cython.float
    t: cython.float
    
    # --- 将 Python 端的 GRADIENT_TABLE 解包到 Cython 静态数组中，以实现最高性能 ---
    # 这段代码只在函数首次调用时执行一次，后续调用会直接复用已转换的静态数据
    cdef float start_w, end_w
    cdef float start_r, start_g, start_b
    cdef float end_r, end_g, end_b

    with cython.nogil:
        for i in range(N):
            w = weights_1d[i]

            # --- 优化分支 1: 处理权重溢出或无效的情况 ---
            if w >= 1.0:
                r, g, b = 1.0, 1.0, 1.0  # 权重 >= 1.0: 纯白 (高亮溢出)
            elif w <= 0.0:
                r, g, b = 0.0, 0.0, 0.0  # 权重 <= 0.0: 纯黑 (无影响)
            
            # --- 优化分支 2: 在梯度表中查找并进行线性插值 ---
            else:
                # 遍历梯度表，找到权重 w 所属的区间
                for j in range(TABLE_SIZE - 1):
                    start_w, (start_r, start_g, start_b) = GRADIENT_TABLE[j]
                    end_w, (end_r, end_g, end_b) = GRADIENT_TABLE[j + 1]

                    if start_w <= w < end_w:
                        # 计算 w 在当前区间的插值系数 t
                        t = (w - start_w) / (end_w - start_w)
                        
                        # 线性插值 (Lerp)
                        r = start_r + t * (end_r - start_r)
                        g = start_g + t * (end_g - start_g)
                        b = start_b + t * (end_b - start_b)
                        break  # 找到区间后立即跳出循环

            color_view[i, 0] = r
            color_view[i, 1] = g
            color_view[i, 2] = b
            color_view[i, 3] = 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
def render_gradient(weights_1d: cython.float[:], color_view: cython.float[:, :], color_a: tuple, color_b: tuple):
    """通用双色插值器 (纯 Python 语法)"""
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
    """纯色填充器 (纯 Python 语法)"""
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
    """散点渐变器：专门用于通过顶点ID精准映射笔刷衰减颜色"""
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
        # 核心：只循环 hit_count 次，绝不多算一点！
        for i in range(hit_count):
            v_idx = hit_indices[i]  # 拿到真实的顶点 ID
            w = hit_weights[i]  # 拿到对应的衰减权重

            # 精准投放到对应的显存位置
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
    # 👆 💥 加上 -> cython.void:
    i: cython.int
    for i in range(count):
        dst_ptr[i] = src_ptr[i] + offset


def offset_indices_direct(
    src_addr: cython.Py_ssize_t,
    dst_addr: cython.Py_ssize_t,
    count: cython.int,
    offset: cython.int,
):
    """Python 包装器：接收整型地址，强转为 unsigned int 指针"""
    src_ptr = cython.cast(cython.p_uint, src_addr)
    dst_ptr = cython.cast(cython.p_uint, dst_addr)

    _offset_indices_direct(src_ptr, dst_ptr, count, cython.cast(cython.uint, offset))
