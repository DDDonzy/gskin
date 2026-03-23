import cython
import ctypes
from cython import nogil


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_unique_edge_indices(tri_indices1D: cython.int[::1]):
    """
    从三角面顶点索引中,计算并提取全局去重的无向边索引。

    Args:
        tri_indices1D (cython.int[::1]): 展平的一维三角面索引数组。

    Returns:
        ctypes.Array: 精确大小的 ctypes 原生连续内存数组。
            格式为[e0_v0, e0_v1, e1_v0, e1_v1, ...]
            包含的边数 (Edge Count) = len(返回值) // 2。
    """
    num_tris: cython.int = tri_indices1D.shape[0] // 3
    i: cython.int
    v0: cython.int
    v1: cython.int
    v2: cython.int

    unique_edges = set()

    for i in range(num_tris):
        v0 = tri_indices1D[i * 3 + 0]
        v1 = tri_indices1D[i * 3 + 1]
        v2 = tri_indices1D[i * 3 + 2]

        unique_edges.add((v0, v1) if v0 < v1 else (v1, v0))
        unique_edges.add((v1, v2) if v1 < v2 else (v2, v1))
        unique_edges.add((v2, v0) if v2 < v0 else (v0, v2))

    unique_count: cython.int = len(unique_edges)
    exact_length: cython.int = unique_count * 2

    out_ctypes = (ctypes.c_int * exact_length)()
    out_view: cython.int[::1] = out_ctypes

    idx: cython.int = 0
    for edge in unique_edges:
        out_view[idx] = edge[0]
        out_view[idx + 1] = edge[1]
        idx += 2

    return out_ctypes


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def build_v2v_adjacency(num_verts: cython.int, edge_indices1D: cython.int[::1]) -> tuple:
    """
    构建顶点到顶点 (Vertex-to-Vertex) 的 CSR 格式邻接表。

    【核心架构】
    1. 直接使用 Python 原生的 `ctypes` 预分配内存,天然支持垃圾回收 (GC),彻底杜绝内存泄漏。
    2. ctypes 内存默认已被 C 语言初始化清零,省去了手动清零的计算开销。
    3. 运算结束后直接返回 ctypes 对象组,实现 Python 与 Cython 之间的 0 拷贝数据交接。

    Args:
        num_verts (cython.int): 模型的顶点总数。
        edge_indices1D (cython.int[::1]): `compute_unique_edge_indices` 生成的去重边数组。

    Returns:
        tuple[ctypes.Array, ctypes.Array]: 返回 (offsets_ctypes, indices_ctypes) 原生内存组。
    """
    num_edges: cython.int = edge_indices1D.shape[0] // 2
    i: cython.int
    v1: cython.int
    v2: cython.int
    idx: cython.int

    # 1. 动态分配归 Python 管理的 ctypes 数组 (已自动清零)
    offsets_ctypes = (ctypes.c_int * (num_verts + 1))()
    indices_ctypes = (ctypes.c_int * (num_edges * 2))()
    temp_cursor_ctypes = (ctypes.c_int * num_verts)()

    # 2. 映射为 C 级别底层视图,方便无缝计算
    offsets_view: cython.int[::1] = offsets_ctypes
    indices_view: cython.int[::1] = indices_ctypes
    temp_cursor: cython.int[::1] = temp_cursor_ctypes

    # 3. 纯 C 飙车模式
    with nogil:
        # 统计度数
        for i in range(num_edges):
            v1 = edge_indices1D[i * 2 + 0]
            v2 = edge_indices1D[i * 2 + 1]
            offsets_view[v1 + 1] += 1
            offsets_view[v2 + 1] += 1

        # 计算前缀和并初始化游标
        for i in range(num_verts):
            offsets_view[i + 1] += offsets_view[i]
            temp_cursor[i] = offsets_view[i]

        # 填充 indices
        for i in range(num_edges):
            v1 = edge_indices1D[i * 2 + 0]
            v2 = edge_indices1D[i * 2 + 1]

            idx = temp_cursor[v1]
            indices_view[idx] = v2
            temp_cursor[v1] += 1

            idx = temp_cursor[v2]
            indices_view[idx] = v1
            temp_cursor[v2] += 1

    # 4. 瞬间交接,0 拷贝返回
    return offsets_ctypes, indices_ctypes


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def build_v2f_adjacency(num_verts: cython.int, tri_indices1D: cython.int[::1]) -> tuple:
    """
    构建顶点到面 (Vertex-to-Face) 的 CSR 格式邻接表。

    同 `build_v2v_adjacency`,采用 0 拷贝与原生 ctypes 返回设计。
    直接将 Maya 的三角面数组转换为高速可查的顶点到面的邻接数据。

    Args:
        num_verts (cython.int): 模型的顶点总数。
        tri_indices1D (cython.int[::1]): 展平的一维三角面顶点数组。

    Returns:
        tuple[ctypes.Array, ctypes.Array]: 返回 (offsets_ctypes, indices_ctypes) 原生内存组。
    """
    num_tris: cython.int = tri_indices1D.shape[0] // 3
    i: cython.int
    v0: cython.int
    v1: cython.int
    v2: cython.int
    idx: cython.int

    # 分配 ctypes 数组
    offsets_ctypes = (ctypes.c_int * (num_verts + 1))()
    indices_ctypes = (ctypes.c_int * (num_tris * 3))()
    temp_cursor_ctypes = (ctypes.c_int * num_verts)()

    # 映射视图
    offsets_view: cython.int[::1] = offsets_ctypes
    indices_view: cython.int[::1] = indices_ctypes
    temp_cursor: cython.int[::1] = temp_cursor_ctypes

    with nogil:
        # 统计度数
        for i in range(num_tris):
            v0 = tri_indices1D[i * 3 + 0]
            v1 = tri_indices1D[i * 3 + 1]
            v2 = tri_indices1D[i * 3 + 2]
            offsets_view[v0 + 1] += 1
            offsets_view[v1 + 1] += 1
            offsets_view[v2 + 1] += 1

        # 计算前缀和
        for i in range(num_verts):
            offsets_view[i + 1] += offsets_view[i]
            temp_cursor[i] = offsets_view[i]

        # 填充 indices
        for i in range(num_tris):
            v0 = tri_indices1D[i * 3 + 0]
            v1 = tri_indices1D[i * 3 + 1]
            v2 = tri_indices1D[i * 3 + 2]

            idx = temp_cursor[v0]
            indices_view[idx] = i
            temp_cursor[v0] += 1

            idx = temp_cursor[v1]
            indices_view[idx] = i
            temp_cursor[v1] += 1

            idx = temp_cursor[v2]
            indices_view[idx] = i
            temp_cursor[v2] += 1

    return offsets_ctypes, indices_ctypes
