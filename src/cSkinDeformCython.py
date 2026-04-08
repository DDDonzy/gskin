# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# distutils: language = c++

import cython

# fmt:off
@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.exceptval(check=False)
@cython.nogil
def _cal_deform_matrices(
    out_rotate_matrix     : cython.p_float,
    out_translate_vector  : cython.p_float,
    num_influences        : cython.int,
    influences_matrix     : cython.p_double,
    bind_pre_matrix       : cython.p_double,
    geo_matrix            : cython.p_double,
    geo_matrix_i          : cython.p_double,
    geo_matrix_is_identity: cython.bint,
) -> cython.void:
    """蒙皮变形矩阵预计算内核 (C-level).

    将 4x4 蒙皮矩阵分解为 3x3 旋转缩放部分和 1x3 位移向量，以减少顶点运算量。
    计算公式: Final = Geo * BindPre * BoneWorld * GeoInv.

    Args:
        out_rotate_matrix (cython.p_float): 3x3 旋转矩阵输出指针 [num_influences * 9].
        out_translate_vector (cython.p_float): 1x3 位移向量输出指针 [num_influences * 3].
        num_influences (cython.int): 骨骼总数.
        influences_matrix (cython.p_double): 骨骼当前世界矩阵指针 (Double4x4) [num_influences * 16].
        bind_pre_matrix (cython.p_double): 骨骼绑定姿态逆矩阵指针 (Double4x4) [num_influences * 16].
        geo_matrix (cython.p_double): 模型变换偏移矩阵指针 (Double4x4) [16].
        geo_matrix_i (cython.p_double): 模型变换偏移逆矩阵指针 (Double4x4) [16].
        geo_matrix_is_identity (cython.bint): 模型变换是否为单位阵 (True 则跳过 Geo 相关计算以提升性能).
    """

    b: cython.int
    i: cython.int
    j: cython.int
    k: cython.int

    T1   : cython.double[4][4] = cython.declare(cython.double[4][4])
    T2   : cython.double[4][4] = cython.declare(cython.double[4][4])
    Final: cython.double[4][4] = cython.declare(cython.double[4][4])

    for b in range(num_influences):
        # 模型 Transform 没有移动
        if geo_matrix_is_identity:
            # 既然 Geo 和 GeoInv 都是空气 直接 Final = BindPre * BoneWorld
            for i in range(4):
                for j in range(4):
                    Final[i][j] = 0.0
                    for k in range(4):
                        Final[i][j] += bind_pre_matrix[b * 16 + i * 4 + k] * influences_matrix[b * 16 + k * 4 + j]

        # 模型 Transform 被移动了
        else:
            # 1. T1 = Geo * Bind
            for i in range(4):
                for j in range(4):
                    T1[i][j] = 0.0
                    for k in range(4):
                        T1[i][j] += geo_matrix[i * 4 + k] * bind_pre_matrix[b * 16 + k * 4 + j]

            # 2. T2 = T1 * Bone
            for i in range(4):
                for j in range(4):
                    T2[i][j] = 0.0
                    for k in range(4):
                        T2[i][j] += T1[i][k] * influences_matrix[b * 16 + k * 4 + j]

            # 3. Final = T2 * GeoInv
            for i in range(4):
                for j in range(4):
                    Final[i][j] = 0.0
                    for k in range(4):
                        Final[i][j] += T2[i][k] * geo_matrix_i[k * 4 + j]

        # 提取 3x3 旋转和 1x3 位移
        for i in range(3):
            for j in range(3):
                out_rotate_matrix[b * 9 + i * 3 + j] = cython.cast(cython.float, Final[i][j])

        out_translate_vector[b * 3 + 0] = cython.cast(cython.float, Final[3][0])
        out_translate_vector[b * 3 + 1] = cython.cast(cython.float, Final[3][1])
        out_translate_vector[b * 3 + 2] = cython.cast(cython.float, Final[3][2])


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.exceptval(check=False)  # 等同于 noexcept
@cython.nogil
def _run_skinning_core(
    out_position     : cython.p_float,
    orig_position    : cython.p_float,
    weights          : cython.p_float,
    rotate_matrix    : cython.p_float,
    translate_vector : cython.p_float,
    num_vertices     : cython.int,
    num_influences   : cython.int,
    envelope         : cython.float,
) -> cython.void:
    """全量线性蒙皮 (LBS) 解算内核 (C-level).

    Args:
        out_position (cython.p_float): 解算后的顶点坐标输出指针 [num_vertices * 3].
        orig_position (cython.p_float): 原始顶点坐标输入指针 [num_vertices * 3].
        weights (cython.p_float): 扁平化的蒙皮权重数组指针 [num_vertices * num_influences].
        rotate_matrix (cython.p_float): 预计算好的 3x3 旋转矩阵指针 [num_influences * 9].
        translate_vector (cython.p_float): 预计算好的 1x3 位移向量指针 [num_influences * 3].
        num_vertices (cython.int): 顶点总数.
        num_influences (cython.int): 骨骼总数.
        envelope (cython.float): 蒙皮强度包络 (0.0-1.0).
    """
    v    : cython.int    # vertex index
    b    : cython.int    # bone index
    w    : cython.float  # weight
    x    : cython.float  # original x
    y    : cython.float  # original y
    z    : cython.float  # original z
    out_x: cython.float  # output x
    out_y: cython.float  # output y
    out_z: cython.float  # output z

    for v in range(num_vertices):  # or prange
        x     = orig_position[v * 3 + 0]
        y     = orig_position[v * 3 + 1]
        z     = orig_position[v * 3 + 2]
        out_x = 0.0
        out_y = 0.0
        out_z = 0.0

        for b in range(num_influences):
            w = weights[v * num_influences + b]
            if w < 0.000001:
                continue

            out_x += w * (x * rotate_matrix[b * 9 + 0] + y * rotate_matrix[b * 9 + 3] + z * rotate_matrix[b * 9 + 6] + translate_vector[b * 3 + 0])
            out_y += w * (x * rotate_matrix[b * 9 + 1] + y * rotate_matrix[b * 9 + 4] + z * rotate_matrix[b * 9 + 7] + translate_vector[b * 3 + 1])
            out_z += w * (x * rotate_matrix[b * 9 + 2] + y * rotate_matrix[b * 9 + 5] + z * rotate_matrix[b * 9 + 8] + translate_vector[b * 3 + 2])

        if envelope != 1.0:
            out_x = (out_x - x) * envelope + x
            out_y = (out_y - y) * envelope + y
            out_z = (out_z - z) * envelope + z

        out_position[v * 3 + 0] = out_x
        out_position[v * 3 + 1] = out_y
        out_position[v * 3 + 2] = out_z


@cython.cfunc
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.exceptval(check=False)  # 等同于 noexcept
@cython.nogil
def _run_partial_skinning_core(
    out_position     : cython.p_float,
    orig_position    : cython.p_float,
    weights          : cython.p_float,
    rotate_matrix    : cython.p_float,
    translate_vector : cython.p_float,
    vertex_indices   : cython.p_int,
    num_vertices     : cython.int,
    num_influences   : cython.int,
    envelope         : cython.float,
) -> cython.void:
    """局部稀疏蒙皮解算内核 (C-level).

    只计算指定的顶点索引，通常用于笔刷涂抹时的实时刷新。

    Args:
        out_position (cython.p_float): 解算后的顶点坐标输出指针.
        orig_position (cython.p_float): 原始顶点坐标输入指针.
        weights (cython.p_float): 蒙皮权重数组指针.
        rotate_matrix (cython.p_float): 3x3 旋转矩阵指针.
        translate_vector (cython.p_float): 1x3 位移向量指针.
        vertex_indices (cython.p_int): 需要计算的顶点索引数组指针.
        num_vertices (cython.int): 待计算的索引数量 (索引数组的长度).
        num_influences (cython.int): 骨骼总数.
        envelope (cython.float): 蒙皮强度包络 (0.0-1.0).
    """

    i    : cython.int    # 循环索引
    v    : cython.int    # 实际顶点 ID
    b    : cython.int    # bone index
    w    : cython.float  # weight
    x    : cython.float  # original x
    y    : cython.float  # original y
    z    : cython.float  # original z
    out_x: cython.float  # output x
    out_y: cython.float  # output y
    out_z: cython.float  # output z

    # 不再遍历所有顶点, 只遍历有效索引
    for i in range(num_vertices):
        v = vertex_indices[i]

        x = orig_position[v * 3 + 0]
        y = orig_position[v * 3 + 1]
        z = orig_position[v * 3 + 2]
        out_x = 0.0
        out_y = 0.0
        out_z = 0.0

        for b in range(num_influences):
            w = weights[v * num_influences + b]
            if w < 0.000001:
                continue

            out_x += w * (x * rotate_matrix[b * 9 + 0] + y * rotate_matrix[b * 9 + 3] + z * rotate_matrix[b * 9 + 6] + translate_vector[b * 3 + 0])
            out_y += w * (x * rotate_matrix[b * 9 + 1] + y * rotate_matrix[b * 9 + 4] + z * rotate_matrix[b * 9 + 7] + translate_vector[b * 3 + 1])
            out_z += w * (x * rotate_matrix[b * 9 + 2] + y * rotate_matrix[b * 9 + 5] + z * rotate_matrix[b * 9 + 8] + translate_vector[b * 3 + 2])

        if envelope != 1.0:
            out_x = (out_x - x) * envelope + x
            out_y = (out_y - y) * envelope + y
            out_z = (out_z - z) * envelope + z

        # 🌟 直接将算好的新坐标就地覆写进输出数组 (共享内存)
        out_position[v * 3 + 0] = out_x
        out_position[v * 3 + 1] = out_y
        out_position[v * 3 + 2] = out_z


# =====================================================================
# 2. Python 接口包装器 (智能提纯 接收视图 -> 提取指针 -> 喂给内核)
# =====================================================================
def cal_deform_matrices(
    out_rotate_matrix_view   : cython.float[:, :],
    out_translate_vector_view: cython.float[:, :],
    influences_matrix_view   : cython.double[:, :],
    bind_pre_matrix_view     : cython.double[:, :],
    geo_matrix               : cython.double[:],
    geo_matrix_i             : cython.double[:],
    geo_matrix_is_identity   : cython.bint,
):
    """供 Python 调用的矩阵混合入口.

    将 Python 层的 MemoryView 提取为 C 指针并调用内核。

    Args:
        out_rotate_matrix_view (cython.float[:, :]): 2D 输出视图 [num_bones, 9].
        out_translate_vector_view (cython.float[:, :]): 2D 输出视图 [num_bones, 3].
        influences_matrix_view (cython.double[:, :]): 2D 输入视图 [num_bones, 16].
        bind_pre_matrix_view (cython.double[:, :]): 2D 输入视图 [num_bones, 16].
        geo_matrix (cython.double[:]): 1D 模型矩阵视图 [16].
        geo_matrix_i (cython.double[:]): 1D 模型逆矩阵视图 [16].
        geo_matrix_is_identity (cython.bint): 是否为单位阵.
    """

    # 自动从视图获取长度 无需手动传参
    num_influences: cython.int = bind_pre_matrix_view.shape[0]

    # 获取底层原生 C 指针
    out_rotate_matrix_ptr    = cython.cast(cython.p_float, cython.address(out_rotate_matrix_view[0, 0]))
    out_translate_vector_ptr = cython.cast(cython.p_float, cython.address(out_translate_vector_view[0, 0]))
    influences_matrix_ptr    = cython.cast(cython.p_double, cython.address(influences_matrix_view[0, 0]))
    bind_pre_matrix_ptr      = cython.cast(cython.p_double, cython.address(bind_pre_matrix_view[0, 0]))
    geo_matrix_ptr           = cython.cast(cython.p_double, cython.address(geo_matrix[0]))
    geo_matrix_i_ptr         = cython.cast(cython.p_double, cython.address(geo_matrix_i[0]))

    _cal_deform_matrices(
        out_rotate_matrix_ptr,
        out_translate_vector_ptr,
        num_influences,
        influences_matrix_ptr,
        bind_pre_matrix_ptr,
        geo_matrix_ptr,
        geo_matrix_i_ptr,
        geo_matrix_is_identity,
    )


def run_skinning_core(
    out_position_view     : cython.float[:],
    original_position_view: cython.float[:],
    weights_view          : cython.float[:],
    rotate_matrix_view    : cython.float[:, :],
    translate_vector_view : cython.float[:, :],
    envelope              : cython.float,
):
    """供 Python 调用的全量蒙皮解算入口.

    Args:
        out_position_view (cython.float[:]): 输出顶点视图 (扁平化 1D).
        original_position_view (cython.float[:]): 原始顶点视图 (扁平化 1D).
        weights_view (cython.float[:]): 权重视图 (扁平化 1D).
        rotate_matrix_view (cython.float[:, :]): 2D 旋转矩阵视图 [B, 9].
        translate_vector_view (cython.float[:, :]): 2D 位移向量视图 [B, 3].
        envelope (cython.float): 蒙皮强度 (0.0-1.0).
    """

    # 自动从视图获取顶点和骨骼数
    num_vertices  : cython.int = original_position_view.shape[0] // 3
    num_influences: cython.int = translate_vector_view.shape[0]

    # 提取底层原生 C 指针
    out_position      = cython.cast(cython.p_float, cython.address(out_position_view[0]))
    original_position = cython.cast(cython.p_float, cython.address(original_position_view[0]))

    weights           = cython.cast(cython.p_float, cython.address(weights_view[0]))
    rotate_matrix     = cython.cast(cython.p_float, cython.address(rotate_matrix_view[0, 0]))
    translate_vector  = cython.cast(cython.p_float, cython.address(translate_vector_view[0, 0]))

    _run_skinning_core(
        out_position,
        original_position,
        weights,
        rotate_matrix,
        translate_vector,
        num_vertices,
        num_influences,
        envelope,
    )


def run_partial_skinning_core(
    out_position_view    : cython.float[:],
    orig_position_view   : cython.float[:],
    vertex_indices_view  : cython.int[:],
    weights_view         : cython.float[:],
    rotate_matrix_view   : cython.float[:, :],
    translate_vector_view: cython.float[:, :],
    envelope             : cython.float,
):
    """供 Python 调用的局部蒙皮解算入口.

    Args:
        out_position_view (cython.float[:]): 输出顶点视图.
        orig_position_view (cython.float[:]): 原始顶点视图.
        vertex_indices_view (cython.int[:]): 待计算的顶点索引视图.
        weights_view (cython.float[:]): 权重视图.
        rotate_matrix_view (cython.float[:, :]): 2D 旋转矩阵视图 [B, 9].
        translate_vector_view (cython.float[:, :]): 2D 位移向量视图 [B, 3].
        envelope (cython.float): 蒙皮强度 (0.0-1.0).
    """

    # 自动获取需要计算的顶点数量
    num_vertices: cython.int = vertex_indices_view.shape[0]
    if num_vertices == 0:
        return
    num_influences: cython.int = translate_vector_view.shape[0]
    if num_influences == 0:
        return

    # 提取底层原生 C 指针
    out_position_ptr     = cython.cast(cython.p_float, cython.address(out_position_view[0]))

    orig_position_ptr    = cython.cast(cython.p_float, cython.address(orig_position_view[0]))
    vertex_indices_ptr   = cython.cast(cython.p_int, cython.address(vertex_indices_view[0]))
    weights_ptr          = cython.cast(cython.p_float, cython.address(weights_view[0]))
    rotate_matrix_ptr    = cython.cast(cython.p_float, cython.address(rotate_matrix_view[0, 0]))
    transform_vector_ptr = cython.cast(cython.p_float, cython.address(translate_vector_view[0, 0]))

    # 调用无 GIL 的纯 C 内核
    _run_partial_skinning_core(
        out_position_ptr,
        orig_position_ptr,
        weights_ptr,
        rotate_matrix_ptr,
        transform_vector_ptr,
        vertex_indices_ptr,
        num_vertices,   
        num_influences,
        envelope,
    )
