import ctypes
import array
import cython
from cython.cimports.libc.math import sqrt, fabs  # type:ignore
from cython.cimports.libc.stdlib import calloc, free  # type:ignore
from cython.cimports.libc.string import memset  # type:ignore


@cython.cfunc
def _blend_math(
    blend_mode: cython.int,
    cur: cython.float,
    tgt: cython.float,
    fal: cython.float,
) -> cython.float:

    if blend_mode == 0:  # Add (加法)
        return cur + (tgt * fal)

    elif blend_mode == 1:  # Sub (减法)  # noqa: RET505
        return cur - (tgt * fal)

    elif blend_mode == 3:  # Multiply (乘法)
        return cur + ((cur * tgt - cur) * fal)

    else:  # Replace (2) 替换及默认兜底
        return cur + ((tgt - cur) * fal)


@cython.cfunc
def _clamp_float(
    val: cython.float,
    clamp_min: cython.float,
    clamp_max: cython.float,
) -> cython.float:
    if val < clamp_min:
        return clamp_min
    if val > clamp_max:
        return clamp_max
    return val


@cython.cclass
class CoreBrushEngine:
    """
    用于解析模型,raycast,计算笔刷衰减等
    """

    # --- 1. 外部传入的绝对只读物理数据 (恢复 2D 格式) ---
    vtx_positions2D: cython.float[:, ::1]
    tri_indices2D: cython.int[:, ::1]
    frozen_vtx_positions2D: cython.float[:, ::1]
    is_mesh_locked: cython.bint

    # --- 2. 引擎私有高速内存视图 (自带防 GC 保护) ---
    adj_offsets: cython.int[::1]
    adj_indices: cython.int[::1]
    v2f_offsets: cython.int[::1]
    v2f_indices: cython.int[::1]

    vertices_epochs: cython.int[::1]
    faces_epochs: cython.int[::1]

    active_hit_indices: cython.int[::1]
    active_hit_falloff: cython.float[::1]

    active_hit_count: cython.int
    brush_epoch: cython.int
    raycast_epoch: cython.int

    last_ray_hit_tri: cython.int

    def __init__(
        self,
        vtx_positions2D: cython.float[:, ::1],
        triangle_indices2D: cython.int[:, ::1],
        v2v_offset: cython.int[::1],
        v2v_indices: cython.int[::1],
        v2f_offset: cython.int[::1],
        v2f_indices: cython.int[::1],
    ):
        """初始化核心引擎 并绑定底层物理内存视图.

        Args:
            vtx_positions2D (cython.float[:, ::1]):
                网格顶点世界坐标矩阵.
                - 形状: [N, 3] 其中 N 为网格的顶点总数.
                - 说明: 每一行存储一个顶点的绝对空间坐标 (X, Y, Z).
                - 示例: `[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], ...]`
                  代表 0号点在(0,1,0) 1号点在(1,0,0).

            triangle_indices2D (cython.int[:, ::1]):
                网格三角面顶点索引矩阵 (Triangle Vertex Indices).
                - 形状: [M, 3] 其中 M 为网格的三角面总数.
                - 说明: 它是 3D 拓扑结构的核心 里面的数字并不是坐标 而是指向 `vtx_positions2D` 的“行号 (ID)”.
                  每一行的 3 个整数 代表构成这个三角面的 3 个顶点 ID 通常按逆时针 Winding Order 排列以确定法线朝向 .
                - 示例: 如果某一行是 `[5, 12, 8]` 则意味着这个三角面是由第 5 号、第 12 号、第 8 号顶点连接而成的一张皮.
                  在 M-T 射线算法中 我们会用这 3 个 ID 去 `vtx_positions2D` 里查出真正的空间坐标来进行相交测试.

        """

        self.vtx_positions2D = vtx_positions2D
        self.tri_indices2D = triangle_indices2D

        self.adj_offsets = v2v_offset
        self.adj_indices = v2v_indices

        self.v2f_offsets = v2f_offset
        self.v2f_indices = v2f_indices

        num_verts: cython.int = vtx_positions2D.shape[0]
        num_tris: cython.int = triangle_indices2D.shape[0]

        # 🌟 新增: 申请一块同样大小的连续 C 内存存放快照
        c_float_arr = (ctypes.c_float * (num_verts * 3))()
        self.frozen_vtx_positions2D = memoryview(c_float_arr).cast("B").cast("f", shape=(num_verts, 3))
        self.is_mesh_locked = False

        # 👑 内部自动分配所有 1D 数组 直接丢给 MemoryView 锚定
        self.vertices_epochs = array.array("i", [0]) * num_verts
        self.faces_epochs = array.array("i", [0]) * num_tris
        self.active_hit_indices = array.array("i", [0]) * num_verts
        self.active_hit_falloff = array.array("f", [0.0]) * num_verts

        self.brush_epoch = 1
        self.raycast_epoch = 1
        self.active_hit_count = 0

        self.last_ray_hit_tri = -1

    def update_vertex_positions(self, new_positions2D: cython.float[:, ::1]):
        self.vtx_positions2D = new_positions2D

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def lock_mesh(self) -> cython.void:
        """极速锁定当前网格形态 (底层编译为 memcpy)"""

        # 🌟 直接使用全切片赋值, 优雅且极致高效
        self.frozen_vtx_positions2D[:, :] = self.vtx_positions2D[:, :]

        self.is_mesh_locked = True

    @cython.ccall
    def unlock_mesh(self) -> cython.void:
        """解除锁定"""
        self.is_mesh_locked = False

    @property
    def raw_hit_indices(self):
        """将内部 C 级命中数组暴露给 Python 层"""
        return self.active_hit_indices.base

    @property
    def raw_hit_falloff(self):
        """将内部 C 级衰减数组暴露给 Python 层"""
        return self.active_hit_falloff.base

    # endregion

    # region ---------- Raycast
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.ccall
    def raycast(self, ray_pos: tuple, ray_dir: tuple, cull_backface: cython.bint = True) -> tuple:
        """单线程极速双轨制射线检测 (彻底剥离 Falloff 依赖, 采用三角面微缓存自愈)."""
        _points: cython.float[:, ::1] = self.frozen_vtx_positions2D if self.is_mesh_locked else self.vtx_positions2D
        _tri_indices: cython.int[:, ::1] = self.tri_indices2D

        orig_x: cython.float = ray_pos[0]
        orig_y: cython.float = ray_pos[1]
        orig_z: cython.float = ray_pos[2]
        dir_x: cython.float = ray_dir[0]
        dir_y: cython.float = ray_dir[1]
        dir_z: cython.float = ray_dir[2]

        num_tris: cython.int = _tri_indices.shape[0]

        v0_idx: cython.int
        v1_idx: cython.int
        v2_idx: cython.int
        edge1_x: cython.float
        edge1_y: cython.float
        edge1_z: cython.float
        edge2_x: cython.float
        edge2_y: cython.float
        edge2_z: cython.float
        h_x: cython.float
        h_y: cython.float
        h_z: cython.float
        s_x: cython.float
        s_y: cython.float
        s_z: cython.float
        q_x: cython.float
        q_y: cython.float
        q_z: cython.float
        a: cython.float
        f: cython.float
        u: cython.float
        v: cython.float
        t: cython.float

        # --- 全局交点结果容器 ---
        global_closest_t: cython.float = 999999.0
        global_hit_tri: cython.int = -1
        global_u: cython.float = 0.0
        global_v: cython.float = 0.0

        i: cython.int
        j: cython.int

        cache_hit: cython.bint = False

        self.raycast_epoch += 1

        # =====================================================================
        # 🚀 轨道一: V2F 局部空间拦截 (((基于last_hit_tri
        # =====================================================================
        # region --- plan 1
        temp_hover_vtx: cython.int[3] = cython.declare(cython.int[3])

        # 只要上一帧击中过任何面, 微缓存就生效.
        if self.last_ray_hit_tri != -1:
            # 提取上一帧击中面的 3 个顶点作为种子
            temp_hover_vtx[0] = _tri_indices[self.last_ray_hit_tri, 0]
            temp_hover_vtx[1] = _tri_indices[self.last_ray_hit_tri, 1]
            temp_hover_vtx[2] = _tri_indices[self.last_ray_hit_tri, 2]

            _curr_r_epoch: cython.int = self.raycast_epoch
            _f_epochs = self.faces_epochs

            v_idx: cython.int
            edge_start: cython.int
            edge_end: cython.int
            test_tri: cython.int

            with cython.nogil:
                # 永远只遍历这 3 个顶点
                for i in range(3):
                    v_idx = temp_hover_vtx[i]
                    edge_start = self.v2f_offsets[v_idx]
                    edge_end = self.v2f_offsets[v_idx + 1]

                    for j in range(edge_start, edge_end):
                        test_tri = self.v2f_indices[j]

                        # 世代掩码: 保证同一个面绝对只测一次
                        if _f_epochs[test_tri] == _curr_r_epoch:
                            continue
                        _f_epochs[test_tri] = _curr_r_epoch

                        v0_idx = _tri_indices[test_tri, 0]
                        v1_idx = _tri_indices[test_tri, 1]
                        v2_idx = _tri_indices[test_tri, 2]

                        edge1_x = _points[v1_idx, 0] - _points[v0_idx, 0]
                        edge1_y = _points[v1_idx, 1] - _points[v0_idx, 1]
                        edge1_z = _points[v1_idx, 2] - _points[v0_idx, 2]

                        edge2_x = _points[v2_idx, 0] - _points[v0_idx, 0]
                        edge2_y = _points[v2_idx, 1] - _points[v0_idx, 1]
                        edge2_z = _points[v2_idx, 2] - _points[v0_idx, 2]

                        h_x = dir_y * edge2_z - dir_z * edge2_y
                        h_y = dir_z * edge2_x - dir_x * edge2_z
                        h_z = dir_x * edge2_y - dir_y * edge2_x

                        a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z

                        if cull_backface:
                            # 🟢 单面极速模式 (开启背面剔除 + 延迟除法)
                            if a < 0.000001:
                                continue

                            s_x = orig_x - _points[v0_idx, 0]
                            s_y = orig_y - _points[v0_idx, 1]
                            s_z = orig_z - _points[v0_idx, 2]

                            u_unscaled = s_x * h_x + s_y * h_y + s_z * h_z
                            if u_unscaled < 0.0 or u_unscaled > a:
                                continue

                            q_x = s_y * edge1_z - s_z * edge1_y
                            q_y = s_z * edge1_x - s_x * edge1_z
                            q_z = s_x * edge1_y - s_y * edge1_x

                            v_unscaled = dir_x * q_x + dir_y * q_y + dir_z * q_z
                            if v_unscaled < 0.0 or u_unscaled + v_unscaled > a:
                                continue

                            t_unscaled = edge2_x * q_x + edge2_y * q_y + edge2_z * q_z

                            if t_unscaled < 0.000001 * a or t_unscaled > global_closest_t * a:
                                continue

                            f = 1.0 / a
                            global_closest_t = t_unscaled * f
                            global_hit_tri = test_tri
                            global_u = u_unscaled * f
                            global_v = v_unscaled * f
                            cache_hit = True

                        else:
                            # 🔴 双面安全模式 (无背面剔除)
                            if a > -0.000001 and a < 0.000001:
                                continue

                            f = 1.0 / a

                            s_x = orig_x - _points[v0_idx, 0]
                            s_y = orig_y - _points[v0_idx, 1]
                            s_z = orig_z - _points[v0_idx, 2]

                            u = f * (s_x * h_x + s_y * h_y + s_z * h_z)
                            if u < 0.0 or u > 1.0:
                                continue

                            q_x = s_y * edge1_z - s_z * edge1_y
                            q_y = s_z * edge1_x - s_x * edge1_z
                            q_z = s_x * edge1_y - s_y * edge1_x

                            v = f * (dir_x * q_x + dir_y * q_y + dir_z * q_z)
                            if v < 0.0 or u + v > 1.0:
                                continue

                            t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)

                            if t < 0.000001 or t > global_closest_t:
                                continue

                            global_closest_t = t
                            global_hit_tri = test_tri
                            global_u = u
                            global_v = v
                            cache_hit = True
        # endregion

        # =====================================================================
        # 🚀 轨道一一二V2F 局部空间拦截 (((基于falloff)
        # =====================================================================
        # region --- plan 2
        if not cache_hit and self.active_hit_count > 0:
            _curr_r_epoch: cython.int = self.raycast_epoch
            _f_epochs = self.faces_epochs

            v_idx: cython.int
            edge_start: cython.int
            edge_end: cython.int
            test_tri: cython.int

            # 🌟 必须加上 nogil, 否则异步调用时会卡死 Maya 主线程
            with cython.nogil:
                for i in range(self.active_hit_count):
                    v_idx = self.active_hit_indices[i]
                    edge_start = self.v2f_offsets[v_idx]
                    edge_end = self.v2f_offsets[v_idx + 1]

                    for j in range(edge_start, edge_end):
                        test_tri = self.v2f_indices[j]

                        # 世代掩码: 保证一帧内同一个面绝对只测一次
                        if _f_epochs[test_tri] == _curr_r_epoch:
                            continue
                        _f_epochs[test_tri] = _curr_r_epoch

                        v0_idx = _tri_indices[test_tri, 0]
                        v1_idx = _tri_indices[test_tri, 1]
                        v2_idx = _tri_indices[test_tri, 2]

                        edge1_x = _points[v1_idx, 0] - _points[v0_idx, 0]
                        edge1_y = _points[v1_idx, 1] - _points[v0_idx, 1]
                        edge1_z = _points[v1_idx, 2] - _points[v0_idx, 2]

                        edge2_x = _points[v2_idx, 0] - _points[v0_idx, 0]
                        edge2_y = _points[v2_idx, 1] - _points[v0_idx, 1]
                        edge2_z = _points[v2_idx, 2] - _points[v0_idx, 2]

                        h_x = dir_y * edge2_z - dir_z * edge2_y
                        h_y = dir_z * edge2_x - dir_x * edge2_z
                        h_z = dir_x * edge2_y - dir_y * edge2_x

                        a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z

                        if cull_backface:
                            # ---------------------------------------------
                            # 🟢 模式 A: 单面极速模式 (开启背面剔除 + 延迟除法)
                            # ---------------------------------------------
                            if a < 0.000001:
                                continue

                            s_x = orig_x - _points[v0_idx, 0]
                            s_y = orig_y - _points[v0_idx, 1]
                            s_z = orig_z - _points[v0_idx, 2]

                            u_unscaled = s_x * h_x + s_y * h_y + s_z * h_z
                            if u_unscaled < 0.0 or u_unscaled > a:
                                continue

                            q_x = s_y * edge1_z - s_z * edge1_y
                            q_y = s_z * edge1_x - s_x * edge1_z
                            q_z = s_x * edge1_y - s_y * edge1_x

                            v_unscaled = dir_x * q_x + dir_y * q_y + dir_z * q_z
                            if v_unscaled < 0.0 or u_unscaled + v_unscaled > a:
                                continue

                            t_unscaled = edge2_x * q_x + edge2_y * q_y + edge2_z * q_z

                            # 深度裁剪: 将 t 放大 a 倍后直接比较
                            if t_unscaled < 0.000001 * a or t_unscaled > global_closest_t * a:
                                continue

                            # 结算: 执行极其昂贵的除法
                            f = 1.0 / a
                            global_closest_t = t_unscaled * f
                            global_hit_tri = test_tri
                            global_u = u_unscaled * f
                            global_v = v_unscaled * f

                            cache_hit = True

                        else:
                            # ---------------------------------------------
                            # 🔴 模式 B: 双面安全模式 (无背面剔除)
                            # ---------------------------------------------
                            if a > -0.000001 and a < 0.000001:
                                continue

                            f = 1.0 / a

                            s_x = orig_x - _points[v0_idx, 0]
                            s_y = orig_y - _points[v0_idx, 1]
                            s_z = orig_z - _points[v0_idx, 2]

                            u = f * (s_x * h_x + s_y * h_y + s_z * h_z)
                            if u < 0.0 or u > 1.0:
                                continue

                            q_x = s_y * edge1_z - s_z * edge1_y
                            q_y = s_z * edge1_x - s_x * edge1_z
                            q_z = s_x * edge1_y - s_y * edge1_x

                            v = f * (dir_x * q_x + dir_y * q_y + dir_z * q_z)
                            if v < 0.0 or u + v > 1.0:
                                continue

                            t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)

                            if t < 0.000001 or t > global_closest_t:
                                continue

                            global_closest_t = t
                            global_hit_tri = test_tri
                            global_u = u
                            global_v = v

                            cache_hit = True
        # endregion

        # =====================================================================
        # 🛡️ 轨道二二三纯单线程极致暴力盲扫 (缓存未命中或首笔落下时触发)
        # =====================================================================
        # region --- plan 3
        if not cache_hit:
            with cython.nogil:
                for i in range(num_tris):
                    v0_idx = _tri_indices[i, 0]
                    v1_idx = _tri_indices[i, 1]
                    v2_idx = _tri_indices[i, 2]

                    edge1_x = _points[v1_idx, 0] - _points[v0_idx, 0]
                    edge1_y = _points[v1_idx, 1] - _points[v0_idx, 1]
                    edge1_z = _points[v1_idx, 2] - _points[v0_idx, 2]

                    edge2_x = _points[v2_idx, 0] - _points[v0_idx, 0]
                    edge2_y = _points[v2_idx, 1] - _points[v0_idx, 1]
                    edge2_z = _points[v2_idx, 2] - _points[v0_idx, 2]

                    h_x = dir_y * edge2_z - dir_z * edge2_y
                    h_y = dir_z * edge2_x - dir_x * edge2_z
                    h_z = dir_x * edge2_y - dir_y * edge2_x

                    a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z

                    if cull_backface:
                        if a < 0.000001:
                            continue

                        s_x = orig_x - _points[v0_idx, 0]
                        s_y = orig_y - _points[v0_idx, 1]
                        s_z = orig_z - _points[v0_idx, 2]

                        u_unscaled = s_x * h_x + s_y * h_y + s_z * h_z
                        if u_unscaled < 0.0 or u_unscaled > a:
                            continue

                        q_x = s_y * edge1_z - s_z * edge1_y
                        q_y = s_z * edge1_x - s_x * edge1_z
                        q_z = s_x * edge1_y - s_y * edge1_x

                        v_unscaled = dir_x * q_x + dir_y * q_y + dir_z * q_z
                        if v_unscaled < 0.0 or u_unscaled + v_unscaled > a:
                            continue

                        t_unscaled = edge2_x * q_x + edge2_y * q_y + edge2_z * q_z

                        if t_unscaled < 0.000001 * a or t_unscaled > global_closest_t * a:
                            continue

                        f = 1.0 / a
                        global_closest_t = t_unscaled * f
                        global_hit_tri = i
                        global_u = u_unscaled * f
                        global_v = v_unscaled * f

                    else:
                        if a > -0.000001 and a < 0.000001:
                            continue

                        f = 1.0 / a

                        s_x = orig_x - _points[v0_idx, 0]
                        s_y = orig_y - _points[v0_idx, 1]
                        s_z = orig_z - _points[v0_idx, 2]

                        u = f * (s_x * h_x + s_y * h_y + s_z * h_z)
                        if u < 0.0 or u > 1.0:
                            continue

                        q_x = s_y * edge1_z - s_z * edge1_y
                        q_y = s_z * edge1_x - s_x * edge1_z
                        q_z = s_x * edge1_y - s_y * edge1_x

                        v = f * (dir_x * q_x + dir_y * q_y + dir_z * q_z)
                        if v < 0.0 or u + v > 1.0:
                            continue

                        t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)

                        if t < 0.000001 or t > global_closest_t:
                            continue

                        global_closest_t = t
                        global_hit_tri = i
                        global_u = u
                        global_v = v
        # endregion

        # =====================================================================
        # 🏁 结算并记录本次命中, 反哺下一帧缓存
        # =====================================================================
        if global_hit_tri != -1:
            # 🌟 核心: 永远记住最后一次命中的面.供下一帧轨道一直接使用.
            self.last_ray_hit_tri = global_hit_tri

            v0_idx = _tri_indices[global_hit_tri, 0]
            v1_idx = _tri_indices[global_hit_tri, 1]
            v2_idx = _tri_indices[global_hit_tri, 2]

            edge1_x = _points[v1_idx, 0] - _points[v0_idx, 0]
            edge1_y = _points[v1_idx, 1] - _points[v0_idx, 1]
            edge1_z = _points[v1_idx, 2] - _points[v0_idx, 2]

            edge2_x = _points[v2_idx, 0] - _points[v0_idx, 0]
            edge2_y = _points[v2_idx, 1] - _points[v0_idx, 1]
            edge2_z = _points[v2_idx, 2] - _points[v0_idx, 2]

            raw_nx: cython.float = edge1_y * edge2_z - edge1_z * edge2_y
            raw_ny: cython.float = edge1_z * edge2_x - edge1_x * edge2_z
            raw_nz: cython.float = edge1_x * edge2_y - edge1_y * edge2_x

            norm_len: cython.float = sqrt(raw_nx * raw_nx + raw_ny * raw_ny + raw_nz * raw_nz)

            nx: cython.float = raw_nx / norm_len if norm_len > 0.000001 else 0.0
            ny: cython.float = raw_ny / norm_len if norm_len > 0.000001 else 0.0
            nz: cython.float = raw_nz / norm_len if norm_len > 0.000001 else 1.0

            hit_x: cython.float = orig_x + dir_x * global_closest_t
            hit_y: cython.float = orig_y + dir_y * global_closest_t
            hit_z: cython.float = orig_z + dir_z * global_closest_t

            return (
                True,
                (hit_x, hit_y, hit_z),
                (nx, ny, nz),
                global_hit_tri,
                global_closest_t,
                global_u,
                global_v,
            )

        # 🌟 如果射线飞到了宇宙边缘(未命中任何物体), 清空微缓存, 强制下一帧盲扫
        self.last_ray_hit_tri = -1
        return False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), -1, 0.0, 0.0, 0.0

    # endregion
    # endregion

    # region ---------- Falloff
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.ccall
    def calc_brush_falloff(
        self,
        hit_position: tuple,
        prev_hit_position: tuple,
        hit_tri_idx : cython.int,
        radius      : cython.float,
        falloff_mode: cython.int,
        use_surface : cython.bint,
    ) -> tuple:  # fmt:off
        """根据交点计算笔刷影响范围与衰减权重.

        Args:
            hit_position (tuple): 笔刷中心坐标 (x, y, z).
            hit_tri_idx (cython.int): 击中的三角面索引 (如果模式为表面拓扑则需要).
            radius (cython.float): 笔刷物理半径.
            falloff_mode (cython.int): 衰减模式 (0:线性, 1:平滑, 2:硬边, 3:穹顶, 4:尖刺).
            use_surface (cython.bint): 是否使用表面拓扑蔓延 (True: 沿网格蔓延, False: 空间球体).

        Returns:
            tuple: 包含以下元素的元组:
                - active_hit_count (cython.int): 实际命中的顶点总数.
                - active_hit_indices (cython.int[::1]): 命中顶点索引的内存视图切片.
                - active_hit_weights (cython.float[::1]): 命中顶点衰减权重的内存视图切片.
        """
        # fmt:off
        _vtx_pos: cython.float[:, ::1] = self.frozen_vtx_positions2D if self.is_mesh_locked else self.vtx_positions2D
        _tris_2d = self.tri_indices2D
        _adj_off = self.adj_offsets
        _adj_idx = self.adj_indices
        _epochs  = self.vertices_epochs
        _out_idx = self.active_hit_indices
        _out_w   = self.active_hit_falloff

        # 世代自增 替代昂贵的 memset 清空数组操作
        self.brush_epoch += 1
        _curr_epoch: cython.int = self.brush_epoch
        hit_count  : cython.int = 0                  # 当前收集的命中点数量
        num_verts  : cython.int = _vtx_pos.shape[0]  # 网格总顶点数

        hit_x: cython.float = hit_position[0]  # 笔刷中心 X
        hit_y: cython.float = hit_position[1]  # 笔刷中心 Y
        hit_z: cython.float = hit_position[2]  # 笔刷中心 Z

        # 🌟 获取线段起点
        prev_x: cython.float = prev_hit_position[0]
        prev_y: cython.float = prev_hit_position[1]
        prev_z: cython.float = prev_hit_position[2]

        # 🌟 计算线段向量 (A->B) 及其长度平方
        ab_x: cython.float = hit_x - prev_x
        ab_y: cython.float = hit_y - prev_y
        ab_z: cython.float = hit_z - prev_z
        ab_sq: cython.float = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z

        vx       : cython.float                    # 临时顶点坐标 X
        vy       : cython.float                    # 临时顶点坐标 Y
        vz       : cython.float                    # 临时顶点坐标 Z
        dx       : cython.float                    # 距离差 X
        dy       : cython.float                    # 距离差 Y
        dz       : cython.float                    # 距离差 Z

        # 🌟 新增: 用于计算投影的变量
        ap_x     : cython.float
        ap_y     : cython.float
        ap_z     : cython.float
        t_proj   : cython.float
        proj_x   : cython.float
        proj_y   : cython.float
        proj_z   : cython.float

        dist_sq  : cython.float                    # 距离的平方
        weight   : cython.float                    # 算出的最终衰减权重
        t2       : cython.float                    # 距离平方与半径平方的比值
        t        : cython.float                    # 标准化距离 (0~1)
        i        : cython.int                      # 循环索引
        j        : cython.int                      # 内层循环索引

        radius_sq: cython.float = radius * radius  # 笔刷半径平方
        # fmt:on

        # -------------------------------------------------------------
        # 模式 A 体积球体扫描 (Volume Mode)
        # -------------------------------------------------------------
        # region ----------- volume mode
        if not use_surface:
            # 🌟 动态计算包含整条线段的 AABB 包围盒, 避免任何方向的裁剪破绽
            min_x: cython.float = (hit_x if hit_x < prev_x else prev_x) - radius
            max_x: cython.float = (hit_x if hit_x > prev_x else prev_x) + radius
            min_y: cython.float = (hit_y if hit_y < prev_y else prev_y) - radius
            max_y: cython.float = (hit_y if hit_y > prev_y else prev_y) + radius
            min_z: cython.float = (hit_z if hit_z < prev_z else prev_z) - radius
            max_z: cython.float = (hit_z if hit_z > prev_z else prev_z) + radius

            with cython.nogil:
                for i in range(num_verts):
                    # 包围盒剔除
                    vx = _vtx_pos[i, 0]
                    if vx < min_x or vx > max_x:
                        continue
                    vy = _vtx_pos[i, 1]
                    if vy < min_y or vy > max_y:
                        continue
                    vz = _vtx_pos[i, 2]
                    if vz < min_z or vz > max_z:
                        continue

                    # 🌟 核心魔法: 计算点到线段的距离 (Capsule Math)
                    if ab_sq < 0.000001:
                        # 如果起点终点重合, 退化为点距离
                        dx = vx - hit_x
                        dy = vy - hit_y
                        dz = vz - hit_z
                    else:
                        # 计算顶点 P 在线段 AB 上的投影比例 t
                        ap_x = vx - prev_x
                        ap_y = vy - prev_y
                        ap_z = vz - prev_z

                        t_proj = (ap_x * ab_x + ap_y * ab_y + ap_z * ab_z) / ab_sq

                        # 将 t 钳制在线段 [0, 1] 内部 (这就是胶囊体两端是半球的原因)
                        if t_proj < 0.0:
                            t_proj = 0.0
                        elif t_proj > 1.0:
                            t_proj = 1.0

                        # 求出投影点坐标
                        proj_x = prev_x + t_proj * ab_x
                        proj_y = prev_y + t_proj * ab_y
                        proj_z = prev_z + t_proj * ab_z

                        # 最终距离 = 顶点到投影点的距离.
                        dx = vx - proj_x
                        dy = vy - proj_y
                        dz = vz - proj_z
                    dist_sq = dx * dx + dy * dy + dz * dz

                    # 衰减模式
                    if dist_sq <= radius_sq:
                        if falloff_mode == 2:  # Solid (硬边)
                            weight = 1.0
                        else:
                            t2 = dist_sq / radius_sq
                            if falloff_mode == 1:  # Airbrush (平滑/二次方衰减)
                                weight = 1.0 - t2
                                weight = weight * weight
                            elif falloff_mode == 0:  # Linear (线性衰减)
                                weight = 1.0 - sqrt(t2)
                            elif falloff_mode == 3:  # Dome (穹顶形)
                                weight = sqrt(1.0 - t2)
                            elif falloff_mode == 4:  # Spike (尖刺形)
                                t = sqrt(t2)
                                weight = 1.0 - t
                                weight = weight * weight * weight
                            else:
                                weight = 1.0

                        _out_idx[hit_count] = i
                        _out_w[hit_count] = weight
                        hit_count += 1

            self.active_hit_count = hit_count
            return (hit_count, self.active_hit_indices[:hit_count], self.active_hit_falloff[:hit_count])
        # endregion

        # -------------------------------------------------------------
        # 模式 B 表面拓扑扫描 (Surface Mode)
        # -------------------------------------------------------------
        # region ---------- surface mode
        if hit_tri_idx < 0:
            self.active_hit_count = 0
            return (0, self.active_hit_indices[:0], self.active_hit_falloff[:0])

        v0: cython.int = _tris_2d[hit_tri_idx, 0]  # 面片顶点 0
        v1: cython.int = _tris_2d[hit_tri_idx, 1]  # 面片顶点 1
        v2: cython.int = _tris_2d[hit_tri_idx, 2]  # 面片顶点 2

        closest_vtx: cython.int = v0  # 存储距离中心最近的种子顶点
        min_dist_sq: cython.float = 9999999.0  # 最小距离平方缓存

        # 比较三点 寻找最近起点
        dx = _vtx_pos[v0, 0] - hit_x
        dy = _vtx_pos[v0, 1] - hit_y
        dz = _vtx_pos[v0, 2] - hit_z
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_vtx = v0

        dx = _vtx_pos[v1, 0] - hit_x
        dy = _vtx_pos[v1, 1] - hit_y
        dz = _vtx_pos[v1, 2] - hit_z
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_vtx = v1

        dx = _vtx_pos[v2, 0] - hit_x
        dy = _vtx_pos[v2, 1] - hit_y
        dz = _vtx_pos[v2, 2] - hit_z
        dist_sq = dx * dx + dy * dy + dz * dz
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_vtx = v2

        with cython.nogil:
            # 初始化 BFS (广度优先搜索) 队列 _out_idx 直接用作队列内存
            _epochs[closest_vtx] = _curr_epoch
            _out_idx[0] = closest_vtx
            _out_w[0] = min_dist_sq

            current_idx: cython.int = 0  # 队列读取游标
            total_found: cython.int = 1  # 队列写入游标 (同时也是命中总数)
            v_curr: cython.int  # 蔓延当前节点
            v_next: cython.int  # 蔓延下一节点
            edge_start: cython.int  # CSR 邻接表起始下标
            edge_end: cython.int  # CSR 邻接表终止下标

            # 沿拓扑边向外蔓延
            while current_idx < total_found:
                v_curr = _out_idx[current_idx]
                current_idx += 1

                edge_start = _adj_off[v_curr]
                edge_end = _adj_off[v_curr + 1]

                for j in range(edge_start, edge_end):
                    v_next = _adj_idx[j]

                    # 使用 epoch 判断该点在当前运算中是否被访问过
                    if _epochs[v_next] != _curr_epoch:
                        _epochs[v_next] = _curr_epoch

                        vx = _vtx_pos[v_next, 0]
                        vy = _vtx_pos[v_next, 1]
                        vz = _vtx_pos[v_next, 2]

                        # 🌟 同样的胶囊体投影魔法
                        if ab_sq < 0.000001:
                            dx = vx - hit_x
                            dy = vy - hit_y
                            dz = vz - hit_z
                        else:
                            ap_x = vx - prev_x
                            ap_y = vy - prev_y
                            ap_z = vz - prev_z
                            t_proj = (ap_x * ab_x + ap_y * ab_y + ap_z * ab_z) / ab_sq
                            if t_proj < 0.0:
                                t_proj = 0.0
                            elif t_proj > 1.0:
                                t_proj = 1.0
                            proj_x = prev_x + t_proj * ab_x
                            proj_y = prev_y + t_proj * ab_y
                            proj_z = prev_z + t_proj * ab_z
                            dx = vx - proj_x
                            dy = vy - proj_y
                            dz = vz - proj_z

                        dist_sq = dx * dx + dy * dy + dz * dz

                        # 如果连接的顶点依然在笔刷空间球体内 则加入队列
                        if dist_sq <= radius_sq:
                            _out_idx[total_found] = v_next
                            _out_w[total_found] = dist_sq
                            total_found += 1

            # 就地将队列里的距离平方 转换为对应的衰减权重
            for i in range(total_found):
                dist_sq = _out_w[i]
                t2 = dist_sq / radius_sq
                if falloff_mode == 2:
                    weight = 1.0
                else:
                    if falloff_mode == 1:
                        weight = 1.0 - t2
                        weight = weight * weight
                    elif falloff_mode == 0:
                        weight = 1.0 - sqrt(t2)
                    elif falloff_mode == 3:
                        weight = sqrt(1.0 - t2)
                    elif falloff_mode == 4:
                        t = sqrt(t2)
                        weight = 1.0 - t
                        weight = weight * weight * weight
                    else:
                        weight = 1.0
                _out_w[i] = weight

        self.active_hit_count = total_found
        return (total_found, self.active_hit_indices[:total_found], self.active_hit_falloff[:total_found])
        # endregion

    # endregion


@cython.cclass
class BrushUndoRecorder:
    """
    记录内存中的数据快照
    在修改内存中数据前后调用类中方法 以记录原始数据 后修改后的数据
    数据以稀疏方式保存 并且赋值返回

    在笔刷绘制之前,我们先实例这个类
    - 1. 调用 `begin_stroke` 方法重置记录器
    - 2. 在修改数据之前调用 `record_snapshot` 记录我们将要修改的数据原始数据
    - 3. 所有修改结束后调用 `end_stroke` 用来返回我们的原始数据和最新数据(原始/新数据以稀疏方式返回,用来处理undo/redo)

    Attributes:
        modified_buffer (cython.float[:, ::1]): 需要被修改的目标数据 2D shape(N, channel_count).
        channel_count (cython.int): 数据的通道数/列宽 (如 XYZ = 3, 骨骼权重 = influencesCount).

        modified_vtx_count (cython.int): 当前行程实际修改的顶点总数.
        modified_vtx_bool_buffer (cython.uchar[::1]): 防重录掩码 记录顶点在当前行程中是否已生成过快照 1D shape(N,).
        modified_vtx_indices_buffer (cython.int[::1]): 当前行程涉及的所有被修改的顶点物理索引池 1D shape(N,).

        undo_buffer (cython.float[:, ::1]): 撤销内存池 存储顶点被修改前的原始快照.
    """

    modified_buffer: cython.float[:, ::1]
    channel_count: cython.int
    modified_vtx_bool_buffer: cython.uchar[::1]

    modified_vtx_count: cython.int
    modified_vtx_indices_buffer: cython.int[::1]
    undo_buffer: cython.float[:, ::1]

    # region ---------- init
    def __init__(
        self,
        modified_buffer: cython.float[:, ::1],
        modified_vtx_indices_buffer: cython.int[::1] = None,
        modified_vtx_bool_buffer: cython.uchar[::1] = None,
        undo_buffer: cython.float[:, ::1] = None,
    ):
        """
        Args:
            modified_buffer (cython.float[:, ::1]): 需要被修改的目标数据 2D shape(N, channel_count)
            modified_vtx_indices_buffer (cython.int): 当前行程涉及的所有被修改的顶点物理索引池 1D shape(N,),如果为`None`自动从modified_buffer的shape数据生成
            modified_vtx_bool_buffer (cython.uchar): 防重录掩码 记录顶点在当前行程中是否已生成过快照 1D shape(N,),如果为`None`自动从modified_buffer的shape数据生成
            undo_buffer (cython.float[:, ::1]): 撤销内存池 存储顶点被修改前的原始快照,如果为`None`自动从modified_buffer的shape数据生成
        """
        self.modified_buffer = modified_buffer
        self.channel_count = modified_buffer.shape[1]
        self.modified_vtx_count = 0

        vtx_count: cython.int = modified_buffer.shape[0]

        # 1. 分配或接收“防重录”标记缓冲区
        if modified_vtx_bool_buffer is None:
            c_bool_arr = (ctypes.c_uint8 * vtx_count)()
            self.modified_vtx_bool_buffer = memoryview(c_bool_arr)
        else:
            self.modified_vtx_bool_buffer = modified_vtx_bool_buffer

        # 2. 分配或接收“被修改顶点索引”缓冲区
        if modified_vtx_indices_buffer is None:
            c_indices_arr = (ctypes.c_int32 * vtx_count)()
            self.modified_vtx_indices_buffer = memoryview(c_indices_arr)
        else:
            self.modified_vtx_indices_buffer = modified_vtx_indices_buffer

        # 3. 分配或接收“撤销”数据缓冲区
        if undo_buffer is None:
            flat_size = vtx_count * self.channel_count
            c_undo_arr = (ctypes.c_float * flat_size)()
            self.undo_buffer = memoryview(c_undo_arr).cast("f", shape=(vtx_count, self.channel_count))
        else:
            self.undo_buffer = undo_buffer

    # endregion

    # region ---------- Begin stroke
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def begin_stroke(self) -> tuple:
        """
        重置记录状态,开始新的一轮记录
        """
        _mask = self.modified_vtx_bool_buffer
        vtx_count: cython.int = cython.cast(cython.int, _mask.shape[0])

        self.modified_vtx_count = 0

        # 使用 memset 快速将内存区域清零
        memset(
            cython.cast(cython.p_void, cython.address(_mask[0])),
            0,
            vtx_count * cython.sizeof(cython.uchar),
        )

    # endregion

    # region ---------- Tick undo snapshot
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def record_snapshot(
        self,
        record_indices: cython.int[::1] = None,  # 设为可选
    ) -> cython.void:
        """
        记录数据快照

        Args:
            record_indices (cython.int[::1]): 要记录数据的indices,如果为`None`则记录全部数据
        """
        # 为提高性能 将类属性提取到局部变量
        _mod_buf = self.modified_buffer
        _undo_buf = self.undo_buffer
        _mask = self.modified_vtx_bool_buffer
        _idx_pool = self.modified_vtx_indices_buffer
        _channels: cython.int = self.channel_count
        _current_count: cython.int = self.modified_vtx_count

        # 确定是全量处理还是部分处理
        use_all: cython.bint = record_indices is None
        final_count: cython.int
        if use_all:  # noqa: SIM108
            final_count = _mod_buf.shape[0]
        else:
            final_count = record_indices.shape[0]

        # --- 核心循环 ---
        i: cython.int
        j: cython.int
        vtx_idx: cython.int

        for i in range(final_count):
            # 获取顶点索引
            vtx_idx = i if use_all else record_indices[i]

            # 检查是否已记录 避免重复工作
            if _mask[vtx_idx] == 0:
                _mask[vtx_idx] = 1  # 标记为已记录

                # 备份修改前的数据快照
                for j in range(_channels):
                    _undo_buf[vtx_idx, j] = _mod_buf[vtx_idx, j]

                # 记录被修改的顶点索引并递增计数
                _idx_pool[_current_count] = vtx_idx
                _current_count += 1

        self.modified_vtx_count = _current_count

    # endregion

    # region ---------- End Stroke
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def end_stroke(self) -> tuple:
        """
        结束快照记录

        Return:
            tuple: 包含所有增量信息的元组
                - modified_indices (Array[c_int32]): 修改的indices
                - modified_channel_indices (Array[c_int32]): 修改的channels
                - old_sparse_ary (Array[c_float]): 原始稀疏数据备份
                - new_sparse_ary (Array[c_float]): 新的稀疏数据

        获取实际的顶点 ID: actual_id = modified_indices[k]
        获取实际的通道 ID: actual_channel_id = modified_channel_indices[i]
        计算在稀疏数组中的索引: sparse_index = k * len(modified_channel_indices) + i
        获取旧值: old_value = old_sparse_ary[sparse_index]
        获取新值: new_value = new_sparse_ary[sparse_index]
        """
        if self.modified_vtx_count == 0:
            return None

        # 将类属性提取到局部变量以提高访问速度
        _modified_vtx_count: cython.int = self.modified_vtx_count
        _channel_count: cython.int = self.channel_count
        _modified = self.modified_buffer
        _indices = self.modified_vtx_indices_buffer
        _undo = self.undo_buffer

        i: cython.int
        j: cython.int
        vtx_idx: cython.int
        diff: cython.float

        # --- 通道压缩 ---
        # 1. 分配临时内存 标记被修改过的通道
        channel_is_dirty: cython.p_char = cython.cast(cython.p_char, calloc(_channel_count, cython.sizeof(cython.char)))

        # 2. 遍历已记录的顶点 比较新旧数据差异
        modified_channel_count: cython.int = 0
        for i in range(_modified_vtx_count):
            vtx_idx = _indices[i]
            for j in range(_channel_count):
                if channel_is_dirty[j] == 0:
                    diff = _modified[vtx_idx, j] - _undo[i, j]
                    if fabs(diff) > 1e-6:
                        channel_is_dirty[j] = 1
                        modified_channel_count += 1

        # 3. 如果没有任何通道被修改 则提前退出
        if modified_channel_count == 0:
            free(channel_is_dirty)
            return None

        # 4. 创建一个只包含“脏”通道索引的数组
        modified_channel_indices = (ctypes.c_int32 * modified_channel_count)()
        modified_channel_view: cython.int[::1] = modified_channel_indices

        write_channel_idx: cython.int = 0
        for j in range(_channel_count):
            if channel_is_dirty[j] == 1:
                modified_channel_view[write_channel_idx] = j
                write_channel_idx += 1

        # 5. 释放临时标记内存
        free(channel_is_dirty)

        # --- 数据打包 ---
        # 6. 创建并填充被修改顶点的索引数组
        modified_indices = (ctypes.c_int32 * _modified_vtx_count)()
        modified_vtx_indices_view: cython.int[::1] = modified_indices
        for i in range(_modified_vtx_count):
            modified_vtx_indices_view[i] = _indices[i]

        # 7. 分配用于存储稀疏数据的1D数组
        sparse_size: cython.int = _modified_vtx_count * modified_channel_count
        old_sparse_ary = (ctypes.c_float * sparse_size)()
        old_sparse_view: cython.float[::1] = old_sparse_ary
        new_sparse_ary = (ctypes.c_float * sparse_size)()
        new_sparse_view: cython.float[::1] = new_sparse_ary

        # 8. 遍历并填充稀疏数据数组
        write_idx: cython.int = 0
        channel_idx: cython.int = 0
        for i in range(_modified_vtx_count):
            vtx_idx = _indices[i]
            for j in range(modified_channel_count):
                channel_idx = modified_channel_view[j]

                # 从undo/redo缓冲区提取数据并填充到一维稀疏数组中
                old_sparse_view[write_idx] = _undo[vtx_idx, channel_idx]
                new_sparse_view[write_idx] = _modified[vtx_idx, channel_idx]
                write_idx += 1

        # 9. 返回包含所有增量信息的元组
        return (
            modified_indices,
            modified_channel_indices,
            old_sparse_ary,
            new_sparse_ary,
        )

    # endregion


# fmt:off
@cython.cclass
class BrushStrokeContext:
    """
    笔刷终极上下文 (The Ultimate Context)
    完美合并了 UI 配置参数与底层物理内存视图！
    """
    # ==========================================
    # 1. 静态参数区 (由 Python 在鼠标按下时填入)
    # ==========================================
    brush_mode     : cython.int
    values         : cython.float[::1]
    channel_indices: cython.int[::1]
    pressure       : cython.float
    clamp_min      : cython.float
    clamp_max      : cython.float
    iterations     : cython.int
    normalize      : cython.bint

    # ==========================================
    # 2. 动态内存区 (由 Cython 在射线命中后挂载)
    # ==========================================
    vertex_count      : cython.int
    vertex_buffer     : cython.int[::1]
    falloff_buffer    : cython.float[::1]
    undo_buffer       : cython.float[:, ::1]
    max_falloff_buffer: cython.float[::1]

    def __init__(
        self,
        brush_mode      : cython.int,
        values          : cython.float[::1],
        channel_indices : cython.int[::1],
        pressure        : cython.float = 1.0,
        clamp_min       : cython.float = 0.0,
        clamp_max       : cython.float = 1.0,
        iterations      : cython.int = 1,
        normalize       : cython.bint = True
    ):
        self.brush_mode      = brush_mode
        self.values          = values
        self.channel_indices = channel_indices
        self.pressure        = pressure
        self.clamp_min       = clamp_min
        self.clamp_max       = clamp_max
        self.iterations      = iterations
        self.normalize       = normalize

        # 动态内存区初始化为 0 或 None，等引擎来挂载
        self.vertex_count = 0
# fmt:on


@cython.cclass
class BrushMathEngine:
    """通用笔刷数学运算处理引擎 (纯 Python 模式).

    只负责接收顶点、衰减、目标数组并执行纯逻辑运算.
    """

    # 类成员变量声明 (C 级字段)
    adj_offsets: cython.int[::1]
    adj_indices: cython.int[::1]
    modified_buffer: cython.float[:, ::1]

    def __init__(
        self,
        modified_buffer: cython.float[:, ::1],
        adj_offsets: cython.int[::1] = None,
        adj_indices: cython.int[::1] = None,
    ):

        self.modified_buffer = modified_buffer
        self.adj_offsets = adj_offsets
        self.adj_indices = adj_indices

    # region ---------- Exec Math
    @cython.cfunc
    def _execute_math_step(self, ctx: BrushStrokeContext) -> cython.void:
        """动态分发: 0-3使用防重叠绝对计算, 4-5使用原始的相对迭代计算"""
        brush_mode = ctx.brush_mode
        if brush_mode == 0 or brush_mode == 1 or brush_mode == 2 or brush_mode == 3:
            self._math_standard_stroke(ctx)
        elif brush_mode == 4:
            self._math_smooth(ctx)
        elif brush_mode == 5:
            self._math_sharp(ctx)

    # endregion

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.ccall
    def _math_standard_stroke(self, ctx: BrushStrokeContext) -> cython.void:

        brush_mode = ctx.brush_mode

        i: cython.int = 0
        j: cython.int = 0
        row: cython.int = 0
        col: cython.int = 0
        fal: cython.float = 0.0
        orig_val: cython.float = 0.0

        _array = self.modified_buffer
        _v_buf = ctx.vertex_buffer
        _f_buf = ctx.falloff_buffer
        _vals = ctx.values
        _pressure = ctx.pressure
        _ch_idx = ctx.channel_indices

        v_count: cython.int = ctx.vertex_count
        c_count: cython.int = _ch_idx.shape[0]
        _min: cython.float = ctx.clamp_min
        _max: cython.float = ctx.clamp_max

        # ==========================================
        if brush_mode == 0:  # Add 加法
            for i in range(v_count):
                row = _v_buf[i]
                fal = _f_buf[i] * _pressure
                if fal > ctx.max_falloff_buffer[row]:
                    ctx.max_falloff_buffer[row] = fal
                    for j in range(c_count):
                        col = _ch_idx[j]
                        orig_val = ctx.undo_buffer[row, col]
                        _array[row, col] = _clamp_float(orig_val + (fal * _vals[j]), _min, _max)

        # ==========================================
        elif brush_mode == 1:  # Sub 减法
            for i in range(v_count):
                row = _v_buf[i]
                fal = _f_buf[i] * _pressure
                if fal > ctx.max_falloff_buffer[row]:
                    ctx.max_falloff_buffer[row] = fal
                    for j in range(c_count):
                        col = _ch_idx[j]
                        orig_val = ctx.undo_buffer[row, col]
                        _array[row, col] = _clamp_float(orig_val - (fal * _vals[j]), _min, _max)

        # ==========================================
        elif brush_mode == 2:  # Replace 替换
            for i in range(v_count):
                row = _v_buf[i]
                fal = _f_buf[i] * _pressure
                if fal > ctx.max_falloff_buffer[row]:
                    ctx.max_falloff_buffer[row] = fal
                    for j in range(c_count):
                        col = _ch_idx[j]
                        orig_val = ctx.undo_buffer[row, col]
                        _array[row, col] = _clamp_float(orig_val + (_vals[j] - orig_val) * fal, _min, _max)

        # ==========================================
        elif brush_mode == 3:  # Multiply 乘法
            for i in range(v_count):
                row = _v_buf[i]
                fal = _f_buf[i] * _pressure
                if fal > ctx.max_falloff_buffer[row]:
                    ctx.max_falloff_buffer[row] = fal
                    for j in range(c_count):
                        col = _ch_idx[j]
                        orig_val = ctx.undo_buffer[row, col]
                        _array[row, col] = _clamp_float(orig_val + (orig_val * _vals[j] - orig_val) * fal, _min, _max)

    # endregion
    # region ---------- Smooth
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.ccall
    def _math_smooth(self, ctx: BrushStrokeContext) -> cython.void:
        if self.adj_offsets is None or self.adj_indices is None:
            raise RuntimeError("Smooth Error: Topology (adj_offsets/indices) is not initialized.")

        # 拓扑专用变量声明
        i: cython.int = 0
        j: cython.int = 0
        n: cython.int = 0
        row: cython.int = 0
        col: cython.int = 0
        n_idx: cython.int = 0
        edge_start: cython.int = 0
        edge_end: cython.int = 0
        n_count: cython.int = 0
        fal: cython.float = 0.0
        n_sum: cython.float = 0.0
        avg: cython.float = 0.0

        _array = self.modified_buffer
        _v_buf = ctx.vertex_buffer
        _f_buf = ctx.falloff_buffer
        _vals = ctx.values
        _ch_idx = ctx.channel_indices
        _adj_off = self.adj_offsets
        _adj_idx = self.adj_indices

        v_count: cython.int = ctx.vertex_count
        c_count: cython.int = _ch_idx.shape[0]
        _min: cython.float = ctx.clamp_min
        _max: cython.float = ctx.clamp_max

        for i in range(v_count):
            row = _v_buf[i]
            fal = _f_buf[i]
            edge_start = _adj_off[row]
            edge_end = _adj_off[row + 1]
            n_count = edge_end - edge_start

            if n_count > 0:
                for j in range(c_count):
                    col = _ch_idx[j]
                    n_sum = 0.0
                    for n in range(edge_start, edge_end):
                        n_idx = _adj_idx[n]
                        n_sum += _array[n_idx, col]

                    avg = n_sum / n_count
                    _array[row, col] = _clamp_float(
                        _array[row, col] + (avg - _array[row, col]) * (_vals[j] * fal),
                        _min,
                        _max,
                    )

    # endregion
    # region ---------- Sharp
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.ccall
    def _math_sharp(self, ctx: BrushStrokeContext) -> cython.void:
        i: cython.int = 0
        j: cython.int = 0
        row: cython.int = 0
        col: cython.int = 0
        fal: cython.float = 0.0
        val: cython.float = 0.0

        _array = self.modified_buffer
        _v_buf = ctx.vertex_buffer
        _f_buf = ctx.falloff_buffer
        _vals = ctx.values
        _ch_idx = ctx.channel_indices

        v_count: cython.int = ctx.vertex_count
        c_count: cython.int = _ch_idx.shape[0]
        _min: cython.float = ctx.clamp_min
        _max: cython.float = ctx.clamp_max

        for i in range(v_count):
            row = _v_buf[i]
            fal = _f_buf[i]
            for j in range(c_count):
                col = _ch_idx[j]
                val = _array[row, col]
                val += (val - 0.5) * (_vals[j] * fal) * 2.0
                _array[row, col] = _clamp_float(val, _min, _max)

    # endregion

    # region ---------- Get (Copy)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def get_custom_array(
        self,
        vertex_indices=None,
        channel_indices=None,
    ) -> array.array:
        """
        从原始缓冲区中提取特定行列,并“压扁”为 1D 副本.

        该函数常用于:
        1. UI 展示:只提取当前选中的顶点数据进行显示.
        2. 数据导出:将计算结果导出为 Python 标准 array 供外部模块如 JSON 使用.
        3. 性能优化:通过将非连续的内存块汇聚到连续的 1D 数组中,方便后续的快速处理.

        Args:
            vertex_indices  : 目标顶点行号序列.None 表示全量.
            channel_indices : 目标通道列号序列.None 表示全量.

        Returns:
            array.array: 一个全新的、包含提取数据的 1D 浮点数组 (Copy)
        """
        _array = self.modified_buffer

        use_all_v: cython.bint = True
        if (vertex_indices is not None) and (len(vertex_indices) > 0):
            use_all_v = False

        use_all_c: cython.bint = True
        if (channel_indices is not None) and (len(channel_indices) > 0):
            use_all_c = False

        _vtx_view: cython.int[::1] = None if use_all_v else vertex_indices
        _ch_view: cython.int[::1] = None if use_all_c else channel_indices

        _v_count: cython.int = _array.shape[0] if use_all_v else _vtx_view.shape[0]
        _c_count: cython.int = _array.shape[1] if use_all_c else _ch_view.shape[0]

        if _v_count == 0 or _c_count == 0:
            return array.array("f")

        total_size: cython.int = _v_count * _c_count
        out_ary = array.array("f", [0.0]) * total_size
        out_view: cython.float[::1] = out_ary

        i: cython.int
        j: cython.int
        row: cython.int
        col: cython.int
        write_idx: cython.int = 0

        for i in range(_v_count):
            row = i if use_all_v else _vtx_view[i]

            for j in range(_c_count):
                col = j if use_all_c else _ch_view[j]

                out_view[write_idx] = _array[row, col]
                write_idx += 1

        return out_ary

    # endregion

    # region ---------- Set(Paste)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def set_custom_array(
        self,
        source_values   : cython.float[::1]       ,
        alpha           : cython.float      = 1.0 ,
        blend_mode      : cython.int        = 2   ,
        vertex_indices                      = None,
        channel_indices                     = None,
        falloff_weights                     = None,
        clamp_min       : cython.float      = 0.0 ,
        clamp_max       : cython.float      = 1.0 ,
    ) -> tuple:  # fmt: skip
        """
        将外部传入的 1D 数组数据高效写回(粘贴)到内部 2D 缓冲区中.

        该函数是 get_custom_array 的逆向操作,主要用于 Undo/Redo 数据应用、
        从外部脚本强制同步权重数据等场景.支持全量写入或基于索引的稀疏写入.

        逻辑链路:
        1. 自动识别输入模式(全量/稀疏/带权重).
        2. 基于指定的混合模式(Blend Mode)将 source_values 与原数据融合.
        3. 自动执行数值限制(Clamping).

        Args:
            source_values   : 待写入的扁平化(1D)浮点数组,长度应为 len(V) * len(C).
            alpha           : 全局强度系数,影响最终混合的贡献度.
            blend_mode      : 混合模式 (0:加法, 1:减法, 2:替换, 3:乘法),默认为2替换模式
            vertex_indices  : 目标顶点索引序列.若为 None,则视为对所有顶点操作.
            channel_indices : 目标通道索引序列.若为 None,则视为对所有通道操作.
            falloff_weights : 顶点级别的权重数组(1D).若提供,将与 alpha 相乘作为最终权重.
            clamp_min       : 数值安全边界限制.
            clamp_max       : 数值安全边界限制.

        Returns:
            tuple : (处理的顶点数, 传入的索引, 修改后的缓冲区引用)
        """
        _array = self.modified_buffer

        use_all_v: cython.bint = True
        if (vertex_indices is not None) and (len(vertex_indices) > 0):
            use_all_v = False

        use_all_c: cython.bint = True
        if (channel_indices is not None) and (len(channel_indices) > 0):
            use_all_c = False

        use_fal: cython.bint = False
        if (falloff_weights is not None) and (len(falloff_weights) > 0):
            use_fal = True

        _vtx_view: cython.int[::1] = None if use_all_v else vertex_indices
        _ch_view: cython.int[::1] = None if use_all_c else channel_indices
        _fal_view: cython.float[::1] = None if not use_fal else falloff_weights

        # 3. 确定循环边界
        _v_count: cython.int = _array.shape[0] if use_all_v else _vtx_view.shape[0]
        _c_count: cython.int = _array.shape[1] if use_all_c else _ch_view.shape[0]

        if _v_count == 0 or _c_count == 0:
            return (0, vertex_indices, self.modified_buffer)

        i: cython.int
        j: cython.int
        row: cython.int
        col: cython.int
        fal: cython.float
        v_j: cython.float
        val: cython.float
        cur: cython.float

        for i in range(_v_count):
            row = i if use_all_v else _vtx_view[i]
            fal = (_fal_view[i] * alpha) if use_fal else alpha

            for j in range(_c_count):
                col = j if use_all_c else _ch_view[j]

                v_j = source_values[i * _c_count + j]
                cur = _array[row, col]

                val = _blend_math(blend_mode, cur, v_j, fal)
                _array[row, col] = _clamp_float(val, clamp_min, clamp_max)

        return (_v_count, vertex_indices, self.modified_buffer)

    # endregion


@cython.cclass
class UtilBrushProcessor:
    """通用笔刷调度中心 (基类).

    协调 Core (物理查询)、Recorder (历史快照) 与 MathEngine (核心算法).
    """

    core: CoreBrushEngine
    recorder: BrushUndoRecorder
    math_engine: BrushMathEngine
    modified_buffer: cython.float[:, ::1]

    max_falloff_buffer: cython.float[::1]  # 新增
    undo_buffer: cython.float[:, ::1]  # 新增

    def __init__(
        self,
        core: CoreBrushEngine,
        modified_buffer: cython.float[:, ::1],
        modified_indices_buffer: cython.int[::1] = None,
        modified_bool_buffer: cython.uchar[::1] = None,
        undo_buffer: cython.float[:, ::1] = None,
    ):
        self.core = core
        self.modified_buffer = modified_buffer

        self.undo_buffer = undo_buffer

        vtx_count: cython.int = core.vtx_positions2D.shape[0]

        if vtx_count > 0:
            c_float_array = (ctypes.c_float * vtx_count)()
            self.max_falloff_buffer = c_float_array

        # 1. 初始化 Recorder (撤销重做管理器)
        self.recorder = BrushUndoRecorder(
            modified_buffer,
            modified_indices_buffer,
            modified_bool_buffer,
            undo_buffer,
        )

        # 2. 初始化 MathEngine (将 Core 的拓扑数据直接桥接给数学引擎)
        self.math_engine = BrushMathEngine(
            modified_buffer,
            adj_offsets=core.adj_offsets,
            adj_indices=core.adj_indices,
        )

    @cython.ccall
    def begin_stroke(self) -> tuple:
        """重置记录器掩码,准备新一轮绘画"""
        res = self.recorder.begin_stroke()

        if self.max_falloff_buffer is not None and self.max_falloff_buffer.shape[0] > 0:
            self.max_falloff_buffer[:] = 0.0
        return res

    @cython.ccall
    def end_stroke(self) -> tuple:
        """完成绘画,返回所有增量修改数据(用于 Undo 系统)"""
        return self.recorder.end_stroke()

    # region ---------- Apply Brush
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def process_stroke(self, brush_stroke_context: BrushStrokeContext) -> tuple:
        """
        统一笔刷计算入口.
        命中检测 -> 构造上下文 -> 记录快照 -> 数学迭代.
        """
        raycast_core = self.core

        # 1. 命中检测拦截
        if raycast_core.active_hit_count == 0:
            return (0, raycast_core.active_hit_indices, self.modified_buffer)

        # 2. 调度快照引擎:在修改前备份受影响顶点的原始数据
        self.recorder.record_snapshot(raycast_core.active_hit_indices)

        # ==========================================
        # UI 层传进来的 ctx 里只有静态配置，我们在这里把 raycast/falloff 层的物理内存给它
        # ==========================================
        brush_stroke_context.vertex_count = self.core.active_hit_count
        brush_stroke_context.vertex_buffer = self.core.active_hit_indices
        brush_stroke_context.falloff_buffer = self.core.active_hit_falloff
        brush_stroke_context.undo_buffer = self.undo_buffer
        brush_stroke_context.max_falloff_buffer = self.max_falloff_buffer

        # 4. 执行数学运算迭代
        _iter: cython.int = 0
        for _iter in range(brush_stroke_context.iterations):
            # 衔接新的 MathEngine 分发接口
            self.math_engine._execute_math_step(brush_stroke_context)

        return (
            raycast_core.active_hit_count,
            raycast_core.active_hit_indices,
            self.modified_buffer,
        )

    # endregion

    # region ---------- Brush Get/Set
    @cython.ccall
    def get_custom_array(self, vertex_indices=None, channel_indices=None) -> array.array:
        """提取指定范围的数据副本."""
        return self.math_engine.get_custom_array(vertex_indices, channel_indices)

    @cython.ccall
    def set_custom_array(
        self,
        source_values,
        alpha=1.0,
        blend_mode=2,
        vertex_indices=None,
        channel_indices=None,
        falloff_weights=None,
        clamp_min=0.0,
        clamp_max=1.0,
    ) -> tuple:
        """执行稀疏或全量数据的直接写回."""
        # 记录快照逻辑保持在调度层,确保 set 操作也能被撤销
        self.recorder.record_snapshot(vertex_indices)

        return self.math_engine.set_custom_array(
            source_values,
            alpha,
            blend_mode,
            vertex_indices,
            channel_indices,
            falloff_weights,
            clamp_min,
            clamp_max,
        )

    # =========================================================================
    # 🚀 图层合成专属  (Layer Compositing Native Operators)
    # =========================================================================
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def clear_buffer_sparse(self, vertex_indices=None) -> cython.void:
        """ 局部或全量清零画布，替代 Python 的 memset 或循环"""
        _out = self.modified_buffer
        _channels: cython.int = _out.shape[1]

        use_all_v: cython.bint = True
        if (vertex_indices is not None) and (len(vertex_indices) > 0):
            use_all_v = False

        _vtx_view: cython.int[::1] = None if use_all_v else vertex_indices
        _v_count: cython.int = _out.shape[0] if use_all_v else _vtx_view.shape[0]

        i: cython.int
        j: cython.int
        row: cython.int

        for i in range(_v_count):
            row = i if use_all_v else _vtx_view[i]
            for j in range(_channels):
                _out[row, j] = 0.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def add_layer_weights(self, layer_weights: cython.float[:1], layer_mask: cython.float[:1], vertex_indices=None) -> cython.void:
        """
        专为加法蒙版合成 (Additive Compositing) 设计
        完美处理：全量输入数组 + 稀疏索引拦截
        """
        _out = self.modified_buffer
        _channels: cython.int = _out.shape[1]

        use_all_v: cython.bint = True
        if (vertex_indices is not None) and (len(vertex_indices) > 0):
            use_all_v = False

        _vtx_view: cython.int[::1] = None if use_all_v else vertex_indices
        _v_count: cython.int = _out.shape[0] if use_all_v else _vtx_view.shape[0]

        i: cython.int
        j: cython.int
        row: cython.int
        mask_val: cython.float

        for i in range(_v_count):
            row = i if use_all_v else _vtx_view[i]
            mask_val = layer_mask[row]

            if mask_val <= 0.000001:
                continue

            for j in range(_channels):
                # 公式: Out[v, c] += Layer[v, c] * Mask[v]
                _out[row, j] += layer_weights[row * _channels + j] * mask_val

    # endregion


@cython.cclass
class SkinWeightProcessor(UtilBrushProcessor):
    """蒙皮权重专属笔刷处理器.

    继承自 `UtilBrushProcessor` 直接复用其底层引用的 Core, Recorder 和 MathEngine.
    以及免费继承了 begin_stroke, end_stroke 等基础方法.
    只专注于添加骨骼锁定和归一化的特有逻辑.
    """

    influences_locks_buffer: cython.uchar[::1]
    channel_count: cython.int

    # region ---------------------- Init
    def __init__(
        self,
        core: CoreBrushEngine,
        modified_buffer: cython.float[:, ::1],
        modified_vtx_indices_buffer: cython.int[::1],
        modified_vtx_bool_buffer: cython.uchar[::1],
        influences_locks_buffer: cython.uchar[::1],
        undo_buffer: cython.float[:, ::1],
    ):
        """初始化权重笔刷 通过 super() 调用父类完成底层引擎的组装."""
        UtilBrushProcessor.__init__(
            self,
            core,
            modified_buffer,
            modified_vtx_indices_buffer,
            modified_vtx_bool_buffer,
            undo_buffer,
        )
        self.influences_locks_buffer = influences_locks_buffer
        self.channel_count = modified_buffer.shape[1]

    # endregion

    # region ---------------------- Apply Weight
    @cython.ccall
    def process_stroke(self, brush_stroke_context: BrushStrokeContext, normalize: bool = True) -> tuple:
        """执行蒙皮权重运算 支持单骨骼或多骨骼统一调度."""

        # 1. 调用父类核心逻辑
        res = UtilBrushProcessor.process_stroke(self, brush_stroke_context)

        # 2. 判定有效修改并执行归一化
        if res[0] > 0:
            # 以 channel_indices 中的第一根骨骼作为优先级骨骼，即当前正在绘制的骨骼，保护它的权重不被随意缩放
            priority_idx: cython.int = brush_stroke_context.channel_indices[0]

            # 只对命中的顶点切片进行归一化
            hit_indices_view = res[1][: res[0]]

            if normalize is True:
                # 执行归一化
                self.normalize_weights(hit_indices_view, priority_idx)

        return res

    # endregion

    # region ---------------------- Normalize
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.ccall
    def normalize_weights(
        self,
        vertex_indices                 = None, 
        priority_influence: cython.int = -1,
    ) -> cython.float[:, ::1]:  # fmt:skip
        """
        权重归一化 (Normalize Weights)

        Args:
            vertex_indices (Optional[cython.int[::1]]):
                目标顶点的物理索引数组。
                - 【局部稀疏】：在笔刷拖拽时，传入被画笔影响的局部顶点，实现光速归一化。
                - 【全量操作】：如果传入 None 或空数组，则对当前图层的所有顶点执行全量归一化。

            priority_influence (cython.int):
                最高优先级骨骼（主权骨骼）的通道索引。默认值为 -1（无优先级，众生平等）。
                - 在笔刷涂抹时，当前正在绘制的骨骼就是 priority_influence。
                  归一化系统会**绝对保护**它的权重值不被随意缩放，而是强迫其他未锁定的骨骼

        Returns:
            cython.float[:, ::1]: 返回修改后的完整 2D 权重内存视图。

        """
        _locks = self.influences_locks_buffer
        _w2D = self.modified_buffer
        num_influences: cython.int = self.channel_count

        use_all_v: cython.bint = True
        if (vertex_indices is not None) and (len(vertex_indices) > 0):
            use_all_v = False

        _vtx_view: cython.int[::1] = None if use_all_v else vertex_indices

        _count: cython.int = _w2D.shape[0] if use_all_v else _vtx_view.shape[0]

        if _count == 0:
            return self.modified_buffer

        i: cython.int
        j: cython.int
        v_idx: cython.int
        locked_sum: cython.float
        unlocked_sum: cython.float
        active_weight: cython.float
        remaining_weight: cython.float
        scale_factor: cython.float
        available_bones_count: cython.int

        for i in range(_count):
            v_idx = i if use_all_v else _vtx_view[i]

            locked_sum = 0.0
            unlocked_sum = 0.0
            available_bones_count = 0

            for j in range(num_influences):
                if _locks[j] == 1:
                    locked_sum += _w2D[v_idx, j]
                elif j != priority_influence:
                    unlocked_sum += _w2D[v_idx, j]
                    available_bones_count += 1

            active_weight = 0.0
            if priority_influence != -1:
                active_weight = _w2D[v_idx, priority_influence]
                if active_weight > 1.0 - locked_sum:
                    active_weight = 1.0 - locked_sum
                    _w2D[v_idx, priority_influence] = active_weight

            remaining_weight = 1.0 - locked_sum - active_weight

            if available_bones_count == 0:
                if priority_influence != -1:
                    _w2D[v_idx, priority_influence] = 1.0 - locked_sum
                continue

            if unlocked_sum > 0.000001:
                scale_factor = remaining_weight / unlocked_sum
                if scale_factor > 0.999999 and scale_factor < 1.000001:
                    continue
                for j in range(num_influences):
                    if j != priority_influence and _locks[j] == 0:
                        _w2D[v_idx, j] *= scale_factor
            else:
                if remaining_weight > 0.000001:
                    scale_factor = remaining_weight / available_bones_count
                    for j in range(num_influences):
                        if j != priority_influence and _locks[j] == 0:
                            _w2D[v_idx, j] = scale_factor

        return self.modified_buffer

    # endregion
