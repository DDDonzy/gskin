import cython
from cython.parallel import prange  # type:ignore

# 在纯 Python 模式下，引入 C 库的方法：
from cython.cimports.openmp import omp_get_thread_num  # type:ignore

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def raycast(
    ray_pos: tuple,
    ray_dir: tuple,
    points: cython.float[:, ::1],
    tri_indices2D: cython.int[:, ::1],
) -> tuple:

    orig_x: cython.float = ray_pos[0]
    orig_y: cython.float = ray_pos[1]
    orig_z: cython.float = ray_pos[2]
    
    dir_x: cython.float = ray_dir[0]
    dir_y: cython.float = ray_dir[1]
    dir_z: cython.float = ray_dir[2]

    num_tris: cython.int = tri_indices2D.shape[0]

    MAX_THREADS: cython.int = 128
    thread_closest_t = cython.declare(cython.float[128])
    thread_hit_tri = cython.declare(cython.int[128])
    thread_u = cython.declare(cython.float[128])
    thread_v = cython.declare(cython.float[128])

    i: cython.int
    tid: cython.int

    for i in range(MAX_THREADS):
        thread_closest_t[i] = 999999.0
        thread_hit_tri[i] = -1
        thread_u[i] = 0.0
        thread_v[i] = 0.0

    v0_idx: cython.int; v1_idx: cython.int; v2_idx: cython.int
    edge1_x: cython.float; edge1_y: cython.float; edge1_z: cython.float
    edge2_x: cython.float; edge2_y: cython.float; edge2_z: cython.float
    h_x: cython.float; h_y: cython.float; h_z: cython.float
    s_x: cython.float; s_y: cython.float; s_z: cython.float
    q_x: cython.float; q_y: cython.float; q_z: cython.float
    a: cython.float; f: cython.float; u: cython.float; v: cython.float; t: cython.float

    for i in prange(num_tris, schedule="guided", nogil=True):
        tid = omp_get_thread_num()
        if tid >= 128:
            tid = 0

        v0_idx = tri_indices2D[i, 0]
        v1_idx = tri_indices2D[i, 1]
        v2_idx = tri_indices2D[i, 2]

        edge1_x = points[v1_idx, 0] - points[v0_idx, 0]
        edge1_y = points[v1_idx, 1] - points[v0_idx, 1]
        edge1_z = points[v1_idx, 2] - points[v0_idx, 2]

        edge2_x = points[v2_idx, 0] - points[v0_idx, 0]
        edge2_y = points[v2_idx, 1] - points[v0_idx, 1]
        edge2_z = points[v2_idx, 2] - points[v0_idx, 2]

        h_x = dir_y * edge2_z - dir_z * edge2_y
        h_y = dir_z * edge2_x - dir_x * edge2_z
        h_z = dir_x * edge2_y - dir_y * edge2_x

        a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z

        if a > -0.0000001 and a < 0.0000001: continue

        f = 1.0 / a
        s_x = orig_x - points[v0_idx, 0]
        s_y = orig_y - points[v0_idx, 1]
        s_z = orig_z - points[v0_idx, 2]

        u = f * (s_x * h_x + s_y * h_y + s_z * h_z)
        if u < 0.0 or u > 1.0: continue

        q_x = s_y * edge1_z - s_z * edge1_y
        q_y = s_z * edge1_x - s_x * edge1_z
        q_z = s_x * edge1_y - s_y * edge1_x

        v = f * (dir_x * q_x + dir_y * q_y + dir_z * q_z)
        if v < 0.0 or u + v > 1.0: continue

        t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)

        if t > 0.000001 and t < thread_closest_t[tid]:
            thread_closest_t[tid] = t
            thread_hit_tri[tid] = i
            thread_u[tid] = u
            thread_v[tid] = v

    # 3. 全局比对
    global_closest_t: cython.float = 999999.0
    global_hit_tri: cython.int = -1
    global_u: cython.float = 0.0
    global_v: cython.float = 0.0

    for i in range(MAX_THREADS):
        if thread_closest_t[i] < global_closest_t:
            global_closest_t = thread_closest_t[i]
            global_hit_tri = thread_hit_tri[i]
            global_u = thread_u[i]
            global_v = thread_v[i]

    # 4. 计算最终结果
    hit_x: cython.float; hit_y: cython.float; hit_z: cython.float
    normal_x: cython.float; normal_y: cython.float; normal_z: cython.float

    if global_hit_tri != -1:
        # --- 在 Cython 中计算法线 ---
        v0_idx = tri_indices2D[global_hit_tri, 0]
        v1_idx = tri_indices2D[global_hit_tri, 1]
        v2_idx = tri_indices2D[global_hit_tri, 2]

        edge1_x = points[v1_idx, 0] - points[v0_idx, 0]
        edge1_y = points[v1_idx, 1] - points[v0_idx, 1]
        edge1_z = points[v1_idx, 2] - points[v0_idx, 2]
        edge2_x = points[v2_idx, 0] - points[v0_idx, 0]
        edge2_y = points[v2_idx, 1] - points[v0_idx, 1]
        edge2_z = points[v2_idx, 2] - points[v0_idx, 2]

        raw_normal_x: cython.float = edge1_y * edge2_z - edge1_z * edge2_y
        raw_normal_y: cython.float = edge1_z * edge2_x - edge1_x * edge2_z
        raw_normal_z: cython.float = edge1_x * edge2_y - edge1_y * edge2_x
        
        norm_len: cython.float = (raw_normal_x**2 + raw_normal_y**2 + raw_normal_z**2)**0.5
        if norm_len > 0.000001:
            normal_x = raw_normal_x / norm_len
            normal_y = raw_normal_y / norm_len
            normal_z = raw_normal_z / norm_len
        else:
            normal_x, normal_y, normal_z = 0.0, 0.0, 1.0

        # --- 计算碰撞点坐标 ---
        hit_x = orig_x + dir_x * global_closest_t
        hit_y = orig_y + dir_y * global_closest_t
        hit_z = orig_z + dir_z * global_closest_t
        
        # --- 返回所有结果 ---
        return True, (hit_x, hit_y, hit_z), (normal_x, normal_y, normal_z), global_hit_tri, global_closest_t, global_u, global_v
    else:
        # --- 未击中 ---
        return False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), -1, 0.0, 0.0, 0.0