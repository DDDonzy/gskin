import array

import cython
from cython.cimports.libc.math import sqrt, fabs  # type:ignore
from cython.cimports.libc.stdlib import malloc, calloc, free  # type:ignore
from cython.parallel import prange  # type:ignore
from cython.cimports.openmp import omp_get_thread_num  # type:ignore


# ====== 注意：没有任何缩进，它和 class 是平级的 ======
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def build_csr_topology(
    num_verts: cython.int,
    edge_indices: cython.int[::1],
    out_offsets: cython.int[::1],
    out_indices: cython.int[::1],
    temp_cursor: cython.int[::1],
):
    num_edges: cython.int = edge_indices.shape[0] // 2
    i: cython.int
    v1: cython.int
    v2: cython.int

    for i in range(num_verts + 1):
        out_offsets[i] = 0

    for i in range(num_edges):
        v1 = edge_indices[i * 2]
        v2 = edge_indices[i * 2 + 1]
        out_offsets[v1 + 1] += 1
        out_offsets[v2 + 1] += 1

    for i in range(num_verts):
        out_offsets[i + 1] += out_offsets[i]
        temp_cursor[i] = out_offsets[i]

    idx: cython.int
    for i in range(num_edges):
        v1 = edge_indices[i * 2]
        v2 = edge_indices[i * 2 + 1]

        idx = temp_cursor[v1]
        out_indices[idx] = v2
        temp_cursor[v1] += 1

        idx = temp_cursor[v2]
        out_indices[idx] = v1
        temp_cursor[v2] += 1


@cython.cclass
class CoreBrushEngine:
    """网格笔刷底层引擎类。

    负责处理物理空间中的光线投射 (Raycast) 与拓扑衰减 (Falloff) 计算。

    Attributes:
        vtx_positions2D (cython.float[:, ::1]): 顶点世界坐标池 [N, 3] (只读)。
        tri_indices2D (cython.int[:, ::1]): 三角面顶点索引 [M, 3] (只读)。
        adj_offsets (cython.int[::1]): 邻接表偏移数组，用于 CSR 格式 (只读)。
        adj_indices (cython.int[::1]): 邻接表目标顶点数组，用于 CSR 格式 (只读)。
        active_hit_count (cython.int): 当前笔刷实际命中的顶点总数。
        active_hit_indices (cython.int[::1]): 当前帧命中的顶点全局 ID 数组 (读写)。
        active_hit_weights (cython.float[::1]): 当前帧命中的顶点对应的笔刷衰减强度 (读写)。
        brush_epoch (cython.int): 当前笔刷的世代编号，每次运算自增。
        vertices_epochs (cython.int[::1]): 顶点世代标记数组，用于广度优先搜索去重 (读写)。

    Methods:
        raycast:
            发射多线程射线，寻找模型上距离相机最近的交点、面索引与法线。
        calc_brush_weights:
            根据交点位置，执行空间球体或表面拓扑扫描，计算影响范围内的顶点衰减权重。
    """

    # fmt:off
    # topology
    vtx_positions2D   : cython.float[:, ::1]
    tri_indices2D     : cython.int[:, ::1]
    adj_offsets       : cython.int[::1]
    adj_indices       : cython.int[::1]
    # hit 
    active_hit_count  : cython.int
    active_hit_indices: cython.int[::1]
    active_hit_falloff: cython.float[::1]
    # brush
    brush_epoch       : cython.int
    vertices_epochs   : cython.int[::1]
    # fmt:on

    def __init__(
        self,
        vtx_positions2D: cython.float[:, ::1], 
        tri_indices2D  : cython.int[:, ::1],
        adj_offsets    : cython.int[::1],
        adj_indices    : cython.int[::1],
        vertices_epochs: cython.int[::1],
        hit_indices    : cython.int[::1],
        hit_weights    : cython.float[::1],
    ):  # fmt:skip
        """初始化核心引擎，绑定底层物理内存视图。

        Args:
            vtx_positions2D (cython.float[:, ::1]): 网格顶点坐标矩阵。前端通常传入形状为 [N, 3] 的 numpy.float32 数组。
            tri_indices2D (cython.int[:, ::1]): 网格三角面索引矩阵。前端通常传入形状为 [M, 3] 的 numpy.int32 数组。
            adj_offsets (cython.int[::1]): CSR 邻接表偏移量。前端通常传入 1D numpy.int32 数组。
            adj_indices (cython.int[::1]): CSR 邻接表目标索引。前端通常传入 1D numpy.int32 数组。
            vertices_epochs (cython.int[::1]): 用于记录遍历状态的世代数组。前端通常传入长度为 N 的 1D numpy.int32 数组。
            hit_indices (cython.int[::1]): 预分配的用于存放命中顶点索引的数组。通常为长度 N 的 1D numpy.int32 数组。
            hit_weights (cython.float[::1]): 预分配的用于存放衰减权重的数组。通常为长度 N 的 1D numpy.float32 数组。
        """
        # fmt:off
        self.vtx_positions2D    = vtx_positions2D
        self.tri_indices2D      = tri_indices2D
        self.adj_offsets        = adj_offsets
        self.adj_indices        = adj_indices
        self.vertices_epochs    = vertices_epochs
        self.active_hit_indices = hit_indices
        self.active_hit_falloff = hit_weights

        self.brush_epoch      = 1
        self.active_hit_count = 0
        # fmt:on

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_vertex_positions(self, new_positions: cython.float[:, ::1]):
        """
        热更新顶点坐标内存视图。
        当 Maya 底层网格的物理内存地址发生改变时调用。
        """
        self.vtx_positions2D = new_positions

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.ccall
    def raycast(self, ray_pos: tuple, ray_dir: tuple) -> tuple:
        """多线程射线检测，寻找模型上距离相机最近的交点。

        采用 Möller-Trumbore 交叉算法，并通过 OpenMP 并行加速。

        Args:
            ray_pos (tuple): 射线的世界坐标起点 (x, y, z)。
            ray_dir (tuple): 射线的世界方向向量 (x, y, z)。

        Returns:
            tuple: 包含以下元素的元组:
                - is_hit (cython.bint): 是否击中模型。
                - hit_pos (x,y,z): 击中点的坐标 (hit_x, hit_y, hit_z)。
                - normal (x,y,z): 击中点所在面的法线 (nx, ny, nz)。
                - hit_tri (cython.int): 击中的三角面索引。
                - t (cython.float): 射线起点到击中点的距离。
                - u (cython.float): 重心坐标 U。
                - v (cython.float): 重心坐标 V。
        """
        _points: cython.float[:, ::1] = self.vtx_positions2D
        _tri_indices: cython.int[:, ::1] = self.tri_indices2D

        # 射线属性拆包
        orig_x: cython.float = ray_pos[0]  # 射线起点 X
        orig_y: cython.float = ray_pos[1]  # 射线起点 Y
        orig_z: cython.float = ray_pos[2]  # 射线起点 Z
        dir_x: cython.float = ray_dir[0]  # 射线方向 X
        dir_y: cython.float = ray_dir[1]  # 射线方向 Y
        dir_z: cython.float = ray_dir[2]  # 射线方向 Z

        num_tris: cython.int = _tri_indices.shape[0]  # 模型三角面总数

        # OpenMP 线程本地缓存声明 (防止多线程写入冲突)
        # 注意：底层 C 栈数组分配必须使用字面量以兼容 MSVC 编译器
        thread_closest_t = cython.declare(cython.float[128])  # 各线程最近距离
        thread_hit_tri = cython.declare(cython.int[128])  # 各线程命中的面
        thread_u = cython.declare(cython.float[128])  # 各线程重心坐标 U
        thread_v = cython.declare(cython.float[128])  # 各线程重心坐标 V

        i: cython.int  # 全局循环索引
        tid: cython.int  # 线程 ID

        # 初始化线程缓存
        for i in range(128):
            thread_closest_t[i] = 999999.0
            thread_hit_tri[i] = -1
            thread_u[i] = 0.0
            thread_v[i] = 0.0

        # M-T 算法所需变量声明
        v0_idx: cython.int  # 顶点 0 索引
        v1_idx: cython.int  # 顶点 1 索引
        v2_idx: cython.int  # 顶点 2 索引
        edge1_x: cython.float  # 边 1 向量 X
        edge1_y: cython.float  # 边 1 向量 Y
        edge1_z: cython.float  # 边 1 向量 Z
        edge2_x: cython.float  # 边 2 向量 X
        edge2_y: cython.float  # 边 2 向量 Y
        edge2_z: cython.float  # 边 2 向量 Z

        h_x: cython.float  # 射线方向与边 2 的叉积 X
        h_y: cython.float  # 射线方向与边 2 的叉积 Y
        h_z: cython.float  # 射线方向与边 2 的叉积 Z
        s_x: cython.float  # 起点到顶点 0 的向量 X
        s_y: cython.float  # 起点到顶点 0 的向量 Y
        s_z: cython.float  # 起点到顶点 0 的向量 Z
        q_x: cython.float  # S 向量与边 1 的叉积 X
        q_y: cython.float  # S 向量与边 1 的叉积 Y
        q_z: cython.float  # S 向量与边 1 的叉积 Z

        a: cython.float  # 行列式 (Determinant)
        f: cython.float  # 行列式倒数 (1 / a)
        u: cython.float  # 临时重心坐标 U
        v: cython.float  # 临时重心坐标 V
        t: cython.float  # 临时距离 T

        # 并行计算射线与三角形交点
        for i in prange(num_tris, schedule="guided", nogil=True):
            tid = omp_get_thread_num()
            if tid >= 128:
                tid = 0

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

            # 如果行列式接近 0，射线与三角形平行
            if a > -0.0000001 and a < 0.0000001:
                continue

            f = 1.0 / a
            s_x = orig_x - _points[v0_idx, 0]
            s_y = orig_y - _points[v0_idx, 1]
            s_z = orig_z - _points[v0_idx, 2]

            u = f * (s_x * h_x + s_y * h_y + s_z * h_z)
            # 如果交点不在三角形内部 (重心坐标 U 不在 0~1 之间)
            if u < 0.0 or u > 1.0:
                continue

            q_x = s_y * edge1_z - s_z * edge1_y
            q_y = s_z * edge1_x - s_x * edge1_z
            q_z = s_x * edge1_y - s_y * edge1_x

            v = f * (dir_x * q_x + dir_y * q_y + dir_z * q_z)
            # 如果交点不在三角形内部 (重心坐标 V 越界)
            if v < 0.0 or u + v > 1.0:
                continue

            t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)

            # 每个线程只记录距离摄像机最近的有效交点
            if t > 0.000001 and t < thread_closest_t[tid]:
                thread_closest_t[tid] = t
                thread_hit_tri[tid] = i
                thread_u[tid] = u
                thread_v[tid] = v

        # 数据归约 (Reduction)：在所有线程找出的交点中，找出全局最近的点
        global_closest_t: cython.float = 999999.0  # 全局最小距离
        global_hit_tri: cython.int = -1  # 全局命中的面索引
        global_u: cython.float = 0.0  # 全局重心坐标 U
        global_v: cython.float = 0.0  # 全局重心坐标 V

        for i in range(128):
            if thread_closest_t[i] < global_closest_t:
                global_closest_t = thread_closest_t[i]
                global_hit_tri = thread_hit_tri[i]
                global_u = thread_u[i]
                global_v = thread_v[i]

        if global_hit_tri != -1:
            # 击中了目标，反算表面法线
            v0_idx = _tri_indices[global_hit_tri, 0]
            v1_idx = _tri_indices[global_hit_tri, 1]
            v2_idx = _tri_indices[global_hit_tri, 2]

            edge1_x = _points[v1_idx, 0] - _points[v0_idx, 0]
            edge1_y = _points[v1_idx, 1] - _points[v0_idx, 1]
            edge1_z = _points[v1_idx, 2] - _points[v0_idx, 2]

            edge2_x = _points[v2_idx, 0] - _points[v0_idx, 0]
            edge2_y = _points[v2_idx, 1] - _points[v0_idx, 1]
            edge2_z = _points[v2_idx, 2] - _points[v0_idx, 2]

            raw_nx: cython.float = edge1_y * edge2_z - edge1_z * edge2_y  # 法线叉积 X
            raw_ny: cython.float = edge1_z * edge2_x - edge1_x * edge2_z  # 法线叉积 Y
            raw_nz: cython.float = edge1_x * edge2_y - edge1_y * edge2_x  # 法线叉积 Z

            # 使用 math.sqrt 代替 ** 0.5，获得极限底层性能
            norm_len: cython.float = sqrt(raw_nx * raw_nx + raw_ny * raw_ny + raw_nz * raw_nz)

            nx: cython.float = raw_nx / norm_len if norm_len > 0.000001 else 0.0  # 归一化法线 X
            ny: cython.float = raw_ny / norm_len if norm_len > 0.000001 else 0.0  # 归一化法线 Y
            nz: cython.float = raw_nz / norm_len if norm_len > 0.000001 else 1.0  # 归一化法线 Z

            hit_x: cython.float = orig_x + dir_x * global_closest_t  # 交点 X
            hit_y: cython.float = orig_y + dir_y * global_closest_t  # 交点 Y
            hit_z: cython.float = orig_z + dir_z * global_closest_t  # 交点 Z

            return True, (hit_x, hit_y, hit_z), (nx, ny, nz), global_hit_tri, global_closest_t, global_u, global_v

        # 未击中任何模型
        return False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), -1, 0.0, 0.0, 0.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.ccall
    def calc_brush_weights(
        self,
        hit_position: tuple,
        hit_tri_idx : cython.int,
        radius      : cython.float,
        falloff_mode: cython.int,
        use_surface : cython.bint,
    ) -> tuple:  # fmt:off
        """根据交点计算笔刷影响范围与衰减权重。

        Args:
            hit_position (tuple): 笔刷中心坐标 (x, y, z)。
            hit_tri_idx (cython.int): 击中的三角面索引 (如果模式为表面拓扑则需要)。
            radius (cython.float): 笔刷物理半径。
            falloff_mode (cython.int): 衰减模式 (0:线性, 1:平滑, 2:硬边, 3:穹顶, 4:尖刺)。
            use_surface (cython.bint): 是否使用表面拓扑蔓延 (True: 沿网格蔓延, False: 空间球体)。

        Returns:
            tuple: 包含以下元素的元组:
                - active_hit_count (cython.int): 实际命中的顶点总数。
                - active_hit_indices (cython.int[::1]): 命中顶点索引的内存视图切片。
                - active_hit_weights (cython.float[::1]): 命中顶点衰减权重的内存视图切片。
        """
        # fmt:off
        _vtx_pos = self.vtx_positions2D
        _tris_2d = self.tri_indices2D
        _adj_off = self.adj_offsets
        _adj_idx = self.adj_indices
        _epochs  = self.vertices_epochs
        _out_idx = self.active_hit_indices
        _out_w   = self.active_hit_falloff

        # 世代自增，替代昂贵的 memset 清空数组操作
        self.brush_epoch += 1
        _curr_epoch: cython.int = self.brush_epoch
        hit_count  : cython.int = 0                  # 当前收集的命中点数量
        num_verts  : cython.int = _vtx_pos.shape[0]  # 网格总顶点数

        hit_x: cython.float = hit_position[0]  # 笔刷中心 X
        hit_y: cython.float = hit_position[1]  # 笔刷中心 Y
        hit_z: cython.float = hit_position[2]  # 笔刷中心 Z

        vx       : cython.float                    # 临时顶点坐标 X
        vy       : cython.float                    # 临时顶点坐标 Y
        vz       : cython.float                    # 临时顶点坐标 Z
        dx       : cython.float                    # 距离差 X
        dy       : cython.float                    # 距离差 Y
        dz       : cython.float                    # 距离差 Z
        dist_sq  : cython.float                    # 距离的平方
        weight   : cython.float                    # 算出的最终衰减权重
        t2       : cython.float                    # 距离平方与半径平方的比值
        t        : cython.float                    # 标准化距离 (0~1)
        i        : cython.int                      # 循环索引
        j        : cython.int                      # 内层循环索引

        radius_sq: cython.float = radius * radius  # 笔刷半径平方
        # fmt:on

        # -------------------------------------------------------------
        # 模式 A：体积球体扫描 (Volume Mode)
        # -------------------------------------------------------------
        if not use_surface:
            # 先用 AABB 边界盒进行快速剔除，避免计算全量距离平方
            min_x: cython.float = hit_x - radius
            max_x: cython.float = hit_x + radius
            min_y: cython.float = hit_y - radius
            max_y: cython.float = hit_y + radius
            min_z: cython.float = hit_z - radius
            max_z: cython.float = hit_z + radius

            with cython.nogil:
                for i in range(num_verts):
                    # 包围盒剔除
                    vx = _vtx_pos[i, 0]
                    if vx < min_x or vx > max_x:
                        continue

                    vy = _vtx_pos[i, 1]
                    if vy < min_y or max_y < vy:
                        continue

                    vz = _vtx_pos[i, 2]
                    if vz < min_z or max_z < vz:
                        continue

                    dx = vx - hit_x
                    dy = vy - hit_y
                    dz = vz - hit_z
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
            return (hit_count, self.active_hit_indices, self.active_hit_falloff)

        # -------------------------------------------------------------
        # 模式 B：表面拓扑扫描 (Surface Mode)
        # -------------------------------------------------------------
        if hit_tri_idx < 0:
            self.active_hit_count = 0
            return (0, self.active_hit_indices, self.active_hit_falloff)

        v0: cython.int = _tris_2d[hit_tri_idx, 0]  # 面片顶点 0
        v1: cython.int = _tris_2d[hit_tri_idx, 1]  # 面片顶点 1
        v2: cython.int = _tris_2d[hit_tri_idx, 2]  # 面片顶点 2

        closest_vtx: cython.int = v0  # 存储距离中心最近的种子顶点
        min_dist_sq: cython.float = 9999999.0  # 最小距离平方缓存

        # 比较三点，寻找最近起点
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
            # 初始化 BFS (广度优先搜索) 队列，_out_idx 直接用作队列内存
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

                        dx = _vtx_pos[v_next, 0] - hit_x
                        dy = _vtx_pos[v_next, 1] - hit_y
                        dz = _vtx_pos[v_next, 2] - hit_z
                        dist_sq = dx * dx + dy * dy + dz * dz

                        # 如果连接的顶点依然在笔刷空间球体内，则加入队列
                        if dist_sq <= radius_sq:
                            _out_idx[total_found] = v_next
                            _out_w[total_found] = dist_sq
                            total_found += 1

            # 就地将队列里的距离平方，转换为对应的衰减权重
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
        return (total_found, self.active_hit_indices, self.active_hit_falloff)


# ==============================================================================
# 撤销记录器基类 (BrushUndoRecorder)
# ==============================================================================
@cython.cclass
class BrushUndoRecorder:
    """笔刷业务的通用撤销/重做基类。

    纯粹只负责提供稀疏数据快照功能，能够备份任意多维目标数据（如权重、位置、法线）。
    绝对不包含任何数学计算和业务逻辑，贯彻单一职责原则。

    Attributes:
        core (CoreBrushEngine): 绑定的核心空间引擎实例。
        modified_buffer (cython.float[:, ::1]): 需要被修改的目标数据 2D shape(N, channel_count)。
        channel_count (cython.int): 数据的通道数/列宽 (如 XYZ = 3, 骨骼权重 = influencesCount)。

        modified_vtx_count (cython.int): 当前行程实际修改的顶点总数。
        modified_vtx_bool_buffer (cython.uchar[::1]): 防重录掩码，记录顶点在当前行程中是否已生成过快照 1D shape(N,)。
        modified_vtx_indices_buffer (cython.int[::1]): 当前行程涉及的所有被修改的顶点物理索引池 1D shape(N,)。

        undo_buffer (cython.float[:, ::1]): 撤销内存池，存储顶点被修改前的原始快照。

    Methods:
        begin_stroke:
            在鼠标按下时调用。开启一次新的笔刷行程，重置顶点的防重录标记与计数器。
        end_stroke:
            在鼠标松开时调用。结束当前行程，提取目标数据的最新状态作为 Redo，并打包返回完整的 Undo/Redo 稀疏数据切片。
        _tick_undo_snapshot:
            (内部受保护方法) 在鼠标拖拽计算前调用。接收命中结果，对首次触碰的顶点进行旧数据快照备份。
    """

    core: CoreBrushEngine

    modified_buffer: cython.float[:, ::1]
    channel_count: cython.int
    modified_vtx_bool_buffer: cython.uchar[::1]

    modified_vtx_count: cython.int
    modified_vtx_indices_buffer: cython.int[::1]
    undo_buffer: cython.float[:, ::1]

    def __init__(
        self,
        core: CoreBrushEngine,
        modified_buffer: cython.float[:, ::1],
        modified_vtx_indices_burrer: cython.int[::1],
        modified_vtx_bool_buffer: cython.uchar[::1],
        undo_buffer: cython.float[:, ::1],
    ):
        """初始化撤销系统。

        Args:
            core (CoreBrushEngine): 笔刷引擎实例。作为唯一的数据源。
            modified_buffer (cython.float[:, ::1]): 需要被修改的目标数据矩阵 [N, channel_count]。
            modified_vtx_indices_buffer (cython.int[::1]): 当前行程 (Stroke) 涉及的所有被修改的顶点物理索引池。
            modified_vtx_bool_buffer (cython.uchar[::1]): 防重录掩码，记录顶点在当前行程中是否已生成过快照 [N]。
            undo_buffer (cython.float[:, ::1]): 撤销内存池，存储顶点被修改前的原始快照。
        """
        self.core = core
        self.modified_buffer = modified_buffer
        self.channel_count = modified_buffer.shape[1]
        self.modified_vtx_bool_buffer = modified_vtx_bool_buffer
        self.modified_vtx_indices_buffer = modified_vtx_indices_burrer
        self.undo_buffer = undo_buffer
        self.modified_vtx_count = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def begin_stroke(self) -> tuple:
        """开启绘制 (Stroke)，初始化防重录标记。

        Returns:
            tuple: 包含以下元素的元组:
                - modified_vtx_count (cython.int): 重置后的修改计数器 (恒为 0)。
                - modified_vtx_bool_buffer (cython.uchar[::1]): 清零后的防重录掩码视图。
        """
        # 1. 把视图抽离成纯 C 的局部变量以避免循环内解引用
        _mask = self.modified_vtx_bool_buffer
        verts_count: cython.int = _mask.shape[0]

        self.modified_vtx_count = 0

        i: cython.int
        for i in range(verts_count):
            _mask[i] = 0  # 零开销指针步进清零

        return (self.modified_vtx_count, self.modified_vtx_bool_buffer)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _tick_undo_snapshot(self) -> tuple:
        """在运算前，抓取新命中的顶点进行快照备份。

        Returns:
            tuple: 包含以下元素的元组:
                - modified_vtx_count (cython.int): 当前行程已备份的顶点总数。
                - modified_indices_buffer (cython.int[::1]): 涉及修改的顶点物理索引池视图。
                - modified_vtx_bool_buffer (cython.uchar[::1]): 当前行程的防重录掩码视图。
                - undo_buffer (cython.float[:, ::1]): 修改前的原始快照内存池视图。
        """
        # 1. 提取 Core 的数据 (局部变量缓存)
        _core = self.core
        _hit_idx = _core.active_hit_indices
        _hit_count = _core.active_hit_count

        # 2. 提取 Processor 本身的所有数据与视图到纯 C 局部变量
        _modified_buffer = self.modified_buffer
        _channel_count: cython.int = self.channel_count
        _modified_vtx_bool_buffer = self.modified_vtx_bool_buffer
        _undo_buffer = self.undo_buffer
        _modified_vtx_indices_buffer = self.modified_vtx_indices_buffer
        _modified_vtx_count: cython.int = self.modified_vtx_count

        i: cython.int  # 循环索引
        j: cython.int  # 通道遍历索引
        vtx_idx: cython.int  # 目标顶点索引

        # 3. 纯 C 级别高速循环，彻底摆脱 self 解引用
        for i in range(_hit_count):
            vtx_idx = _hit_idx[i]

            if _modified_vtx_bool_buffer[vtx_idx] == 0:
                _modified_vtx_bool_buffer[vtx_idx] = 1

                for j in range(_channel_count):
                    _undo_buffer[_modified_vtx_count, j] = _modified_buffer[vtx_idx, j]

                _modified_vtx_indices_buffer[_modified_vtx_count] = vtx_idx
                _modified_vtx_count += 1  # 纯寄存器级别的累加

        # 4. 循环结束后，一把梭将最终计数器刷回结构体内存
        self.modified_vtx_count = _modified_vtx_count

        return (self.modified_vtx_count, self.modified_vtx_indices_buffer, self.undo_buffer, self.modified_vtx_bool_buffer)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def end_stroke(self) -> tuple:
        """结束绘制，打包出最新的 Undo & Redo 状态数据（包含双重稀疏压缩）。

        Returns:
            tuple: 若当前行程没有任何顶点被修改，则返回 None, 否则返回包含以下元素的元组:
                - modified_vertex_indices (array.array['i']): 涉及修改的顶点物理索引池（已截断有效长度）。
                - modified_channel_indices (array.array['i']): 实际发生变动的局部列索引。
                - old_sparse_ary (array.array['f']): 极限压缩的 1D 旧快照。
                - new_sparse_ary (array.array['f']): 极限压缩的 1D 新状态。
        """
        if self.modified_vtx_count == 0:
            return None

        # 提取局部 C 变量 (严格遵循原有变量名)
        _modified_vtx_count: cython.int = self.modified_vtx_count
        _channel_count: cython.int = self.channel_count
        _modified = self.modified_buffer
        _indices = self.modified_vtx_indices_buffer
        _undo = self.undo_buffer

        i: cython.int
        j: cython.int
        vtx_idx: cython.int
        diff: cython.float

        # --------------------------------------------------------------------------------------------------------------------------------
        # 用来记录通道是否修改，   channel count 长度的 bool 数组
        channel_is_dirty: cython.p_char = cython.cast(cython.p_char, calloc(_channel_count, cython.sizeof(cython.char)))
        # _modified 数组和 _undo 数组逐元素对比，差异大于 1e-6 则为有变化，把 channel_is_dirty 中对应的数据设置为 1
        modified_channel_count: cython.int = 0
        for i in range(_modified_vtx_count):
            vtx_idx = _indices[i]
            for j in range(_channel_count):
                if channel_is_dirty[j] == 0:
                    diff = _modified[vtx_idx, j] - _undo[i, j]
                    if fabs(diff) > 1e-6:
                        channel_is_dirty[j] = 1
                        modified_channel_count += 1
        # 如果没有骨骼修改，代表这次绘制没有任何效果，直接释放内存，结束函数
        if modified_channel_count == 0:
            free(channel_is_dirty)
            return None

        # 提取并记录脏列 ID，生成 python array
        modified_channel_indices = array.array("i", [0] * modified_channel_count)
        modified_channel_view: cython.int[::1] = modified_channel_indices
        # 迭代查询channel真实的index,设置到python array中
        write_channel_idx: cython.int = 0
        for j in range(_channel_count):
            if channel_is_dirty[j] == 1:
                modified_channel_view[write_channel_idx] = j
                write_channel_idx += 1
        # 释放临时内存
        free(channel_is_dirty)

        # ----------------------------------------------------------------------------------------------------------------------------
        # 截取有效顶点索引 (原本的 buffer 是全量尺寸，这里切出有效长度)
        # 申请 python array
        modified_vertex_indices = array.array("i", [0] * _modified_vtx_count)
        modified_vtx_indices_view: cython.int[::1] = modified_vertex_indices
        # 查询 vtx_index 放入 python array
        for i in range(_modified_vtx_count):
            modified_vtx_indices_view[i] = _indices[i]

        # ------------------------------------------------------------------------------------------------------------------------------
        # 申请 1D 稀疏数组内存
        sparse_size: cython.int = _modified_vtx_count * modified_channel_count

        old_sparse_ary = array.array("f", [0.0] * sparse_size)
        old_sparse_view: cython.float[::1] = old_sparse_ary

        new_sparse_ary = array.array("f", [0.0] * sparse_size)
        new_sparse_view: cython.float[::1] = new_sparse_ary

        # 双向提取channel value (代替原有的全量双层 for 循环)
        write_idx: cython.int = 0
        channel_idx: cython.int = 0
        for i in range(_modified_vtx_count):
            vtx_idx = _indices[i]
            for j in range(modified_channel_count):
                channel_idx = modified_channel_view[j]

                # 提取点对点的 1D 压缩数据，用于返回给外部 Undo 栈
                old_sparse_view[write_idx] = _undo[i, channel_idx]
                new_sparse_view[write_idx] = _modified[vtx_idx, channel_idx]
                write_idx += 1

        return (modified_vertex_indices, modified_channel_indices, old_sparse_ary, new_sparse_ary)


# ==============================================================================
# 通用笔刷
# ==============================================================================
@cython.cclass
class UtilsBrushProcessor(BrushUndoRecorder):
    """通用笔刷数学运算处理类。

    继承 `BrushUndoRecorder`，将底层加减乘除平滑等数学运算进行统一封装。
    能够针对任何 `shape(N, Channels)` 的二维数组进行运算（如顶点色、普通形变缓冲等），
    并将运算影响的范围自动交由父类记录为 Undo/Redo 快照。
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def apply_brush_math(
        self,
        brush_mode: cython.int,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        iterations: cython.int = 1,
    ) -> tuple:
        """执行通用的多通道笔刷数学运算。

        会自动调用 Undo 记录，并原地修改 modified_buffer。

        Args:
            brush_mode (cython.int): `0=Add`, `1=Sub`, `2=Replace`, `3=Multiply`, `4=Smooth`, `5=Sharp`
            values (cython.float[::1]): 对应通道的强度值或目标值。
            channel_indices (cython.int[::1]): 要修改的通道索引列表 (LUT)。
            iterations (cython.int): 仅对 Smooth 模式有效的迭代次数。
        """
        _core = self.core

        # 命中检查
        if _core.active_hit_count == 0:
            return (0, _core.active_hit_indices, self.modified_buffer)

        # 记录 Undo 旧数据
        self._tick_undo_snapshot()

        _iter: cython.int

        for _iter in range(iterations):
            if brush_mode == 0:  # Add
                self._math_add(values, channel_indices)

            elif brush_mode == 1:  # Sub
                self._math_sub(values, channel_indices)

            elif brush_mode == 2:  # Replace
                self._math_replace(values, channel_indices)

            elif brush_mode == 3:  # Multiply
                self._math_multiply(values, channel_indices)

            elif brush_mode == 4:  # Smooth
                self._math_smooth(values, channel_indices)

            elif brush_mode == 5:  # Sharp
                self._math_sharp(values, channel_indices)

            self._post_math_callback()

        return (_core.active_hit_count, _core.active_hit_indices, self.modified_buffer)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_add(self, values: cython.float[::1], channel_indices: cython.int[::1]) -> cython.float[:, ::1]:
        """
        多通道加法运算 (二维数组, 纯数学行列数组视角)。

        采用间接寻址 (LUT)，通过传入 channel_indices 数组，
        完美支持连续通道 (如 RGB [0,1,2]) 与极其稀疏的分散通道 (如骨骼 [15, 115]) 的同频合并计算。

        Args:
            values (cython.float[::1]): 每列(通道)对应的增量数值 (列标量), 亦可以理解为多通道的笔刷强度。
            channel_indices (cython.int[::1]): 需要被修改的真实列号/通道号 (LUT索引表)。

        Returns:
            modified_buffer (cython.float[:, ::1]): 原地修改后的 modified_buffer 视图。
        """
        # fmt:off
        _core = self.core

        # 提取数组映射视图
        _active_rows = _core.active_hit_indices     # 稀疏行索引表
        _row_falloff = _core.active_hit_falloff     # 衰减
        _array       = self.modified_buffer         # 目标二维数组 shape(N, C)

        # 数组迭代边界
        num_rows: cython.int = _core.active_hit_count
        num_cols: cython.int = channel_indices.shape[0]

        # 3. 纯数学循环变量声明
        i  : cython.int      # 行迭代器
        j  : cython.int      # 列迭代器
        row: cython.int      # 目标数组的真实行号
        col: cython.int      # 目标数组的真实列号
        fal: cython.float    # 第 i 行的权重 (Falloff of row i)
        v_j: cython.float    # 第 j 列的强度 (Strength of col j)
        # fmt:on

        for i in range(num_rows):
            fal = _row_falloff[i]
            row = _active_rows[i]

            for j in range(num_cols):
                col = channel_indices[j]
                v_j = values[j]

                _array[row, col] += fal * v_j

        return self.modified_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_sub(self, values: cython.float[::1], channel_indices: cython.int[::1]) -> cython.float[:, ::1]:
        # fmt:off
        _core = self.core

        # 提取数组映射视图
        _active_rows = _core.active_hit_indices     # 稀疏行索引表
        _row_falloff = _core.active_hit_falloff     # 衰减
        _array       = self.modified_buffer         # 目标二维数组 shape(N, C)

        # 数组迭代边界
        num_rows: cython.int = _core.active_hit_count
        num_cols: cython.int = channel_indices.shape[0]

        # 3. 纯数学循环变量声明
        i  : cython.int      # 行迭代器
        j  : cython.int      # 列迭代器
        row: cython.int      # 目标数组的真实行号
        col: cython.int      # 目标数组的真实列号
        fal: cython.float    # 第 i 行的权重 (Falloff of row i)
        v_j: cython.float    # 第 j 列的强度 (Strength of col j)
        # fmt:on

        for i in range(num_rows):
            fal = _row_falloff[i]
            row = _active_rows[i]

            for j in range(num_cols):
                col = channel_indices[j]
                v_j = values[j]

                _array[row, col] += fal * v_j

        return self.modified_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_replace(self, values: cython.float[::1], channel_indices: cython.int[::1]) -> cython.float[:, ::1]:
        """替换运算 (线性逼近插值 Lerp)"""
        # fmt:off
        _core = self.core

        # 提取数组映射视图
        _array       = self.modified_buffer         # 目标二维数组 shape(N, C)
        _active_rows = _core.active_hit_indices     # 稀疏行索引表
        _row_falloff = _core.active_hit_falloff     # 衰减

        # 数组迭代边界
        num_rows: cython.int = _core.active_hit_count
        num_cols: cython.int = channel_indices.shape[0]

        # 3. 纯数学循环变量声明
        i  : cython.int      # 行迭代器
        j  : cython.int      # 列迭代器
        row: cython.int      # 目标数组的真实行号
        col: cython.int      # 目标数组的真实列号
        fal: cython.float    # 第 i 行的权重 (Falloff of row i)
        v_j: cython.float    # 第 j 列的强度 (Strength of col j)
        # fmt:on

        for i in range(num_rows):
            fal = _row_falloff[i]
            row = _active_rows[i]

            for j in range(num_cols):
                col = channel_indices[j]
                v_j = values[j]
                # (目标值 - 当前值) * 行权重
                _array[row, col] += (v_j - _array[row, col]) * fal

        return self.modified_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_multiply(self, values: cython.float[::1], channel_indices: cython.int[::1]) -> cython.float[:, ::1]:
        """替换运算 (线性逼近插值 Lerp)"""
        # fmt:off
        _core = self.core

        # 提取数组映射视图
        _array       = self.modified_buffer         # 目标二维数组 shape(N, C)
        _active_rows = _core.active_hit_indices     # 稀疏行索引表
        _row_falloff = _core.active_hit_falloff     # 衰减

        # 数组迭代边界
        num_rows: cython.int = _core.active_hit_count
        num_cols: cython.int = channel_indices.shape[0]

        # 3. 纯数学循环变量声明
        i  : cython.int      # 行迭代器
        j  : cython.int      # 列迭代器
        row: cython.int      # 目标数组的真实行号
        col: cython.int      # 目标数组的真实列号
        fal: cython.float    # 第 i 行的权重 (Falloff of row i)
        v_j: cython.float    # 第 j 列的强度 (Strength of col j)
        # fmt:on

        for i in range(num_rows):
            fal = _row_falloff[i]
            row = _active_rows[i]

            for j in range(num_cols):
                col = channel_indices[j]
                v_j = values[j]
                _array[row, col] += (_array[row, col] * v_j - _array[row, col]) * fal

        return self.modified_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_smooth(self, values: cython.float[::1], channel_indices: cython.int[::1]) -> cython.float[:, ::1]:
        """拓扑平滑运算 (Topological Laplacian Smooth)"""

        # fmt:off
        _core = self.core

        # 提取数组映射视图
        _array       = self.modified_buffer         # 目标二维数组 shape(N, C)
        _active_rows = _core.active_hit_indices     # 稀疏行索引表
        _row_falloff = _core.active_hit_falloff     # 衰减

        # 提取拓扑数据视图 (CSR)
        _adj_offset  = _core.adj_offsets            # 邻接表偏移
        _adj_idx     = _core.adj_indices            # 邻接表目标顶点

        # 数组迭代边界
        num_rows: cython.int = _core.active_hit_count
        num_cols: cython.int = channel_indices.shape[0]

        # 3. 纯数学循环变量声明
        i    : cython.int    # 行迭代器
        j    : cython.int    # 列迭代器
        n    : cython.int    # 邻居迭代器
        row  : cython.int    # 目标数组的真实行号
        col  : cython.int    # 目标数组的真实列号
        fal  : cython.float  # 第 i 行的权重 (Falloff)
        v_j  : cython.float  # 第 j 列的强度 (Strength)

        edge_start: cython.int
        edge_end  : cython.int
        n_count   : cython.int
        n_idx     : cython.int
        n_sum     : cython.float
        avg       : cython.float
        # fmt:on

        for i in range(num_rows):
            fal = _row_falloff[i]
            row = _active_rows[i]

            # 针对当前行(顶点)，查找它相连的所有邻居边界
            edge_start = _adj_offset[row]  # fmt:skip
            edge_end   = _adj_offset[row + 1]  # fmt:skip
            n_count    = edge_end - edge_start  # fmt:skip

            if n_count > 0:
                for j in range(num_cols):
                    col = channel_indices[j]
                    v_j = values[j]

                    # 遍历邻居，计算当前列(通道)的周围总和
                    n_sum = 0.0
                    for n in range(edge_start, edge_end):
                        n_idx = _adj_idx[n]
                        n_sum += _array[n_idx, col]

                    # 朝着周围邻居的平均值进行平滑插值 (Lerp)
                    avg = n_sum / n_count
                    _array[row, col] += (avg - _array[row, col]) * (v_j * fal)

        return self.modified_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_sharp(self, strength: cython.float, channel_idx: cython.int) -> cython.float[:, ::1]:
        """对比度锐化运算 (Contrast Sharpen)"""
        _core = self.core
        _hit_idx = _core.active_hit_indices
        _hit_w = _core.active_hit_falloff
        _target = self.modified_buffer
        _count: cython.int = _core.active_hit_count

        i: cython.int
        v_idx: cython.int
        mask: cython.float
        val: cython.float

        for i in range(_count):
            mask = _hit_w[i]
            if mask <= 0.0:
                continue
            v_idx = _hit_idx[i]

            val = _target[v_idx, channel_idx]

            # 以 0.5 为分水岭向两极拉伸 (极低开销对比度公式)
            val += (val - 0.5) * (strength * mask)

            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            _target[v_idx, channel_idx] = val

        return self.modified_buffer

    @cython.cfunc
    def _post_math_callback(self) -> cython.void:
        pass


# ==============================================================================
# 权重笔刷类
# ==============================================================================
@cython.cclass
class SkinWeightProcessor(UtilsBrushProcessor):
    """蒙皮权重专属笔刷处理器。

    继承自 `UtilsBrushProcessor`，引入了骨骼锁定的前置判断逻辑，
    并将归一化过程完全剥离暴露为外部按需调用的独立接口。

    Attributes:
        influences_locks_buffer (cython.uchar[::1]): 骨骼锁定标识数组 [N], 1 表示锁定。
    """

    influences_locks_buffer: cython.uchar[::1]

    def __init__(
        self,
        core: CoreBrushEngine,
        modified_buffer: cython.float[:, ::1],
        modified_vtx_indices_buffer: cython.int[::1],
        modified_vtx_bool_buffer: cython.uchar[::1],
        influences_locks_buffer: cython.uchar[::1],
        undo_buffer: cython.float[:, ::1],
    ):
        """初始化权重笔刷，将权重数据与笔刷引擎托管给父类进行通用运算。

        Args:
            core (CoreBrushEngine): 绑定的笔刷引擎实例。
            modified_buffer (cython.float[:, ::1]): 需要被修改的权重矩阵 [N, influencesCount]。
            modified_vtx_indices_buffer (cython.int[::1]): 顶点物理索引池。
            modified_vtx_bool_buffer (cython.uchar[::1]): 防重录掩码。
            influences_locks_buffer (cython.uchar[::1]): 骨骼锁定标识数组 [influencesCount]。
            undo_buffer (cython.float[:, ::1]): 撤销内存池。
        """
        super().__init__(core, modified_buffer, modified_vtx_indices_buffer, modified_vtx_bool_buffer, undo_buffer)
        self.influences_locks_buffer = influences_locks_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def apply_weight(
        self,
        brush_strength: cython.float,
        brush_mode: cython.int,
        influence_idx: cython.int,
        iterations: cython.int = 1,
    ) -> tuple:
        """执行权重运算（拦截锁定，调用父类数学方法）。不触发自动归一化。

        Args:
            brush_strength (cython.float): 用户设定的笔刷强度。
            brush_mode (cython.int): 笔刷模式 (`0=Add`, `1=Sub`, 等)。
            influence_idx (cython.int): 目标骨骼 (Influence) 的层级索引。

        Returns:
            tuple: 包含命中的顶点数、顶点索引以及未归一化的修改后数据。
        """
        # 前置拦截：如果未命中，或者目标骨骼已被锁定，直接拒绝写入并返回
        if (self.core.active_hit_count == 0) or (self.influences_locks_buffer[influence_idx] == 1):
            return (0, self.core.active_hit_indices, self.modified_buffer)

        # 路由给父类执行通用的数学运算与 Undo 记录
        return self.apply_brush_math(brush_strength, brush_mode, influence_idx, iterations)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.ccall
    def normalize_weights(self, target_idx: cython.int) -> cython.float[:, ::1]:
        """[对外暴露接口] 交互式归一化 (Interactive Normalize)。

        可在外部按需调用。保障各个顶点的所有骨骼权重加起来严格等于 1.0。
        在挤压其余骨骼时，会自动跳过被 Lock 锁定的权重影响。

        Args:
            target_idx (cython.int): 正在受核心操作的骨骼索引。
        """
        # 提取至纯 C 变量，消除所有循环内的 self 寻址开销
        _core = self.core
        _locks = self.influences_locks_buffer
        _hit_idx = _core.active_hit_indices
        _w2D = self.modified_buffer
        _count: cython.int = _core.active_hit_count
        num_influences: cython.int = self.channel_count

        i: cython.int  # 被修修改顶点循环
        j: cython.int  # 骨骼影响力遍历循环
        v_idx: cython.int  # 当前操作的顶点物理索引

        locked_sum: cython.float  # 被锁定骨骼占据的权重总和
        unlocked_sum: cython.float  # 未锁定骨骼占据的权重总和
        active_weight: cython.float  # 目标骨骼(正在被刷的骨骼)的当前权重
        remaining_weight: cython.float  # 减去锁定空间和目标空间后，可供平摊的剩余空间
        scale_factor: cython.float  # 强行挤压/拉伸系数

        # 预计算：一共有多少根非目标且未锁定的骨骼可以用来“分摊”
        global_unlocked_count: cython.int = 0
        for j in range(num_influences):
            if j != target_idx and _locks[j] == 0:
                global_unlocked_count += 1

        # 遍历需要校验归一化的所有命中顶点
        for i in range(_count):
            v_idx = _hit_idx[i]
            locked_sum = 0.0
            unlocked_sum = 0.0

            # 摸查当前顶点上所有骨骼的空间占用情况
            for j in range(num_influences):
                if j == target_idx:
                    continue
                if _locks[j] == 1:
                    locked_sum += _w2D[v_idx, j]
                else:
                    unlocked_sum += _w2D[v_idx, j]

            active_weight = _w2D[v_idx, target_idx]

            # 保护机制 1：目标骨骼抢占空间时，绝对不可挤爆已锁定骨骼的地盘
            if active_weight > 1.0 - locked_sum:
                active_weight = 1.0 - locked_sum
                _w2D[v_idx, target_idx] = active_weight

            remaining_weight = 1.0 - locked_sum - active_weight

            # 保护机制 2：如果没有其他骨骼可以拉伸，目标骨骼霸占所有可用空间
            if global_unlocked_count == 0:
                _w2D[v_idx, target_idx] = 1.0 - locked_sum
                continue

            if unlocked_sum > 0.000001:
                # 正常分摊：按其他可用骨骼原有的权重比例，将 remaining_weight 分发出去
                scale_factor = remaining_weight / unlocked_sum

                # 极限性能优化：如果挤压系数在容差内，无需进行大量乘法计算
                if scale_factor > 0.999999 and scale_factor < 1.000001:
                    continue

                for j in range(num_influences):
                    if j != target_idx and _locks[j] == 0:
                        _w2D[v_idx, j] *= scale_factor
            else:
                # 边缘情况修复：如果周边骨骼原有总重趋近于 0，强行执行绝对平均分配
                if remaining_weight > 0.000001:
                    scale_factor = remaining_weight / global_unlocked_count
                    for j in range(num_influences):
                        if j != target_idx and _locks[j] == 0:
                            _w2D[v_idx, j] = scale_factor

        return self.modified_buffer
