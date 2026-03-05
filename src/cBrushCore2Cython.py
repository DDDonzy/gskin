import cython
from cython.cimports.libc.math import sqrt  # type:ignore
from cython.parallel import prange  # type:ignore
from cython.cimports.openmp import omp_get_thread_num  # type:ignore


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
    active_hit_weights: cython.float[::1]
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
        self.active_hit_weights = hit_weights

        self.brush_epoch      = 1
        self.active_hit_count = 0
        # fmt:on

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
        _out_w   = self.active_hit_weights

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
            return (hit_count, self.active_hit_indices, self.active_hit_weights)

        # -------------------------------------------------------------
        # 模式 B：表面拓扑扫描 (Surface Mode)
        # -------------------------------------------------------------
        if hit_tri_idx < 0:
            self.active_hit_count = 0
            return (0, self.active_hit_indices, self.active_hit_weights)

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
        return (total_found, self.active_hit_indices, self.active_hit_weights)


# 撤销记录器基类 (BaseBrushProcessor)
@cython.cclass
class BaseBrushProcessor:
    """笔刷业务的通用撤销/重做基类。

    提供稀疏数据快照功能，能够备份任意多维目标数据（如权重、位置、法线）。

    Attributes:
        core (CoreBrushEngine): 绑定的核心空间引擎实例。
        modified_buffer (cython.float[:, ::1]): 需要被修改的目标数据矩阵 [N, channel_count]。
        channel_count (cython.int): 数据的通道数/列宽 (如 XYZ = 3, 骨骼权重 = influencesCount)。
        stroke_mask (cython.uchar[::1]): 防重录掩码，记录顶点在当前行程中是否已生成过快照 [N]。
        modified_count (cython.int): 当前行程实际修改的顶点总数。
        modified_indices (cython.int[::1]): 当前行程涉及的所有被修改的顶点物理索引池。
        undo_buffer (cython.float[:, ::1]): 撤销内存池，存储顶点被修改前的原始快照。
        redo_buffer (cython.float[:, ::1]): 重做内存池，存储顶点被修改后的最终快照。

    Methods:
        begin_stroke:
            在鼠标按下时调用。开启一次新的笔刷行程，重置顶点的防重录标记与计数器。
        end_stroke:
            在鼠标松开时调用。结束当前行程，提取目标数据的最新状态作为 Redo，并打包返回完整的 Undo/Redo 稀疏数据切片。
        _record_undo:
            (内部受保护方法) 在鼠标拖拽计算前调用。接收命中结果，对首次触碰的顶点进行旧数据快照备份。
    """

    core: CoreBrushEngine

    modified_buffer: cython.float[:, ::1]
    channel_count: cython.int
    stroke_mask: cython.uchar[::1]

    modified_count: cython.int
    modified_indices: cython.int[::1]
    undo_buffer: cython.float[:, ::1]
    redo_buffer: cython.float[:, ::1]

    def __init__(
        self,
        core: CoreBrushEngine,
        modified_buffer: cython.float[:, ::1],
        modified_indices: cython.int[::1],
        undo_buffer: cython.float[:, ::1],
        redo_buffer: cython.float[:, ::1],
        stroke_mask: cython.uchar[::1],
    ):
        """初始化撤销系统。

        Args:
            core (CoreBrushEngine): 笔刷引擎实例。作为唯一的数据源。
            modified_buffer (cython.float[:, ::1]): 需要被修改的目标数据矩阵 [N, channel_count]。
            modified_indices (cython.int[::1]): 当前行程 (Stroke) 涉及的所有被修改的顶点物理索引池。
            undo_buffer (cython.float[:, ::1]): 撤销内存池，存储顶点被修改前的原始快照。
            redo_buffer (cython.float[:, ::1]): 重做内存池，存储顶点被修改后的最终快照。
            stroke_mask (cython.uchar[::1]): 防重录掩码，记录顶点在当前行程中是否已生成过快照 [N]。
        """
        self.core = core
        self.modified_buffer = modified_buffer
        self.channel_count = modified_buffer.shape[1]
        self.stroke_mask = stroke_mask
        self.modified_indices = modified_indices
        self.undo_buffer = undo_buffer
        self.redo_buffer = redo_buffer
        self.modified_count = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def begin_stroke(self) -> tuple:
        """开启绘制 (Stroke)，初始化防重录标记。

        Returns:
            tuple: 包含以下元素的元组:
                - modified_count (cython.int): 重置后的修改计数器 (恒为 0)。
                - stroke_mask (cython.uchar[::1]): 清零后的防重录掩码视图。
        """
        # 1. 把视图抽离成纯 C 的局部变量以避免循环内解引用
        _mask = self.stroke_mask
        verts_count: cython.int = _mask.shape[0]

        self.modified_count = 0

        i: cython.int
        for i in range(verts_count):
            _mask[i] = 0  # 零开销指针步进清零

        return (self.modified_count, self.stroke_mask)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _record_undo(self) -> tuple:
        """在运算前，抓取新命中的顶点进行快照备份。

        Returns:
            tuple: 包含以下元素的元组:
                - modified_count (cython.int): 当前行程已备份的顶点总数。
                - modified_indices (cython.int[::1]): 涉及修改的顶点物理索引池视图。
                - undo_buffer (cython.float[:, ::1]): 修改前的原始快照内存池视图。
                - stroke_mask (cython.uchar[::1]): 当前行程的防重录掩码视图。
        """
        # 1. 提取 Core 的数据 (局部变量缓存)
        _core = self.core
        _hit_idx = _core.active_hit_indices
        _hit_count = _core.active_hit_count

        # 2. 提取 Processor 本身的所有数据与视图到纯 C 局部变量
        _modified_buffer = self.modified_buffer
        _channel_count: cython.int = self.channel_count
        _stroke_mask = self.stroke_mask
        _undo_buffer = self.undo_buffer
        _modified_indices = self.modified_indices
        _modified_count: cython.int = self.modified_count

        i: cython.int  # 循环索引
        j: cython.int  # 通道遍历索引
        vtx_idx: cython.int  # 目标顶点索引

        # 3. 纯 C 级别高速循环，彻底摆脱 self 解引用
        for i in range(_hit_count):
            vtx_idx = _hit_idx[i]

            if _stroke_mask[vtx_idx] == 0:
                _stroke_mask[vtx_idx] = 1

                for j in range(_channel_count):
                    _undo_buffer[_modified_count, j] = _modified_buffer[vtx_idx, j]

                _modified_indices[_modified_count] = vtx_idx
                _modified_count += 1  # 纯寄存器级别的累加

        # 4. 循环结束后，一把梭将最终计数器刷回结构体内存
        self.modified_count = _modified_count

        return (self.modified_count, self.modified_indices, self.undo_buffer, self.stroke_mask)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def end_stroke(self) -> tuple:
        """结束绘制，打包出最新的 Undo & Redo 状态数据。

        Returns:
            tuple: 若当前行程没有任何顶点被修改，则返回 None。
                否则返回包含以下元素的元组:
                - modified_count (cython.int): 实际修改的顶点总数。
                - modified_indices (cython.int[::1]): 涉及修改的顶点物理索引池视图。
                - undo_buffer (cython.float[:, ::1]): 修改前的原始快照内存池视图。
                - redo_buffer (cython.float[:, ::1]): 修改后的最终状态内存池视图。
        """
        if self.modified_count == 0:
            return None

        # 提取局部 C 变量
        _count: cython.int = self.modified_count
        _channel_count: cython.int = self.channel_count
        _modified = self.modified_buffer
        _indices = self.modified_indices
        _redo = self.redo_buffer

        i: cython.int
        j: cython.int
        vtx_idx: cython.int

        # 针对本次 stroke 修改过的所有顶点，提取最终结果作为 Redo
        for i in range(_count):
            vtx_idx = _indices[i]
            for j in range(_channel_count):
                _redo[i, j] = _modified[vtx_idx, j]

        return (_count, self.modified_indices, self.undo_buffer, self.redo_buffer)


@cython.cclass
class SkinWeightProcessor(BaseBrushProcessor):
    """蒙皮权重专属。

    继承 `BaseBrushProcessor`，负责将笔刷的加减乘除映射至权重数组。

    Attributes:
        influences_locks (cython.uchar[::1]): 骨骼锁定标识数组 [N], 1 表示锁定。
    """

    influences_locks: cython.uchar[::1]

    def __init__(
        self,
        core: CoreBrushEngine,
        modified_buffer: cython.float[:, ::1],
        influences_locks: cython.uchar[::1],
        modified_indices: cython.int[::1],
        undo_buffer: cython.float[:, ::1],
        redo_buffer: cython.float[:, ::1],
        stroke_mask: cython.uchar[::1],
    ):
        """初始化权重笔刷，将权重数据与笔刷引擎托管给基类用于历史记录。

        Args:
            core (CoreBrushEngine): 绑定的笔刷引擎实例。
            modified_buffer (cython.float[:, ::1]): 需要被修改的权重矩阵 [N, influencesCount]。
            influences_locks (cython.uchar[::1]): 骨骼锁定标识数组 [influencesCount]。
            modified_indices (cython.int[::1]): 顶点物理索引池。
            undo_buffer (cython.float[:, ::1]): 撤销内存池。
            redo_buffer (cython.float[:, ::1]): 重做内存池。
            stroke_mask (cython.uchar[::1]): 防重录掩码。
        """
        super().__init__(core, modified_buffer, modified_indices, undo_buffer, redo_buffer, stroke_mask)
        self.influences_locks = influences_locks

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
        """执行完整权重运算生命周期。

        流程: Undo 数据 -> 分发计算模式 -> 归一化限制。

        Args:
            brush_strength (cython.float): 用户设定的笔刷强度。
            brush_mode (cython.int): `0=Add`, `1=Sub`, `2=Replace`, `3=Multiply`, `4=Smooth`, `5=Sharp`
            influence_idx (cython.int): 目标骨骼 (Influence) 的层级索引。

        Returns:
            tuple: 包含以下元素的元组:
                - active_hit_count (cython.int): 本次实际受到笔刷影响并修改的顶点总数。
                - active_hit_indices (cython.int[::1]): 当前帧命中的顶点物理索引视图 (前端用于局部刷新视口)。
                - modified_buffer (cython.float[:, ::1]): 处理完笔刷且经过归一化修复后的最新权重视图。
        """
        _core = self.core

        # 如果未命中，或目标骨骼不可变，直接中断
        if (_core.active_hit_count == 0) or (self.influences_locks[influence_idx] == 1):
            return (0, self.modified_buffer)

        # 1. 前置拦截 (记录 Undo 旧数据，内部会自动提取 self.core)
        self._record_undo()

        # 2. 路由分发数学运算 (原地修改 modified_buffer 内存)
        if brush_mode == 0:
            self._math_add(brush_strength, influence_idx)
        elif brush_mode == 1:
            self._math_sub(brush_strength, influence_idx)
        elif brush_mode == 2:
            self._math_replace(brush_strength, influence_idx)
        elif brush_mode == 3:
            self._math_multiply(brush_strength, influence_idx)
        elif brush_mode == 4:
            self._math_smooth(brush_strength, influence_idx, iterations)
        elif brush_mode == 5:
            self._math_sharp(brush_strength, influence_idx)

        # 3. 如果受多根骨骼影响，强制执行归一化修复
        if self.channel_count > 1:
            self._interactive_normalize2D(influence_idx)

        # 显式抛出修改后的内存视图，数据流向清晰
        return (_core.active_hit_count, _core.active_hit_indices, self.modified_buffer)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_add(self, strength: cython.float, inf_idx: cython.int) -> cython.float[:,::1]:
        """加法笔刷。
        Args:
            strength (cython.float): 笔刷强度。
            inf_idx (cython.int): 目标骨骼索引。

        Returns:
            modified_buffer (cython.float[:,::1]): `modified_buffer`数据视图, return只是为了显示传递。
        """
        _core = self.core
        _hit_idx = _core.active_hit_indices
        _hit_w = _core.active_hit_weights
        _target = self.modified_buffer
        _count: cython.int = _core.active_hit_count

        i: cython.int  # 循环计数器
        v_idx: cython.int  # 获取的物理顶点索引
        mask: cython.float  # 当前顶点的笔刷衰减系数
        val: cython.float  # 临时运算用的权重值

        for i in range(_count):
            mask = _hit_w[i]
            if mask <= 0.0:
                continue
            v_idx = _hit_idx[i]

            val = _target[v_idx, inf_idx] + strength * mask
            if val > 1.0:
                val = 1.0  # 安全防爆顶
            _target[v_idx, inf_idx] = val

        return self.modified_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_sub(self, strength: cython.float, inf_idx: cython.int) -> cython.float[:,::1]:
        """减法笔刷。

        Args:
            strength (cython.float): 笔刷强度。
            inf_idx (cython.int): 目标骨骼索引。

        Returns:
            modified_buffer (cython.float[:,::1]): `modified_buffer`数据视图, return只是为了显示传递。
        """
        _core = self.core
        _hit_idx = _core.active_hit_indices
        _hit_w = _core.active_hit_weights
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

            val = _target[v_idx, inf_idx] - strength * mask
            if val < 0.0:
                val = 0.0  # 安全防击穿
            _target[v_idx, inf_idx] = val

        return self.modified_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_replace(self, strength: cython.float, inf_idx: cython.int) -> cython.float[:,::1]:
        """替换笔刷 (线性逼近插值 Lerp)。

        Args:
            strength (cython.float): 笔刷强度。
            inf_idx (cython.int): 目标骨骼索引。

        Returns:
            modified_buffer (cython.float[:, ::1]): `modified_buffer`数据视图, return只是为了显示传递。
        """
        _core = self.core
        _hit_idx = _core.active_hit_indices
        _hit_w = _core.active_hit_weights
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

            val = _target[v_idx, inf_idx]
            val += (strength - val) * mask
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            _target[v_idx, inf_idx] = val

        return self.modified_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_multiply(self, strength: cython.float, inf_idx: cython.int) -> cython.float[:,::1]:
        """相乘/缩放笔刷。

        Args:
            strength (cython.float): 笔刷强度。
            inf_idx (cython.int): 目标骨骼索引。

        Returns:
            modified_buffer (cython.float[:, ::1]): `modified_buffer`数据视图, return只是为了显示传递。
        """
        _core = self.core
        _hit_idx = _core.active_hit_indices
        _hit_w = _core.active_hit_weights
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

            val = _target[v_idx, inf_idx]
            # 引入 mask 衰减系数的平滑乘法
            val += (val * strength - val) * mask
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            _target[v_idx, inf_idx] = val

        return self.modified_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_smooth(self, strength: cython.float, inf_idx: cython.int, iterations: cython.int) -> cython.float[:,::1]:
        """执行平滑笔刷的权重运算 (Topological Laplacian Smooth)。

        利用核心引擎的 CSR 邻接表，获取当前顶点在物理拓扑上的所有相连邻居，
        计算邻居的平均权重，并使用 Lerp 将当前顶点的权重向该平均值逼近。

        Args:
            strength (cython.float): 平滑强度 (向邻居平均值逼近的速率)。
            inf_idx (cython.int): 目标骨骼索引。
            iterations (cython.int): Smooth 迭代次数

        Returns:
            modified_buffer (cython.float[:, ::1]): `modified_buffer`数据视图, return只是为了显示传递。
        """
        _core = self.core
        _hit_idx = _core.active_hit_indices
        _hit_w = _core.active_hit_weights
        _target = self.modified_buffer
        _count: cython.int = _core.active_hit_count

        _adj_off = _core.adj_offsets
        _adj_idx = _core.adj_indices

        iter: cython.int
        i: cython.int
        j: cython.int
        v_idx: cython.int

        i: cython.int
        j: cython.int
        v_idx: cython.int
        mask: cython.float
        val: cython.float

        edge_start: cython.int
        edge_end: cython.int
        neighbor_count: cython.int
        neighbor_idx: cython.int
        neighbor_sum: cython.float
        avg_weight: cython.float

        for iter in range(iterations):
            for i in range(_count):
                mask = _hit_w[i]
                if mask <= 0.0:
                    continue
                v_idx = _hit_idx[i]

                # 1. 查找当前顶点相连的所有邻居顶点
                edge_start = _adj_off[v_idx]
                edge_end = _adj_off[v_idx + 1]
                neighbor_count = edge_end - edge_start

                # 2. 计算邻居在当前骨骼上的平均权重
                if neighbor_count > 0:
                    neighbor_sum = 0.0
                    for j in range(edge_start, edge_end):
                        neighbor_idx = _adj_idx[j]
                        neighbor_sum += _target[neighbor_idx, inf_idx]

                    avg_weight = neighbor_sum / neighbor_count
                    val = _target[v_idx, inf_idx]

                    # 3. 朝着周围邻居的平均值进行平滑插值 (Lerp)
                    val += (avg_weight - val) * (strength * mask)

                    if val < 0.0:
                        val = 0.0
                    elif val > 1.0:
                        val = 1.0
                    _target[v_idx, inf_idx] = val

        return self.modified_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_sharp(self, strength: cython.float, inf_idx: cython.int) -> cython.float[:, ::1]:
        """执行锐化笔刷的权重运算 (Contrast Sharpen)。

        采用极速对比度拉伸算法，将原本大于 0.5 的权重推向 1.0，
        小于 0.5 的推向 0.0，使权重边缘变得冷硬锐利。

        Args:
            strength (cython.float): 锐化/对比度增加的强度。
            inf_idx (cython.int): 目标骨骼索引。

        Returns:
            modified_buffer (cython.float[:, ::1]): `modified_buffer`数据视图, return只是为了显示传递。
        """
        _core = self.core
        _hit_idx = _core.active_hit_indices
        _hit_w = _core.active_hit_weights
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

            val = _target[v_idx, inf_idx]

            # 以 0.5 为分水岭向两极拉伸 (极低开销对比度公式)
            val += (val - 0.5) * (strength * mask)

            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            _target[v_idx, inf_idx] = val

        return self.modified_buffer

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _interactive_normalize2D(self, target_idx: cython.int) -> cython.float[:, ::1]:
        """归一化 (Interactive Normalize)。

        保障各个顶点的所有骨骼权重加起来严格等于 1.0。
        在挤压其余骨骼时，跳过被 Lock 锁定的权重影响。

        Args:
            target_idx (cython.int): 正在受笔刷影响的目标骨骼索引。

        Returns:
            modified_buffer (cython.float[:, ::1]): `modified_buffer`数据视图, return只是为了显示传递。
        """
        # 提取至纯 C 变量，消除所有循环内的 self 寻址开销
        _core = self.core
        _locks = self.influences_locks
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
