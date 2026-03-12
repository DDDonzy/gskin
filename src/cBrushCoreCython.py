"""
==============================================================================
High-Performance Cython Brush Engine for Maya Skin Weights
高性能 Maya 蒙皮权重笔刷与拓扑运算引擎 (纯 C 级底层核心)
==============================================================================

本模块利用 Cython 和 OpenMP 实现了极限性能的网格射线检测、拓扑寻路与权重运算。
采用“数据与逻辑解耦”的架构设计，所有内存由上层 (Python/Maya API) 分配并持有，
本引擎仅接收内存视图 (MemoryView) 进行纯 C 级别的原地突变 (In-place Mutation)。

【核心架构层级】
    1. CoreBrushEngine     : 空间引擎。负责 BVH/AABB 射线检测、M-T 算法交点计算、BFS 拓扑衰减。
    2. BrushUndoRecorder   : 快照引擎。负责极限压缩的双向稀疏 Undo/Redo 数据流。
    3. UtilBrushProcessor  : 数学引擎。提供纯函数式的加减乘除、平滑、拉普拉斯锐化等矩阵运算。
    4. SkinWeightProcessor : 业务层。掌控流水线，强校验骨骼锁定与权重归一化物理法则。

【内存分配规范 (铁律)】
    所有传入 `__init__` 的 `buffer` 参数，必须在外部由 `numpy` 或 Python 原生 `array` 提前分配妥当。
    本引擎内部绝对不会调用任何 `malloc` 或生成新数组，确保 100FPS 的零 GC 延迟。

==============================================================================
使用指南 (Usage Examples)
==============================================================================

场景 A：交互式笔刷 (Interactive Stroke - 绑定在鼠标拖拽事件中)
------------------------------------------------------------------------------
    # 1. 初始化 (在插件加载或工具激活时执行一次)
    core = CoreBrushEngine(vtx_pos_ary, tri_idx_ary, adj_offset, adj_idx, epochs, hit_idx, hit_w)
    processor = SkinWeightProcessor(core, weights_ary, undo_idx, undo_mask, locks_ary, undo_pool)

    # 2. 鼠标按下 (Mouse Press)
    processor.begin_stroke()

    # 3. 鼠标拖拽 (Mouse Drag - 每帧执行)
    is_hit, hit_pos, normal, tri_idx, t, u, v = core.raycast(ray_pos, ray_dir)
    if is_hit:
        # 计算空间拓扑衰减 (结果存入 core 的内存中)
        core.calc_brush_weights(hit_pos, tri_idx, radius=5.0, falloff_mode=1, use_surface=True)

        # 执行权重运算与归一化 (自动读取 core 缓存，并存入 Undo 队列)
        processor.apply_weight(
            brush_mode=0,                  # 0: Add
            values=array.array('f', [0.1]),# 笔刷强度
            channel_indices=target_bones   # 当前绘制的骨骼 ID 数组
        )

    # 4. 鼠标松开 (Mouse Release)
    undo_data = processor.end_stroke()
    if undo_data:
        vtx_idx, ch_idx, old_vals, new_vals = undo_data
        # 将解包后的稀疏数据推入 Maya 原生的 MPxCommand 撤销栈...


场景 B：UI 按钮一键调用 (API Direct Call - 如“一键平滑选中顶点”)
------------------------------------------------------------------------------
    # 直接向引擎灌入指定的顶点索引，无需执行 raycast！
    selected_verts = array.array('i', [15, 102, 334, 1056])

    # 瞬间完成 5 次平滑迭代，并自动执行 5 次归一化，生成完美 Undo 快照
    processor.apply_weight(
        brush_mode=4,                           # 4: Smooth
        values=array.array('f', [1.0]),         # 100% 平滑力度
        channel_indices=target_bones,           # 目标骨骼
        iterations=5,                           # 迭代 5 次以扩大拓扑蔓延
        vertex_indices=selected_verts           # 显式传入目标顶点 (绕过射线引擎)
        # falloff_weights 留空，底层将自动生成全 1.0 的满强度衰减
    )

    # 获取撤销数据并推入栈
    undo_data = processor.end_stroke()

==============================================================================
"""

import array
import cython
from cython.cimports.libc.math import sqrt, fabs  # type:ignore
from cython.cimports.libc.stdlib import calloc, free  # type:ignore
from cython.parallel import prange  # type:ignore
from cython.cimports.openmp import omp_get_thread_num  # type:ignore


@cython.cfunc
def _clamp_float(
    val: cython.float,
    clamp_min: cython.float,
    clamp_max: cython.float,
) -> cython.float:
    """
    极速标量截断函数 (Zero-overhead Inline Helper)。

    由于该函数极其短小且没有任何 Python 对象交互，
    底层 C 编译器会自动将其内联 (Inline) 展开到调用者的循环中，
    转化为无分支的 CMOV 指令，实现零性能损耗。
    """
    if val < clamp_min:
        return clamp_min
    if val > clamp_max:
        return clamp_max
    return val


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
    def calc_brush_falloff(
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
            (内部受保护方法) 在运算前调用。接收命中结果，对首次触碰的顶点进行旧数据快照备份。
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
        modified_vtx_indices_buffer: cython.int[::1],
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
        self.modified_vtx_indices_buffer = modified_vtx_indices_buffer
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
    def _tick_undo_snapshot(self, count: cython.int, vertex_buffer: cython.int[::1]) -> cython.void:
        """在运算前，抓取新命中的顶点进行快照备份（已解耦外部入参）。"""
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
        for i in range(count):
            vtx_idx = vertex_buffer[i]

            if _modified_vtx_bool_buffer[vtx_idx] == 0:
                _modified_vtx_bool_buffer[vtx_idx] = 1

                for j in range(_channel_count):
                    _undo_buffer[_modified_vtx_count, j] = _modified_buffer[vtx_idx, j]

                _modified_vtx_indices_buffer[_modified_vtx_count] = vtx_idx
                _modified_vtx_count += 1  # 纯寄存器级别的累加

        # 4. 循环结束后，一把梭将最终计数器刷回结构体内存
        self.modified_vtx_count = _modified_vtx_count

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
class UtilBrushProcessor(BrushUndoRecorder):
    """通用笔刷数学运算处理类。

    继承 `BrushUndoRecorder`，将底层加减乘除平滑等数学运算进行统一封装。
    能够针对任何 `shape(N, Channels)` 的二维数组进行运算（如顶点色、普通形变缓冲等），
    并将运算影响的范围自动交由父类记录为 Undo/Redo 快照。
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def apply_brush_operation(
        self,
        brush_mode: cython.int,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        clamp_min: cython.float = 0.0,
        clamp_max: cython.float = 1.0,
        iterations: cython.int = 1,
    ) -> tuple:
        """
        为外部普通调用。

        基于底层 `CoreBrushEngine` 鼠标命中缓存 (hit cache)
        命中检测 -> Undo快照 -> 多循环迭代 -> 执行数学计算 -> 写会内存缓冲区。
        适用于 BlendShape 形变、顶点色等不需要严格权重归一化约束的通用二维数组绘制。

        Args:
            brush_mode (cython.int): 笔刷运算模式枚举 (0:Add, 1:Sub, 2:Replace, 3:Multiply, 4:Smooth, 5:SharpVal, 6:SharpGlobal, 7:SharpLocal)。
            values (cython.float[::1]): 对应目标通道的设定值/笔刷强度数组 (1D)。
            channel_indices (cython.int[::1]): 需要被修改的目标列 (通道) 索引数组 (1D)。
            clamp_min (cython.float, optional): 物理下限极值限制，防止数据越界。默认为 0.0。
            clamp_max (cython.float, optional): 物理上限极值限制，防止数据越界。默认为 1.0。
            iterations (cython.int, optional): 数学运算的全局迭代次数。对于 Smooth(平滑) 和 Sharp(锐化) 等拓扑蔓延算法，增加迭代可显著扩大作用范围。默认为 1。
        Returns:
            tuple: 包含本次绘制结果的元组。
                - active_hit_count (cython.int): 本次受影响的有效顶点总数。
                - active_hit_indices (cython.int[::1]): 命中顶点的物理索引数组视图。
                - modified_buffer (cython.float[:, ::1]): 原地修改后的二维数据主矩阵视图。
        """

        _core = self.core
        if _core.active_hit_count == 0:
            return (0, _core.active_hit_indices, self.modified_buffer)

        _vertex_count = _core.active_hit_count
        _vertex_buffer = _core.active_hit_indices
        _falloff_buffer = _core.active_hit_falloff

        self._tick_undo_snapshot(_vertex_count, _vertex_buffer)

        _iter: cython.int
        for _iter in range(iterations):
            self._execute_math_step(
                brush_mode,
                values,
                channel_indices,
                clamp_min,
                clamp_max,
                _vertex_count,
                _vertex_buffer,
                _falloff_buffer,
            )

        return (_core.active_hit_count, _core.active_hit_indices, self.modified_buffer)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cfunc
    def _execute_math_step(
        self,
        brush_mode: cython.int,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        clamp_min: cython.float,
        clamp_max: cython.float,
        vertex_count: cython.int,
        vertex_buffer: cython.int[::1],
        falloff_buffer: cython.float[::1],
    ) -> cython.void:
        """
        Args:
            brush_mode (cython.int): 笔刷运算模式枚举枚举值：
                - 0 = Add (加法)
                - 1 = Sub (减法)
                - 2 = Replace (替换 / Lerp插值)
                - 3 = Multiply (乘法)
                - 4 = Smooth (拓扑平滑)
                - 5 = Sharp Value (纯数值两极极化)
                - 6 = Sharp Topo Global (拓扑锐化，带全局范围截断)
                - 7 = Sharp Topo Local (拓扑局部阶梯化，自带极值侦测，无需全局截断)
            values (cython.float[::1]): 对应目标通道的设定值/笔刷强度数组 (1D)。
            channel_indices (cython.int[::1]): 要修改的目标列(通道/骨骼)索引数组 (1D)。
            clamp_min (cython.float): 物理下限极值 (蒙皮中通常为 0.0)，防止发散运算越界。
            clamp_max (cython.float): 物理上限极值 (蒙皮中通常为 1.0)，防止发散运算越界。
            vertex_count (cython.int): 本次运算涉及的有效顶点总数。
            vertex_buffer (cython.int[::1]): 目标顶点的物理索引数组。
            falloff_buffer (cython.float[::1]): 与目标顶点一一对应的笔刷空间衰减强度 (0.0~1.0)。
        """

        """💥 新增：极其纯粹的单步数学路由模块。没有任何多余废话。"""
        if brush_mode == 0:
            self._math_add(values, channel_indices, clamp_min, clamp_max, vertex_count, vertex_buffer, falloff_buffer)
        elif brush_mode == 1:
            self._math_sub(values, channel_indices, clamp_min, clamp_max, vertex_count, vertex_buffer, falloff_buffer)
        elif brush_mode == 2:
            self._math_replace(values, channel_indices, clamp_min, clamp_max, vertex_count, vertex_buffer, falloff_buffer)
        elif brush_mode == 3:
            self._math_multiply(values, channel_indices, clamp_min, clamp_max, vertex_count, vertex_buffer, falloff_buffer)
        elif brush_mode == 4:
            self._math_smooth(values, channel_indices, clamp_min, clamp_max, vertex_count, vertex_buffer, falloff_buffer)
        elif brush_mode == 5:
            self._math_sharp_value(values, channel_indices, clamp_min, clamp_max, vertex_count, vertex_buffer, falloff_buffer)
        elif brush_mode == 6:
            self._math_sharp_global(values, channel_indices, clamp_min, clamp_max, vertex_count, vertex_buffer, falloff_buffer)
        elif brush_mode == 7:
            self._math_sharp_local(values, channel_indices, clamp_min, clamp_max, vertex_count, vertex_buffer, falloff_buffer)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_add(
        self,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        clamp_min: cython.float,
        clamp_max: cython.float,
        vertex_count: cython.int,
        vertex_buffer: cython.int[::1],
        falloff_buffer: cython.float[::1],
    ) -> cython.void:
        """
        多通道加法运算 (二维数组, 纯数学行列数组视角)。

        采用间接寻址 (LUT)，通过传入 channel_indices 数组，
        完美支持连续通道 (如 RGB [0,1,2]) 与极其稀疏的分散通道 (如骨骼 [15, 115]) 的同频合并计算。

        Args:
            values (cython.float[::1]): 每列(通道)对应的增量数值 (列标量), 亦可以理解为多通道的笔刷强度。
            channel_indices (cython.int[::1]): 需要被修改的真实列号/通道号 (LUT索引表)。
            vertex_count (cython.int): 需要处理的顶点数量。
            vertex_buffer (cython.int[::1]): 目标顶点的物理索引数组。
            falloff_buffer (cython.float[::1]): 对应顶点的衰减强度数组。
        """
        # fmt:off
        _array       = self.modified_buffer         # 目标二维数组 shape(N, C)
        num_cols: cython.int = channel_indices.shape[0]

        # 3. 纯数学循环变量声明
        i  : cython.int      # 行迭代器
        j  : cython.int      # 列迭代器
        row: cython.int      # 目标数组的真实行号
        col: cython.int      # 目标数组的真实列号
        fal: cython.float    # 第 i 行的权重 (Falloff of row i)
        v_j: cython.float    # 第 j 列的强度 (Strength of col j)
        val: cython.float   
        # fmt:on

        for i in range(vertex_count):
            fal = falloff_buffer[i]
            row = vertex_buffer[i]

            for j in range(num_cols):
                col = channel_indices[j]
                v_j = values[j]
                val = _array[row, col] + (fal * v_j)
                _array[row, col] = _clamp_float(val, clamp_min, clamp_max)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_sub(
        self,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        clamp_min: cython.float,
        clamp_max: cython.float,
        vertex_count: cython.int,
        vertex_buffer: cython.int[::1],
        falloff_buffer: cython.float[::1],
    ) -> cython.void:
        # fmt:off
        _array       = self.modified_buffer         # 目标二维数组 shape(N, C)
        num_cols: cython.int = channel_indices.shape[0]

        # 3. 纯数学循环变量声明
        i  : cython.int      # 行迭代器
        j  : cython.int      # 列迭代器
        row: cython.int      # 目标数组的真实行号
        col: cython.int      # 目标数组的真实列号
        fal: cython.float    # 第 i 行的权重 (Falloff of row i)
        v_j: cython.float    # 第 j 列的强度 (Strength of col j)
        val: cython.float    
        # fmt:on

        for i in range(vertex_count):
            fal = falloff_buffer[i]
            row = vertex_buffer[i]

            for j in range(num_cols):
                col = channel_indices[j]
                v_j = values[j]
                val = _array[row, col] - (fal * v_j)
                _array[row, col] = _clamp_float(val, clamp_min, clamp_max)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_replace(
        self,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        clamp_min: cython.float,
        clamp_max: cython.float,
        vertex_count: cython.int,
        vertex_buffer: cython.int[::1],
        falloff_buffer: cython.float[::1],
    ) -> cython.void:
        """替换运算 (线性逼近插值 Lerp)"""
        # fmt:off
        _array       = self.modified_buffer         # 目标二维数组 shape(N, C)
        num_cols: cython.int = channel_indices.shape[0]

        # 3. 纯数学循环变量声明
        i  : cython.int      # 行迭代器
        j  : cython.int      # 列迭代器
        row: cython.int      # 目标数组的真实行号
        col: cython.int      # 目标数组的真实列号
        fal: cython.float    # 第 i 行的权重 (Falloff of row i)
        v_j: cython.float    # 第 j 列的强度 (Strength of col j)
        val: cython.float
        # fmt:on

        for i in range(vertex_count):
            fal = falloff_buffer[i]
            row = vertex_buffer[i]

            for j in range(num_cols):
                col = channel_indices[j]
                v_j = values[j]
                val = _array[row, col] + ((v_j - _array[row, col]) * fal)  # (目标值 - 当前值) * 行权重
                _array[row, col] = _clamp_float(val, clamp_min, clamp_max)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_multiply(
        self,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        clamp_min: cython.float,
        clamp_max: cython.float,
        vertex_count: cython.int,
        vertex_buffer: cython.int[::1],
        falloff_buffer: cython.float[::1],
    ) -> cython.void:
        """替换运算 (线性逼近插值 Lerp)"""
        # fmt:off
        _array       = self.modified_buffer         # 目标二维数组 shape(N, C)
        num_cols: cython.int = channel_indices.shape[0]

        # 3. 纯数学循环变量声明
        i  : cython.int      # 行迭代器
        j  : cython.int      # 列迭代器
        row: cython.int      # 目标数组的真实行号
        col: cython.int      # 目标数组的真实列号
        fal: cython.float    # 第 i 行的权重 (Falloff of row i)
        v_j: cython.float    # 第 j 列的强度 (Strength of col j)
        val: cython.float
        # fmt:on

        for i in range(vertex_count):
            fal = falloff_buffer[i]
            row = vertex_buffer[i]

            for j in range(num_cols):
                col = channel_indices[j]
                v_j = values[j]
                val = _array[row, col] + (_array[row, col] * v_j - _array[row, col]) * fal
                _array[row, col] = _clamp_float(val, clamp_min, clamp_max)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_smooth(
        self,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        clamp_min: cython.float,
        clamp_max: cython.float,
        vertex_count: cython.int,
        vertex_buffer: cython.int[::1],
        falloff_buffer: cython.float[::1],
    ) -> cython.void:
        """拓扑平滑运算 (Topological Laplacian Smooth)"""

        # fmt:off
        _core = self.core
        _array       = self.modified_buffer         # 目标二维数组 shape(N, C)

        # 提取拓扑数据视图 (CSR)
        _adj_offset  = _core.adj_offsets            # 邻接表偏移
        _adj_idx     = _core.adj_indices            # 邻接表目标顶点
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
        val       : cython.float
        # fmt:on

        for i in range(vertex_count):
            fal = falloff_buffer[i]
            row = vertex_buffer[i]

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
                    val = _array[row, col] + (avg - _array[row, col]) * (v_j * fal)
                    _array[row, col] = _clamp_float(val, clamp_min, clamp_max)

    # ---------------------------------------------------------
    # 不看拓扑，只按当前数值以 0.5 为中心向两极推挤
    # ---------------------------------------------------------
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_sharp_value(
        self,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        clamp_min: cython.float,
        clamp_max: cython.float,
        vertex_count: cython.int,
        vertex_buffer: cython.int[::1],
        falloff_buffer: cython.float[::1],
    ) -> cython.void:
        _array = self.modified_buffer
        num_cols: cython.int = channel_indices.shape[0]

        i: cython.int
        j: cython.int
        row: cython.int
        col: cython.int
        fal: cython.float
        v_j: cython.float
        val: cython.float

        for i in range(vertex_count):
            fal = falloff_buffer[i]
            row = vertex_buffer[i]

            for j in range(num_cols):
                col = channel_indices[j]
                v_j = values[j]
                val = _array[row, col]

                val += (val - 0.5) * (v_j * fal) * 2.0

                _array[row, col] = _clamp_float(val, clamp_min, clamp_max)

    # ---------------------------------------------------------
    # Sharp 模式 2：全局限制 (普通蒙皮锐化)
    # ---------------------------------------------------------
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_sharp_global(
        self,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        clamp_min: cython.float,
        clamp_max: cython.float,
        vertex_count: cython.int,
        vertex_buffer: cython.int[::1],
        falloff_buffer: cython.float[::1],
    ) -> cython.void:
        _core = self.core
        _array = self.modified_buffer
        _adj_offset = _core.adj_offsets
        _adj_idx = _core.adj_indices

        num_cols: cython.int = channel_indices.shape[0]

        i: cython.int
        j: cython.int
        n: cython.int
        row: cython.int
        col: cython.int
        fal: cython.float
        v_j: cython.float
        edge_start: cython.int
        edge_end: cython.int
        n_count: cython.int
        n_sum: cython.float
        val: cython.float

        for i in range(vertex_count):
            fal = falloff_buffer[i]
            row = vertex_buffer[i]
            edge_start = _adj_offset[row]
            edge_end = _adj_offset[row + 1]
            n_count = edge_end - edge_start

            if n_count > 0:
                for j in range(num_cols):
                    col = channel_indices[j]
                    v_j = values[j]

                    n_sum = 0.0
                    for n in range(edge_start, edge_end):
                        n_sum += _array[_adj_idx[n], col]

                    val = _array[row, col]
                    val += (val - n_sum / n_count) * (v_j * fal)

                    _array[row, col] = _clamp_float(val, clamp_min, clamp_max)

    # ---------------------------------------------------------
    # Sharp 模式 3：局部限制 (神级阶梯化，带极值侦测)
    # ---------------------------------------------------------
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_sharp_local(
        self,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        clamp_min: cython.float,
        clamp_max: cython.float,
        vertex_count: cython.int,
        vertex_buffer: cython.int[::1],
        falloff_buffer: cython.float[::1],
    ) -> cython.void:
        _core = self.core
        _array = self.modified_buffer
        _adj_offset = _core.adj_offsets
        _adj_idx = _core.adj_indices

        num_cols: cython.int = channel_indices.shape[0]

        i: cython.int
        j: cython.int
        n: cython.int
        row: cython.int
        col: cython.int
        fal: cython.float
        v_j: cython.float
        edge_start: cython.int
        edge_end: cython.int
        n_count: cython.int
        n_idx: cython.int
        n_val: cython.float
        n_sum: cython.float
        val: cython.float
        n_min: cython.float
        n_max: cython.float

        for i in range(vertex_count):
            fal = falloff_buffer[i]
            row = vertex_buffer[i]
            edge_start = _adj_offset[row]
            edge_end = _adj_offset[row + 1]
            n_count = edge_end - edge_start

            if n_count > 0:
                for j in range(num_cols):
                    col = channel_indices[j]
                    v_j = values[j]
                    val = _array[row, col]

                    n_sum = 0.0
                    n_min = val
                    n_max = val

                    # 内层循环：求和与极值打擂台
                    for n in range(edge_start, edge_end):
                        n_idx = _adj_idx[n]
                        n_val = _array[n_idx, col]
                        n_sum += n_val

                        if n_val < n_min:
                            n_min = n_val
                        elif n_val > n_max:
                            n_max = n_val

                    val += (val - n_sum / n_count) * (v_j * fal)

                    _array[row, col] = _clamp_float(val, clamp_min, clamp_max)


# ==============================================================================
# 权重笔刷类
# ==============================================================================
@cython.cclass
class SkinWeightProcessor(UtilBrushProcessor):
    """蒙皮权重专属笔刷处理器。

    继承自 `UtilsBrushProcessor`，引入了骨骼锁定的前置判断逻辑，
    并将归一化过程完全剥离暴露为外部按需调用的独立接口。

    Attributes:
        influences_locks_buffer (cython.uchar[::1]): 骨骼锁定标识数组 [N], 1 表示锁定。
    """

    _single_value_view: cython.float[::1]
    _single_chanel_view: cython.int[::1]

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
            modified_buffer (cython.float[:, ::1]): 需要被修改的权重矩阵 shape(N, influencesCount)。
            modified_vtx_indices_buffer (cython.int[::1]): 顶点物理索引池 shape(N,)
            modified_vtx_bool_buffer (cython.uchar[::1]): 防重录掩码 shape(N,)
            influences_locks_buffer (cython.uchar[::1]): 骨骼锁定标识数组 shape(influencesCount,)。
            undo_buffer (cython.float[:, ::1]): 撤销内存池 shape(N,influencesCount)
        """
        super().__init__(
            core,
            modified_buffer,
            modified_vtx_indices_buffer,
            modified_vtx_bool_buffer,
            undo_buffer,
        )
        self.influences_locks_buffer = influences_locks_buffer
        self._single_value_view = array.array("f", [0.0])
        self._single_chanel_view = array.array("i", [0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def apply_weight_single(
        self,
        brush_mode: cython.int,
        value: cython.float,
        channel_index: cython.int,
        iterations: cython.int = 1,
        vertex_indices: cython.int[::1] = None,
        falloff_weights: cython.float[::1] = None,
    ) -> tuple:
        """
        单骨骼绘制的极速包装器 (Zero-cost Wrapper)。

        作为 `apply_weight` 的标量版本前置接口，专门优化日常绘制中最常见的“单笔刷/单骨骼”操作。
        底层利用 C 语言函数栈 (Stack) 瞬间分配定长为 1 的物理数组并生成切片视图，
        彻底绕过 Python 层面的 array 对象开销与 GC (垃圾回收) 负担，无缝路由至多通道主干 API。

        Args:
            brush_mode (cython.int): 笔刷数学模式 (0:Add, 1:Sub, 2:Replace, 3:Multiply, 4:Smooth, 5:SharpVal, 6:SharpGlobal, 7:SharpLocal)。
            value (cython.float): 目标骨骼对应的笔刷强度/设定值 (标量)。
            channel_index (cython.int): 需要被修改的目标骨骼的物理索引 (标量)。该骨骼在归一化时将自动享有最高优先级。
            iterations (cython.int, optional): 数学运算的全局迭代次数。默认为 1。
            vertex_indices (cython.int[::1], optional): 显式指定要修改的顶点物理索引数组。默认为 None (使用底层射线命中缓存)。
            falloff_weights (cython.float[::1], optional): 与指定的顶点一一对应的衰减权重。默认为 None。

        Returns:
            tuple: 包含本次绘制结果的元组。
                - active_count (cython.int): 实际受影响的顶点总数。
                - active_vtx_array (cython.int[::1]): 实际修改顶点的物理索引数组视图。
                - modified_buffer (cython.float[:, ::1]): 经过完美归一化修缮后的权重主矩阵视图。
        """
        # 在 C 语言的函数栈(Stack)上直接声明两个长度为 1 的纯 C 数组。
        # 栈内存的分配时间是 0 纳秒，函数结束自动销毁，绝对不产生内存碎片。
        _value_buffer = cython.declare(cython.float[1])
        _channel_buffer = cython.declare(cython.int[1])

        self._single_value_view[0] = value
        self._single_chanel_view[0] = channel_index

        # 利用 [:1] 切片操作，将 C 数组瞬间零拷贝转换为 memoryview 视图，透传给主函数
        return self.apply_weight(
            brush_mode=brush_mode,
            values=self._single_value_view,
            channel_indices=self._single_chanel_view,
            iterations=iterations,
            vertex_indices=vertex_indices,
            falloff_weights=falloff_weights,
        )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def apply_weight(
        self,
        brush_mode: cython.int,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        iterations: cython.int = 1,
        vertex_indices: cython.int[::1] = None,
        falloff_weights: cython.float[::1] = None,
    ) -> tuple:
        """
        执行蒙皮权重专属运算 (支持双向数据源驱动)。

        1. 骨骼锁定保护：自动拦截对已锁定骨骼的非法修改。
        2. 极值截断：强制在 0.0 到 1.0 的绝对物理空间内推挤权重。
        3. 自动归一化：在每一次运算迭代后，自动触发基于优先级的权重归一化修缮。

        动态数据源路由机制：
        - 场景 A (交互绘制): 当 `vertex_indices` 为 `None` 时，自动读取底层射线引擎的高速命中缓存。
        - 场景 B (UI 一键调用): 当传入 `vertex_indices` ，直接针对传入的顶点执行极限计算。

        Args:
            brush_mode (cython.int): 笔刷数学模式 (0:Add, 1:Sub, 2:Replace, 3:Multiply, 4:Smooth, 5:SharpVal, 6:SharpGlobal, 7:SharpLocal)。
            values (cython.float[::1]): 对应目标骨骼的设定值/笔刷强度数组 (1D)。
            channel_indices (cython.int[::1]): 要修改的目标骨骼索引数组 (1D)。注意：引擎默认将 `channel_indices[0]` 视为归一化时的优先保护骨骼 (Priority Influence)。
            iterations (cython.int, optional): 数学运算的迭代次数，对 Smooth/Sharp 算法尤为关键。默认为 1。
            vertex_indices (cython.int[::1], optional): 显式指定要修改的顶点物理索引。传入此参数将接管引擎的执行目标。默认为 `None`
            falloff_weights (cython.float[::1], optional): 与指定的顶点一一对应的衰减权重。若传入了 `vertex_indices` 却未传此项，底层将极其贴心地自动生成全 1.0 (100%力度) 的衰减数组。默认为 `None`

        Returns:
            tuple: 包含本次绘制结果的元组。
                - active_count (cython.int): 实际受影响的顶点总数 (若因锁定被拦截则返回 0)。
                - active_vtx_array (cython.int[::1]): 实际修改顶点的物理索引数组视图。
                - modified_buffer (cython.float[:, ::1]): 经过完美归一化修缮后的权重主矩阵视图。
        """
        _vertex_count: cython.int
        _vertex_buffer: cython.int[::1]
        _fal_array: cython.float[::1]

        # 动态确定目标数据源，完美支持外部 UI 一键调用
        if vertex_indices is not None:
            _vertex_buffer = vertex_indices
            _vertex_count = vertex_indices.shape[0]
            if falloff_weights is not None:
                _fal_array = falloff_weights
            else:
                _fal_array = array.array("f", [1.0] * _vertex_count)
        else:
            _vertex_buffer = self.core.active_hit_indices
            _fal_array = self.core.active_hit_falloff
            _vertex_count = self.core.active_hit_count

        if _vertex_count == 0:
            # 没有点
            return (0, _vertex_buffer, self.modified_buffer)
        if self.influences_locks_buffer[channel_indices[0]] == 1:
            # 骨骼被锁定
            return (0, _vertex_buffer, self.modified_buffer)

        # 缓存快照
        self._tick_undo_snapshot(_vertex_count, _vertex_buffer)

        _iter: cython.int
        priority_influence_idx: cython.int = channel_indices[0]
        hit_slice = _vertex_buffer[:_vertex_count]

        for _iter in range(iterations):
            self._execute_math_step(
                brush_mode=brush_mode,
                values=values,
                channel_indices=channel_indices,
                clamp_min=0.0,
                clamp_max=1.0,
                vertex_count=_vertex_count,
                vertex_buffer=_vertex_buffer,
                falloff_buffer=_fal_array,
            )
            self._normalize_weights(hit_slice, priority_influence_idx)

        return (_vertex_count, _vertex_buffer, self.modified_buffer)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.ccall
    def _normalize_weights(
        self,
        vertex_indices: cython.int[::1] = None,  # 目标顶点数组
        priority_influence: cython.int = -1,  # 享有绝对优先权的骨骼
    ) -> cython.float[:, ::1]:
        """权重归一化。

        Args:
            vertex_indices (cython.int[::1], optional): 需要归一化的顶点物理 ID 数组。设置为`None`对所有点进行归一化处理。
            priority_influence (cython.int): 优先保持权重的骨骼。默认为 `-1`,绘制权重的时候,设置为当前骨骼的index,保证这根骨骼具有最大有权进行归一化.
        """
        _locks = self.influences_locks_buffer
        _w2D = self.modified_buffer
        num_influences: cython.int = self.channel_count

        # 确定遍历边界与模式
        _count: cython.int
        use_array: cython.bint = False

        if vertex_indices is not None:
            use_array = True
            _count = vertex_indices.shape[0]
        else:
            use_array = False
            _count = _w2D.shape[0]  # 不传参数时，总数直接取模型顶点总数

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
            # 传了数组就读数组，没传数组就直接用 i 作为顶点物理 ID
            if use_array:
                v_idx = vertex_indices[i]
            else:
                v_idx = i
            locked_sum = 0.0
            unlocked_sum = 0.0
            available_bones_count = 0

            # 查当前顶点上所有骨骼的空间占用情况
            for j in range(num_influences):
                if _locks[j] == 1:
                    locked_sum += _w2D[v_idx, j]
                elif j != priority_influence:
                    unlocked_sum += _w2D[v_idx, j]
                    available_bones_count += 1

            # 计算并保护优先骨骼
            active_weight = 0.0
            if priority_influence != -1:
                active_weight = _w2D[v_idx, priority_influence]
                # 优先骨骼不可抢占已锁定骨骼的权重
                if active_weight > 1.0 - locked_sum:
                    active_weight = 1.0 - locked_sum
                    _w2D[v_idx, priority_influence] = active_weight

            remaining_weight = 1.0 - locked_sum - active_weight

            # 如果没有其他骨骼的权重占用，把未分配的权重给优先骨骼
            if available_bones_count == 0:
                if priority_influence != -1:
                    _w2D[v_idx, priority_influence] = 1.0 - locked_sum
                continue

            # 按其他可用骨骼原有的权重比例，将剩余权重分发出去
            if unlocked_sum > 0.000001:
                scale_factor = remaining_weight / unlocked_sum

                # 极限性能优化：如果挤压系数在容差内，无需进行大量乘法计算
                if scale_factor > 0.999999 and scale_factor < 1.000001:
                    continue

                for j in range(num_influences):
                    if j != priority_influence and _locks[j] == 0:
                        _w2D[v_idx, j] *= scale_factor
            else:
                # 边缘情况修复：如果周边可用骨骼原有总重趋近于 0，强行执行绝对平均分配
                if remaining_weight > 0.000001:
                    scale_factor = remaining_weight / available_bones_count
                    for j in range(num_influences):
                        if j != priority_influence and _locks[j] == 0:
                            _w2D[v_idx, j] = scale_factor

        return self.modified_buffer
