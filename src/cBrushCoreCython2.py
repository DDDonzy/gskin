"""
==============================================================================
High-Performance Cython Brush Engine for Maya Skin Weights
高性能 Maya 蒙皮权重笔刷与拓扑运算引擎 (纯 C 级底层核心)
==============================================================================

本模块利用 Cython 和 OpenMP 实现了极限性能的网格射线检测、拓扑寻路与权重运算。
采用“数据与逻辑解耦”的架构设计，所有内存由上层 (Python/Maya API) 分配并持有。
本引擎仅接收内存视图 (MemoryView) 进行纯 C 级别的原地突变 (In-place Mutation)。

【核心架构层级】 (采用组合模式进行高度解耦)
    1. CoreBrushEngine     : 空间引擎。负责 BVH/AABB 射线检测、M-T 算法交点计算、BFS 拓扑衰减。
    2. BrushUndoRecorder   : 快照引擎。负责极限压缩的双向稀疏 Undo/Redo 数据流。
    3. BrushMathEngine     : 数学引擎。提供纯函数式的加减乘除、平滑、拉普拉斯锐化等矩阵运算。
    4. UtilBrushProcessor  : 通用调度器。利用组合模式统筹空间、快照与数学引擎，处理通用数组。
    5. SkinWeightProcessor : 业务管线调度器。掌控流水线，强校验骨骼锁定与权重归一化物理法则。

【内存分配规范 (铁律)】
    所有传入 `__init__` 的 `buffer` 参数，必须在外部由 `numpy` 或 Python 原生 `array` 提前分配妥当。
    本引擎内部绝对不会调用任何 `malloc` 或生成新数组，确保 100FPS 的零 GC 延迟。

==============================================================================
使用指南 (Usage Examples)
==============================================================================

场景 A 交互式笔刷 (Interactive Stroke - 绑定在鼠标拖拽事件中)
------------------------------------------------------------------------------

    # 1. 初始化 (在插件加载或工具激活时执行一次)
    # core 用于解析模型,raycast,计算笔刷衰减等
    core = CoreBrushEngine(vtx_pos_ary, tri_idx_ary, adj_offset, adj_idx, epochs, hit_idx, hit_w)

    # processor 用于记录基本数据,执行绘制事件 (组合了 core)
    processor = SkinWeightProcessor(core, weights_ary, undo_idx, undo_mask, locks_ary, undo_pool)


    # 2. 鼠标按下 (Mouse Press)
    processor.begin_stroke()

    # 3. 鼠标拖拽 (Mouse Drag - 每帧执行)
    is_hit, hit_pos, normal, tri_idx, t, u, v = core.raycast(ray_pos, ray_dir)
    if is_hit:
        # 计算空间拓扑衰减 (结果存入 core 的内存中)
        core.calc_brush_weights(hit_pos, tri_idx, radius=5.0, falloff_mode=1, use_surface=True)

        # 执行权重运算与归一化 (自动读取 core 缓存 并存入 Undo 队列)
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


场景 B UI 按钮一键调用 (API Direct Call - 如“一键平滑选中顶点”)
------------------------------------------------------------------------------
    # 直接向引擎灌入指定的顶点索引 无需执行 raycast
    selected_verts = array.array('i', [15, 102, 334, 1056])

    # 瞬间完成 5 次平滑迭代 并自动执行 5 次归一化 生成完美 Undo 快照
    processor.apply_weight(
        brush_mode=4,                           # 4: Smooth
        values=array.array('f', [1.0]),         # 100% 平滑力度
        channel_indices=target_bones,           # 目标骨骼
        iterations=5,                           # 迭代 5 次以扩大拓扑蔓延
        vertex_indices=selected_verts           # 显式传入目标顶点 (绕过射线引擎)
        # falloff_weights 留空 底层将自动生成全 1.0 的满强度衰减
    )

    # 获取撤销数据并推入栈
    undo_data = processor.end_stroke()

==============================================================================
"""

import ctypes
import array
import cython
from cython.cimports.libc.math import sqrt, fabs  # type:ignore
from cython.cimports.libc.stdlib import calloc, free  # type:ignore
from cython.cimports.libc.string import memset  # type:ignore
from cython.parallel import prange  # type:ignore
from cython.cimports.openmp import omp_get_thread_num, omp_get_max_threads  # type:ignore


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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def build_csr_topology(
    num_verts: cython.int,
    edge_indices1D: cython.int[::1],  # 这里保持 1D 因为你传进来的是 1D
    out_offsets: cython.int[::1],
    out_indices: cython.int[::1],
    temp_cursor: cython.int[::1],
):
    """构建 V2V (顶点到顶点) CSR 邻接表"""
    num_edges: cython.int = edge_indices1D.shape[0] // 2
    i: cython.int
    v1: cython.int
    v2: cython.int

    for i in range(num_verts + 1):
        out_offsets[i] = 0

    for i in range(num_edges):
        v1 = edge_indices1D[i * 2 + 0]
        v2 = edge_indices1D[i * 2 + 1]
        out_offsets[v1 + 1] += 1
        out_offsets[v2 + 1] += 1

    for i in range(num_verts):
        out_offsets[i + 1] += out_offsets[i]
        temp_cursor[i] = out_offsets[i]

    idx: cython.int
    for i in range(num_edges):
        v1 = edge_indices1D[i * 2 + 0]
        v2 = edge_indices1D[i * 2 + 1]

        idx = temp_cursor[v1]
        out_indices[idx] = v2
        temp_cursor[v1] += 1

        idx = temp_cursor[v2]
        out_indices[idx] = v1
        temp_cursor[v2] += 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def build_v2f_topology(
    num_verts: cython.int,
    tri_indices2D: cython.int[:, ::1],  # 🌟 改回 2D 矩阵读取
    out_offsets: cython.int[::1],
    out_indices: cython.int[::1],
    temp_cursor: cython.int[::1],
):
    """构建 V2F (顶点到面) CSR 邻接表"""
    num_tris: cython.int = tri_indices2D.shape[0]
    i: cython.int
    v0: cython.int
    v1: cython.int
    v2: cython.int

    for i in range(num_verts + 1):
        out_offsets[i] = 0

    for i in range(num_tris):
        v0 = tri_indices2D[i, 0]
        v1 = tri_indices2D[i, 1]
        v2 = tri_indices2D[i, 2]
        out_offsets[v0 + 1] += 1
        out_offsets[v1 + 1] += 1
        out_offsets[v2 + 1] += 1

    for i in range(num_verts):
        out_offsets[i + 1] += out_offsets[i]
        temp_cursor[i] = out_offsets[i]

    idx: cython.int
    for i in range(num_tris):
        v0 = tri_indices2D[i, 0]
        v1 = tri_indices2D[i, 1]
        v2 = tri_indices2D[i, 2]

        idx = temp_cursor[v0]
        out_indices[idx] = i
        temp_cursor[v0] += 1

        idx = temp_cursor[v1]
        out_indices[idx] = i
        temp_cursor[v1] += 1

        idx = temp_cursor[v2]
        out_indices[idx] = i
        temp_cursor[v2] += 1


@cython.cclass
class CoreBrushEngine:
    """
    用于解析模型,raycast,计算笔刷衰减等
    """

    # --- 1. 外部传入的绝对只读物理数据 (恢复 2D 格式) ---
    vtx_positions2D: cython.float[:, ::1]
    tri_indices2D: cython.int[:, ::1]

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

    def __init__(
        self,
        vtx_positions2D: cython.float[:, ::1],
        tri_indices2D: cython.int[:, ::1],
        edge_indices1D: cython.int[::1],
    ):
        """初始化核心引擎 并绑定底层物理内存视图。

        Args:
            vtx_positions2D (cython.float[:, ::1]):
                网格顶点世界坐标矩阵。
                - 形状: [N, 3] 其中 N 为网格的顶点总数。
                - 说明: 每一行存储一个顶点的绝对空间坐标 (X, Y, Z)。
                - 示例: `[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], ...]`
                  代表 0号点在(0,1,0) 1号点在(1,0,0)。

            tri_indices2D (cython.int[:, ::1]):
                网格三角面顶点索引矩阵 (Triangle Vertex Indices)。
                - 形状: [M, 3] 其中 M 为网格的三角面总数。
                - 说明: 它是 3D 拓扑结构的核心 里面的数字并不是坐标 而是指向 `vtx_positions2D` 的“行号 (ID)”。
                  每一行的 3 个整数 代表构成这个三角面的 3 个顶点 ID 通常按逆时针 Winding Order 排列以确定法线朝向 。
                - 示例: 如果某一行是 `[5, 12, 8]` 则意味着这个三角面是由第 5 号、第 12 号、第 8 号顶点连接而成的一张皮。
                  在 M-T 射线算法中 我们会用这 3 个 ID 去 `vtx_positions2D` 里查出真正的空间坐标来进行相交测试。

            edge_indices1D (cython.int[::1]):
                网格边缘的扁平化索引数组 (Flattened Edge Indices)。
                - 形状: [E * 2] 其中 E 为网格的无向边总数。
                - 说明: 因为每条边由 2 个顶点构成 在 1D 数组中它们是“成对 (Pair)”相邻存放的。
                  索引 `[i * 2]` 和 `[i * 2 + 1]` 就是第 i 条边的起点 ID 和终点 ID。
                - 示例: `[4, 7, 12, 9, ...]` 代表系统中有两条边 第一条连接 4号和7号点 第二条连接 12号和9号点。
                  主要用于在初始化时构建 V2V (顶点到顶点) 的 CSR 空间扩散拓扑表。
        """

        self.vtx_positions2D = vtx_positions2D
        self.tri_indices2D = tri_indices2D

        num_verts: cython.int = vtx_positions2D.shape[0]
        num_tris: cython.int = tri_indices2D.shape[0]
        num_edges: cython.int = edge_indices1D.shape[0] // 2

        # 👑 内部自动分配所有 1D 数组 直接丢给 MemoryView 锚定
        self.vertices_epochs = array.array("i", [0]) * num_verts
        self.faces_epochs = array.array("i", [0]) * num_tris
        self.active_hit_indices = array.array("i", [0]) * num_verts
        self.active_hit_falloff = array.array("f", [0.0]) * num_verts

        # 拓扑表内存分配
        self.adj_offsets = array.array("i", [0]) * (num_verts + 1)
        self.adj_indices = array.array("i", [0]) * (num_edges * 2)

        self.v2f_offsets = array.array("i", [0]) * (num_verts + 1)
        self.v2f_indices = array.array("i", [0]) * (num_tris * 3)

        # 构建 CSR 拓扑表
        _temp_cursor = array.array("i", [0]) * num_verts
        build_csr_topology(
            num_verts, edge_indices1D, self.adj_offsets, self.adj_indices, _temp_cursor
        )

        # 重置游标并构建 V2F 拓扑表
        for i in range(num_verts):
            _temp_cursor[i] = 0
        build_v2f_topology(
            num_verts, tri_indices2D, self.v2f_offsets, self.v2f_indices, _temp_cursor
        )

        self.brush_epoch = 1
        self.raycast_epoch = 1
        self.active_hit_count = 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_vertex_positions(self, new_positions2D: cython.float[:, ::1]):
        self.vtx_positions2D = new_positions2D

    def get_hit_indices(self):
        """将内部 C 级命中数组暴露给 Python 层"""
        return self.active_hit_indices.base

    def get_hit_falloff(self):
        """将内部 C 级衰减数组暴露给 Python 层"""
        return self.active_hit_falloff.base

    # endregion

    # region ---------- Raycast
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.ccall
    def raycast(self, ray_pos: tuple, ray_dir: tuple) -> tuple:
        """双轨制射线检测 局部缓存拦截 + 多线程暴力兜底 寻找最近交点。

        采用了业界顶级的 "Coherent Spatial Caching (相干性空间缓存)" 架构。
        - 轨道一 (O(1) 级) 优先测试上一帧笔刷衰减圈内的面片 单线程极速拦截。
        - 轨道二 (O(N) 级) 若缓存未命中 (例如鼠标高速大范围甩动) 唤醒 OpenMP 多线程全量遍历。

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

        # M-T 算法所需核心数学变量声明
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
        global_closest_t: cython.float = 999999.0  # 全局最小距离
        global_hit_tri: cython.int = -1  # 全局命中的面索引
        global_u: cython.float = 0.0  # 全局重心坐标 U
        global_v: cython.float = 0.0  # 全局重心坐标 V

        i: cython.int  # 全局循环索引
        j: cython.int  # V2F 局部循环索引

        cache_hit: cython.bint = False  # 缓存命中标记

        # =====================================================================
        # 🚀 轨道一 V2F 局部空间缓存拦截 (单线程极速 O(1) 级判定)
        # =====================================================================
        if self.active_hit_count > 0:
            self.raycast_epoch += 1  # 更新本帧的射线测试世代
            _curr_r_epoch: cython.int = self.raycast_epoch
            _f_epochs = self.faces_epochs  # 面片测试掩码 防止同一个面被测试多次

            v_idx: cython.int  # 缓存圈内的顶点索引
            edge_start: cython.int  # V2F 拓扑表起始游标
            edge_end: cython.int  # V2F 拓扑表结束游标
            test_tri: cython.int  # 当前需要测试的相邻面

            # 遍历上一帧笔刷衰减圈内的所有命中顶点
            for i in range(self.active_hit_count):
                v_idx = self.active_hit_indices[i]
                edge_start = self.v2f_offsets[v_idx]
                edge_end = self.v2f_offsets[v_idx + 1]

                # 遍历该顶点连接的所有三角面 (通常是 3~6 个)
                for j in range(edge_start, edge_end):
                    test_tri = self.v2f_indices[j]

                    # 🌟 冗余防御 利用世代掩码 保证一帧内同一个面绝对只测试一次
                    if _f_epochs[test_tri] == _curr_r_epoch:
                        continue
                    _f_epochs[test_tri] = _curr_r_epoch

                    # 读取三角面顶点坐标 (2D 读取模式)
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

                    # 射线与三角形平行
                    if a > -0.0000001 and a < 0.0000001:
                        continue

                    # 🌟 背面剔除 (Backface Culling) 防止薄片模型在拖拽时穿透画到背面
                    if a < 0.0:
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

                    # 记录并更新局部最小值
                    if t > 0.000001 and t < global_closest_t:
                        global_closest_t = t
                        global_hit_tri = test_tri
                        global_u = u
                        global_v = v
                        cache_hit = True

        # =====================================================================
        # 🛡️ 轨道二 缓存未命中 启动 OpenMP 多核全量搜索兜底
        # =====================================================================
        if not cache_hit:
            # 🌟 纯 Python 模式获取硬件线程数
            hw_threads: cython.int = omp_get_max_threads()
            active_threads: cython.int = hw_threads - 2

            # 安全防线 保证线程数在合理范围内
            if active_threads < 1:
                active_threads = 1
            elif active_threads > 128:
                active_threads = 128

            # OpenMP 线程本地缓存声明 (使用 cython.declare 分配 C 栈数组)
            thread_closest_t = cython.declare(cython.float[128])
            thread_hit_tri = cython.declare(cython.int[128])
            thread_u = cython.declare(cython.float[128])
            thread_v = cython.declare(cython.float[128])

            # 初始化线程缓存
            for i in range(128):
                thread_closest_t[i] = 999999.0
                thread_hit_tri[i] = -1
                thread_u[i] = 0.0
                thread_v[i] = 0.0

            tid: cython.int  # 线程 ID

            # 🌟 调度策略切换回 guided 用于 Profiler 性能测试对比
            for i in prange(
                num_tris, schedule="guided", num_threads=active_threads, nogil=True
            ):
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

                # 行列式为 0 代表平行 忽略
                if a > -0.0000001 and a < 0.0000001:
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

                # 每个线程只记录该物理核心计算出的一批面片中的最近点
                if t > 0.000001 and t < thread_closest_t[tid]:
                    thread_closest_t[tid] = t
                    thread_hit_tri[tid] = i
                    thread_u[tid] = u
                    thread_v[tid] = v

            # 🏁 数据归约 各线程上报结果 统筹出全局最近点
            for i in range(128):
                if thread_closest_t[i] < global_closest_t:
                    global_closest_t = thread_closest_t[i]
                    global_hit_tri = thread_hit_tri[i]
                    global_u = thread_u[i]
                    global_v = thread_v[i]

        # =====================================================================
        # 🏁 计算交点属性并返回
        # =====================================================================
        if global_hit_tri != -1:
            # 重新获取命中面的顶点信息
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

            # 获取法线模长 (调用 C 标准库 sqrt 极限提速)
            norm_len: cython.float = sqrt(
                raw_nx * raw_nx + raw_ny * raw_ny + raw_nz * raw_nz
            )

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

        # 未击中任何模型
        return False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), -1, 0.0, 0.0, 0.0

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

        # 世代自增 替代昂贵的 memset 清空数组操作
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
        # 模式 A 体积球体扫描 (Volume Mode)
        # -------------------------------------------------------------
        # region ----------- volume mode
        if not use_surface:
            # 先用 AABB 边界盒进行快速剔除 避免计算全量距离平方
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
        # endregion

        # -------------------------------------------------------------
        # 模式 B 表面拓扑扫描 (Surface Mode)
        # -------------------------------------------------------------
        # region ---------- surface mode
        if hit_tri_idx < 0:
            self.active_hit_count = 0
            return (0, self.active_hit_indices, self.active_hit_falloff)

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

                        dx = _vtx_pos[v_next, 0] - hit_x
                        dy = _vtx_pos[v_next, 1] - hit_y
                        dz = _vtx_pos[v_next, 2] - hit_z
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
        return (total_found, self.active_hit_indices, self.active_hit_falloff)
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
        modified_buffer (cython.float[:, ::1]): 需要被修改的目标数据 2D shape(N, channel_count)。
        channel_count (cython.int): 数据的通道数/列宽 (如 XYZ = 3, 骨骼权重 = influencesCount)。

        modified_vtx_count (cython.int): 当前行程实际修改的顶点总数。
        modified_vtx_bool_buffer (cython.uchar[::1]): 防重录掩码 记录顶点在当前行程中是否已生成过快照 1D shape(N,)。
        modified_vtx_indices_buffer (cython.int[::1]): 当前行程涉及的所有被修改的顶点物理索引池 1D shape(N,)。

        undo_buffer (cython.float[:, ::1]): 撤销内存池 存储顶点被修改前的原始快照。
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
            self.undo_buffer = memoryview(c_undo_arr).cast(
                "f", shape=(vtx_count, self.channel_count)
            )
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
        # 为提高性能，将类属性提取到局部变量
        _mod_buf = self.modified_buffer
        _undo_buf = self.undo_buffer
        _mask = self.modified_vtx_bool_buffer
        _idx_pool = self.modified_vtx_indices_buffer
        _channels: cython.int = self.channel_count
        _current_count: cython.int = self.modified_vtx_count

        # 确定是全量处理还是部分处理
        use_all: cython.bint = record_indices is None
        final_count: cython.int
        if use_all:
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

            # 检查是否已记录，避免重复工作
            if _mask[vtx_idx] == 0:
                _mask[vtx_idx] = 1  # 标记为已记录

                # 备份修改前的数据快照
                for j in range(_channels):
                    _undo_buf[_current_count, j] = _mod_buf[vtx_idx, j]

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
        # 1. 分配临时内存，标记被修改过的通道
        channel_is_dirty: cython.p_char = cython.cast(
            cython.p_char, calloc(_channel_count, cython.sizeof(cython.char))
        )

        # 2. 遍历已记录的顶点，比较新旧数据差异
        modified_channel_count: cython.int = 0
        for i in range(_modified_vtx_count):
            vtx_idx = _indices[i]
            for j in range(_channel_count):
                if channel_is_dirty[j] == 0:
                    diff = _modified[vtx_idx, j] - _undo[i, j]
                    if fabs(diff) > 1e-6:
                        channel_is_dirty[j] = 1
                        modified_channel_count += 1

        # 3. 如果没有任何通道被修改，则提前退出
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
                old_sparse_view[write_idx] = _undo[i, channel_idx]
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


@cython.cclass
class BrushMathEngine:
    """通用笔刷数学运算处理引擎 (纯逻辑层)。

    作为组合模式的组件之一，它不再包含任何状态记录或撤销逻辑。
    只负责接收顶点、衰减、目标数组，并执行纯函数式的加减乘除与拓扑平滑等原地内存运算。
    """

    core: CoreBrushEngine
    modified_buffer: cython.float[:, ::1]

    def __init__(self, core: CoreBrushEngine, modified_buffer: cython.float[:, ::1]):
        self.core = core
        self.modified_buffer = modified_buffer

    # region ---------- Execute Math
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
        执行具体的数学步进运算。
        """
        if brush_mode == 0:
            self._math_add(
                values,
                channel_indices,
                clamp_min,
                clamp_max,
                vertex_count,
                vertex_buffer,
                falloff_buffer,
            )
        elif brush_mode == 1:
            self._math_sub(
                values,
                channel_indices,
                clamp_min,
                clamp_max,
                vertex_count,
                vertex_buffer,
                falloff_buffer,
            )
        elif brush_mode == 2:
            self._math_replace(
                values,
                channel_indices,
                clamp_min,
                clamp_max,
                vertex_count,
                vertex_buffer,
                falloff_buffer,
            )
        elif brush_mode == 3:
            self._math_multiply(
                values,
                channel_indices,
                clamp_min,
                clamp_max,
                vertex_count,
                vertex_buffer,
                falloff_buffer,
            )
        elif brush_mode == 4:
            self._math_smooth(
                values,
                channel_indices,
                clamp_min,
                clamp_max,
                vertex_count,
                vertex_buffer,
                falloff_buffer,
            )
        elif brush_mode == 5:
            self._math_sharp(
                values,
                channel_indices,
                clamp_min,
                clamp_max,
                vertex_count,
                vertex_buffer,
                falloff_buffer,
            )

    # endregion

    # region ---------- Add
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
        """多通道加法运算 (二维数组, 纯数学行列数组视角)。"""
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

    # endregion

    # region ---------- sub
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

        i  : cython.int     
        j  : cython.int     
        row: cython.int     
        col: cython.int     
        fal: cython.float   
        v_j: cython.float   
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

    # endregion

    # region ---------- replace
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

        i  : cython.int      
        j  : cython.int      
        row: cython.int      
        col: cython.int      
        fal: cython.float    
        v_j: cython.float    
        val: cython.float
        # fmt:on

        for i in range(vertex_count):
            fal = falloff_buffer[i]
            row = vertex_buffer[i]

            for j in range(num_cols):
                col = channel_indices[j]
                v_j = values[j]
                val = _array[row, col] + (
                    (v_j - _array[row, col]) * fal
                )  # (目标值 - 当前值) * 行权重
                _array[row, col] = _clamp_float(val, clamp_min, clamp_max)

    # endregion

    # region ---------- Mult
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
        """乘法运算"""
        # fmt:off
        _array       = self.modified_buffer         # 目标二维数组 shape(N, C)
        num_cols: cython.int = channel_indices.shape[0]

        i  : cython.int      
        j  : cython.int      
        row: cython.int      
        col: cython.int      
        fal: cython.float    
        v_j: cython.float    
        val: cython.float
        # fmt:on

        for i in range(vertex_count):
            fal = falloff_buffer[i]
            row = vertex_buffer[i]

            for j in range(num_cols):
                col = channel_indices[j]
                v_j = values[j]
                val = (
                    _array[row, col] + (_array[row, col] * v_j - _array[row, col]) * fal
                )
                _array[row, col] = _clamp_float(val, clamp_min, clamp_max)

    # endregion

    # region ---------- Smooth
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

        i    : cython.int    
        j    : cython.int    
        n    : cython.int    
        row  : cython.int    
        col  : cython.int    
        fal  : cython.float  
        v_j  : cython.float  

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

            # 针对当前行(顶点) 查找它相连的所有邻居边界
            edge_start = _adj_offset[row]  # fmt:skip
            edge_end   = _adj_offset[row + 1]  # fmt:skip
            n_count    = edge_end - edge_start  # fmt:skip

            if n_count > 0:
                for j in range(num_cols):
                    col = channel_indices[j]
                    v_j = values[j]

                    # 遍历邻居 计算当前列(通道)的周围总和
                    n_sum = 0.0
                    for n in range(edge_start, edge_end):
                        n_idx = _adj_idx[n]
                        n_sum += _array[n_idx, col]

                    # 朝着周围邻居的平均值进行平滑插值 (Lerp)
                    avg = n_sum / n_count
                    val = _array[row, col] + (avg - _array[row, col]) * (v_j * fal)
                    _array[row, col] = _clamp_float(val, clamp_min, clamp_max)

    # endregion

    # region ---------- Sharp
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    @cython.cfunc
    def _math_sharp(
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

    # endregion

    # region ---------- Get Custom Array (Copy)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def get_custom_array(
        self,
        vertex_indices=None,
        channel_indices=None,
    ) -> array.array:
        """从内存中提取指定数组副本"""
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

    # region ---------- Apply Custom Array (Paste) - Pure Math Only
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
        """将外部数组数据写入内存 (纯写入，不包含撤销。撤销由调度器调用)"""
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
    """通用笔刷调度中心 (基类)。

    采用混合架构：它本身组合了 Core, Recorder, MathEngine 三大独立引擎。
    同时作为基类，供更复杂的业务笔刷 (如蒙皮笔刷) 继承。
    """

    core: CoreBrushEngine
    recorder: BrushUndoRecorder
    math_engine: BrushMathEngine
    modified_buffer: cython.float[:, ::1]

    def __init__(
        self,
        core: CoreBrushEngine,
        modified_buffer: cython.float[:, ::1],
        modified_vtx_indices_buffer: cython.int[::1] = None,
        modified_vtx_bool_buffer: cython.uchar[::1] = None,
        undo_buffer: cython.float[:, ::1] = None,
    ):
        self.core = core
        self.modified_buffer = modified_buffer
        self.recorder = BrushUndoRecorder(
            modified_buffer,
            modified_vtx_indices_buffer,
            modified_vtx_bool_buffer,
            undo_buffer,
        )
        self.math_engine = BrushMathEngine(core, modified_buffer)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def begin_stroke(self) -> tuple:
        return self.recorder.begin_stroke()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def end_stroke(self) -> tuple:
        return self.recorder.end_stroke()

    # region ---------- Apply Brush Operation
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
        命中检测 -> Undo快照调度 -> 多循环迭代调度 -> 写回内存缓冲区。
        """
        _core = self.core
        if _core.active_hit_count == 0:
            return (0, _core.active_hit_indices, self.modified_buffer)

        _vertex_count = _core.active_hit_count
        _vertex_buffer = _core.active_hit_indices
        _falloff_buffer = _core.active_hit_falloff

        # 1. 调度快照引擎记录历史
        self.recorder.record_snapshot(_vertex_buffer)

        # 2. 调度数学引擎执行运算
        _iter: cython.int
        for _iter in range(iterations):
            self.math_engine._execute_math_step(
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

    # endregion

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def get_custom_array(
        self, vertex_indices=None, channel_indices=None
    ) -> array.array:
        return self.math_engine.get_custom_array(vertex_indices, channel_indices)

    @cython.boundscheck(False)
    @cython.wraparound(False)
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
        use_all_v: cython.bint = True
        if (vertex_indices is not None) and (len(vertex_indices) > 0):
            use_all_v = False

        _vtx_view: cython.int[::1] = None if use_all_v else vertex_indices
        _v_count: cython.int = (
            self.modified_buffer.shape[0] if use_all_v else _vtx_view.shape[0]
        )

        if _v_count > 0:
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



@cython.cclass
class SkinWeightProcessor(UtilBrushProcessor):
    """蒙皮权重专属笔刷处理器。

    继承自 `UtilBrushProcessor`，直接复用其底层引用的 Core, Recorder 和 MathEngine。
    以及免费继承了 begin_stroke, end_stroke 等基础方法。
    只专注于添加骨骼锁定和归一化的特有逻辑。
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
        """初始化权重笔刷，通过 super() 调用父类完成底层引擎的组装。"""
        super().__init__(
            core,
            modified_buffer,
            modified_vtx_indices_buffer,
            modified_vtx_bool_buffer,
            undo_buffer,
        )
        self.influences_locks_buffer = influences_locks_buffer
        self.channel_count = modified_buffer.shape[1]

    # endregion

    # region ---------------------- Apply Weights Single
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def apply_weight_single(
        self,
        brush_mode     : cython.int,
        value          : cython.float,
        channel_index  : cython.int,
        vertex_count   : cython.int      = -1,
        vertex_indices                   = None,
        falloff_weights                  = None,
        iterations     : cython.int      = 1,
    ) -> tuple:  # fmt:skip
        """单骨骼绘制的极速包装器"""
        _value_ary = array.array("f", [value])
        _channel_ary = array.array("i", [channel_index])
        return self.apply_weight(
            brush_mode      = brush_mode,
            values          = _value_ary,   
            channel_indices = _channel_ary,  
            vertex_count    = vertex_count,
            vertex_indices  = vertex_indices,
            iterations      = iterations,
            falloff_weights = falloff_weights,
        )  # fmt:skip

    # endregion

    # region ---------------------- Apply Weight
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def apply_weight(
        self,
        brush_mode: cython.int,
        values: cython.float[::1],
        channel_indices: cython.int[::1],
        vertex_count: cython.int = -1,
        vertex_indices=None,
        falloff_weights=None,
        iterations: cython.int = 1,
    ) -> tuple:
        """执行蒙皮权重专属运算"""
        _v_count: cython.int = vertex_count

        is_all_v: cython.bint = False
        if (vertex_indices is None) or (len(vertex_indices) == 0):
            is_all_v = True

        if _v_count < 0:
            if is_all_v:  # noqa: SIM108
                _v_count = self.modified_buffer.shape[0]
            else:
                _v_count = len(vertex_indices)

        if _v_count == 0:
            return (0, vertex_indices, self.modified_buffer)

        _vertex_buffer: cython.int[::1]
        if is_all_v:  # noqa: SIM108
            _vertex_buffer = array.array("i", range(_v_count))
        else:
            _vertex_buffer = vertex_indices

        _fal_array: cython.float[::1]
        if (falloff_weights is not None) and (len(falloff_weights) > 0):  # noqa: SIM108
            _fal_array = falloff_weights
        else:
            _fal_array = array.array("f", [1.0]) * _v_count

        # 1. 调度继承来的快照引擎记录缓存
        self.recorder.record_snapshot(_vertex_buffer)

        _iter: cython.int
        priority_influence_idx: cython.int = channel_indices[0]
        hit_slice = _vertex_buffer[:_v_count]

        # 2. 调度继承来的数学引擎执行运算
        for _iter in range(iterations):
            self.math_engine._execute_math_step(
                brush_mode=brush_mode,
                values=values,
                channel_indices=channel_indices,
                clamp_min=0.0,
                clamp_max=1.0,
                vertex_count=_v_count,
                vertex_buffer=_vertex_buffer,
                falloff_buffer=_fal_array,
            )
            # 执行自身特有的归一化
            self.normalize_weights(hit_slice, priority_influence_idx)

        return (_v_count, vertex_indices, self.modified_buffer)

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
        """权重归一化。"""
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
