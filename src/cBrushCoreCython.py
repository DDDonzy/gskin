import cython
from cython.cimports.libc.math import sqrt  # type:ignore


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.ccall
def brush_math(
    hit_indices   : cython.int[::1],
    hit_weights   : cython.float[::1],
    hit_count     : cython.int,
    brush_strength: cython.float,
    brush_mode    : cython.int,
    target_values : cython.float[:],
):
    i    : cython.int
    v_idx: cython.int
    mask : cython.float
    val  : cython.float
    
    if brush_mode == 0:     # Add (相加)
        for i in range(hit_count):
            mask = hit_weights[i]
            if mask <= 0.0:
                continue
            v_idx = hit_indices[i]
            val = target_values[v_idx] + brush_strength * mask
            if val > 1.0:
                val = 1.0  # 只需要防爆顶
            target_values[v_idx] = val

    elif brush_mode == 1:  # Sub (相减)
        for i in range(hit_count):
            mask = hit_weights[i]
            if mask <= 0.0:
                continue
            v_idx = hit_indices[i]
            val = target_values[v_idx] - brush_strength * mask
            if val < 0.0:
                val = 0.0  # 只需要防击穿
            target_values[v_idx] = val

    elif brush_mode == 2:  # Replace (替换/平滑逼近)
        for i in range(hit_count):
            mask = hit_weights[i]
            if mask <= 0.0:
                continue
            v_idx = hit_indices[i]
            val = target_values[v_idx]
            # 标准的 Lerp (线性插值) 逼近目标强度
            val += (brush_strength - val) * mask
            # Replace 通常不会超出 0~1，如果你的输入安全，甚至可以省掉 clamping
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            target_values[v_idx] = val

    elif brush_mode == 3:  # Multiply (缩放)
        for i in range(hit_count):
            mask = hit_weights[i]
            if mask <= 0.0:
                continue
            v_idx = hit_indices[i]
            val = target_values[v_idx]
            # 💥 修正后的正宗 Multiply 算法 (带有柔和边界过渡)
            val += (val * brush_strength - val) * mask
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            target_values[v_idx] = val



# =====================================================================
# 模块 3：蒙皮专用的后处理模块 
# =====================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)  # 💥 加上这个关闭初始化检查
@cython.ccall
def interactive_normalize2D(
    target_idx   : cython.int,
    element_locks: cython.uchar[::1],      # 💥 加上 ::1
    hit_indices  : cython.int[::1],        # 💥 加上 ::1
    hit_count    : cython.int,
    weights2D    : cython.float[:, ::1],
):
    i               : cython.int
    j               : cython.int
    v_idx           : cython.int
    num_influences  : cython.int = weights2D.shape[1]
    locked_sum      : cython.float
    unlocked_sum    : cython.float
    active_weight   : cython.float
    remaining_weight: cython.float
    scale_factor    : cython.float
    
    # ====================================================
    # 💥 优化 2：把全局不变量提取到循环外部！整个笔刷只算 1 次！
    # ====================================================
    global_unlocked_count: cython.int = 0
    for j in range(num_influences):
        if j != target_idx and element_locks[j] == 0:
            global_unlocked_count += 1

    # 开始遍历受影响的顶点
    for i in range(hit_count):
        v_idx = hit_indices[i]

        locked_sum = 0.0
        unlocked_sum = 0.0

        # 第一遍扫描：收集当前顶点的能量分布
        for j in range(num_influences):
            if j == target_idx:
                continue
            if element_locks[j] == 1:
                locked_sum += weights2D[v_idx, j]
            else:
                unlocked_sum += weights2D[v_idx, j]

        active_weight = weights2D[v_idx, target_idx]

        # 保护机制：目标骨骼不能挤爆被锁定骨骼的空间
        if active_weight > 1.0 - locked_sum:
            active_weight = 1.0 - locked_sum
            weights2D[v_idx, target_idx] = active_weight

        remaining_weight = 1.0 - locked_sum - active_weight

        # 如果没有其他可以吸血/反哺的骨骼，它只能吞掉所有剩余空间
        if global_unlocked_count == 0:
            weights2D[v_idx, target_idx] = 1.0 - locked_sum
            continue

        # 第二遍扫描：归一化能量分配
        if unlocked_sum > 0.000001:
            scale_factor = remaining_weight / unlocked_sum
            
            # 💥 优化 3：Fast-Path 短路跳出！
            # 如果 ratio 接近 1.0，说明无需缩放，直接省掉内层 for 循环！
            if scale_factor > 0.999999 and scale_factor < 1.000001:
                continue
                
            for j in range(num_influences):
                if j != target_idx and element_locks[j] == 0:
                    weights2D[v_idx, j] *= scale_factor
        else:
            if remaining_weight > 0.000001:
                scale_factor = remaining_weight / global_unlocked_count
                for j in range(num_influences):
                    if j != target_idx and element_locks[j] == 0:
                        weights2D[v_idx, j] = scale_factor



# =====================================================================
# 模块 4：二维归一化数据管线
# =====================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.ccall
def skin_weight_brush(
    brush_strength  : cython.float,
    brush_mode      : cython.int,
    influence_idx   : cython.int,
    influences_locks: cython.uchar[::1],
    hit_indices     : cython.int[::1],
    hit_weights     : cython.float[::1],
    hit_count       : cython.int,
    weights2D       : cython.float[:, ::1],

) -> cython.int:
    
    if hit_count == 0 or influences_locks[influence_idx] == 1:
        return 0

    brush_math(
        hit_indices, 
        hit_weights, 
        hit_count, 
        brush_strength, 
        brush_mode, 
        weights2D[:, influence_idx] 
    )

    if weights2D.shape[1] > 1:
        interactive_normalize2D(
            influence_idx,
            influences_locks,
            hit_indices,
            hit_count,
            weights2D,
        )








# ==============================================================================
# 🎨 cBrushCython.py - 纯 Python 语法的终极笔刷核心 (V8 引擎版)
# ==============================================================================
# fmt:off
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.ccall
def calc_brush_weights( vtx_positions  : cython.float[:, :],   # all vertices positions
                        hit_position   : tuple,                # hit center position
                        hit_tri_idx    : cython.int,           # hit tri index
                        # topology
                        tris_indices   : cython.int[:],
                        adj_offsets    : cython.int[:],
                        adj_indices    : cython.int[:],
                        # brush
                        radius         : cython.float,
                        falloff_mode   : cython.int,
                        use_surface    : cython.bint,
                        # pool
                        brush_epoch    : cython.int,
                        vertices_epochs: cython.int[:],
                        # output
                        out_hit_indices: cython.int[:],
                        out_hit_weights: cython.float[:] ) -> cython.int:

    i        : cython.int
    j        : cython.int
    hit_count: cython.int = 0
    num_verts: cython.int = vtx_positions.shape[0]

    hit_x: cython.float = hit_position[0]
    hit_y: cython.float = hit_position[1]
    hit_z: cython.float = hit_position[2]

    vx     : cython.float
    vy     : cython.float
    vz     : cython.float

    dx     : cython.float
    dy     : cython.float
    dz     : cython.float

    radius_sq: cython.float = radius * radius
    dist_sq  : cython.float
    weight   : cython.float
    t        : cython.float
    t2       : cython.float
    # ==========================================================================
    # 🔮 模式 A：体积球体衰减 (Volume Mode) - 空间 AABB 极速剔除
    # ==========================================================================
    if not use_surface:
        min_x: cython.float = hit_x - radius
        max_x: cython.float = hit_x + radius
        min_y: cython.float = hit_y - radius
        max_y: cython.float = hit_y + radius
        min_z: cython.float = hit_z - radius
        max_z: cython.float = hit_z + radius

        with cython.nogil:
            for i in range(num_verts):
                vx = vtx_positions[i, 0]
                if vx < min_x or vx > max_x:
                    continue
                vy = vtx_positions[i, 1]
                if vy < min_y or vy > max_y:
                    continue
                vz = vtx_positions[i, 2]
                if vz < min_z or vz > max_z:
                    continue

                dx = vx - hit_x
                dy = vy - hit_y
                dz = vz - hit_z
                dist_sq = dx * dx + dy * dy + dz * dz

                if dist_sq <= radius_sq:
                    if falloff_mode == 2:
                        weight = 1.0  # Solid
                    else:
                        t2 = dist_sq / radius_sq
                        if falloff_mode == 1:
                            weight = 1.0 - t2
                            weight = weight * weight  # Airbrush
                        elif falloff_mode == 0:
                            weight = 1.0 - sqrt(t2)  # Linear
                        elif falloff_mode == 3:
                            weight = sqrt(1.0 - t2)  # Dome
                        elif falloff_mode == 4:
                            t = sqrt(t2)
                            weight = 1.0 - t
                            weight = weight * weight * weight  # Spike
                        else:
                            weight = 1.0

                    out_hit_indices[hit_count] = i
                    out_hit_weights[hit_count] = weight
                    hit_count += 1

        return hit_count

    # ==========================================================================
    # 🕸️ 模式 B：圆形表面拓扑衰减 (Topological Mode)
    # ==========================================================================
    if hit_tri_idx < 0:
        return 0

    # 🟢 极速寻址：在 C 语言层直接通过 offset 获取三个顶点
    v0: cython.int = tris_indices[hit_tri_idx * 3]
    v1: cython.int = tris_indices[hit_tri_idx * 3 + 1]
    v2: cython.int = tris_indices[hit_tri_idx * 3 + 2]

    closest_vtx: cython.int = v0
    min_dist_sq: cython.float = 9999999.0

    # 找出离靶心最近的一个作为种子点
    dx = vtx_positions[v0, 0] - hit_x
    dy = vtx_positions[v0, 1] - hit_y
    dz = vtx_positions[v0, 2] - hit_z
    dist_sq = dx * dx + dy * dy + dz * dz
    if dist_sq < min_dist_sq:
        min_dist_sq = dist_sq
        closest_vtx = v0

    dx = vtx_positions[v1, 0] - hit_x
    dy = vtx_positions[v1, 1] - hit_y
    dz = vtx_positions[v1, 2] - hit_z
    dist_sq = dx * dx + dy * dy + dz * dz
    if dist_sq < min_dist_sq:
        min_dist_sq = dist_sq
        closest_vtx = v1

    dx = vtx_positions[v2, 0] - hit_x
    dy = vtx_positions[v2, 1] - hit_y
    dz = vtx_positions[v2, 2] - hit_z
    dist_sq = dx * dx + dy * dy + dz * dz
    if dist_sq < min_dist_sq:
        min_dist_sq = dist_sq
        closest_vtx = v2

    with cython.nogil:
        vertices_epochs[closest_vtx] = brush_epoch

        out_hit_indices[0] = closest_vtx
        out_hit_weights[0] = min_dist_sq

        current_idx: cython.int = 0
        total_found: cython.int = 1

        v1: cython.int
        v2: cython.int
        edge_start: cython.int
        edge_end: cython.int

        while current_idx < total_found:
            v1 = out_hit_indices[current_idx]
            current_idx += 1

            edge_start = adj_offsets[v1]
            edge_end = adj_offsets[v1 + 1]

            for j in range(edge_start, edge_end):
                v2 = adj_indices[j]

                if vertices_epochs[v2] != brush_epoch:
                    vertices_epochs[v2] = brush_epoch

                    dx = vtx_positions[v2, 0] - hit_x
                    dy = vtx_positions[v2, 1] - hit_y
                    dz = vtx_positions[v2, 2] - hit_z
                    dist_sq = dx * dx + dy * dy + dz * dz

                    if dist_sq <= radius_sq:
                        out_hit_indices[total_found] = v2
                        out_hit_weights[total_found] = dist_sq
                        total_found += 1

        # 将就地存储的距离平方，原地转换为权重
        for i in range(total_found):
            dist_sq = out_hit_weights[i]
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

            out_hit_weights[i] = weight

    return total_found
# fmt:on
