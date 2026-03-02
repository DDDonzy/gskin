import cython
from cython.cimports.libc.math import sqrt  # type:ignore


# fmt:off
# =====================================================================
# 模块 1：纯粹的空间碰撞与衰减引擎 (侦察兵)
# =====================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)  # 进一步关闭变量初始化检查，压榨最后一点性能
@cython.ccall
def compute_radial_weights( center_xyz      : tuple,
                            vertex_positions: cython.float[:, ::1],
                            radius          : cython.float,
                            falloff_mode    : cython.int,
                            out_hit_indices : cython.int[::1],
                            out_hit_weights : cython.float[::1],
) -> cython.int   : 
    i        : cython.int
    hit_count: cython.int = 0
    num_verts: cython.int = vertex_positions.shape[0]

    hit_x    : cython.float = center_xyz[0]
    hit_y    : cython.float = center_xyz[1]
    hit_z    : cython.float = center_xyz[2]

    # 💥 优化 1：预先计算 AABB 包围盒边界 (极速剔除用)
    min_x: cython.float = hit_x - radius
    max_x: cython.float = hit_x + radius
    min_y: cython.float = hit_y - radius
    max_y: cython.float = hit_y + radius
    min_z: cython.float = hit_z - radius
    max_z: cython.float = hit_z + radius

    vx     : cython.float
    vy     : cython.float
    vz     : cython.float
    dx     : cython.float
    dy     : cython.float
    dz     : cython.float
    dist_sq: cython.float
    weight : cython.float
    t      : cython.float

    radius_sq: cython.float = radius * radius

    for i in range(num_verts):
        vx = vertex_positions[i, 0]
        # 💥 极速剔除：如果轴向超出包围盒，直接跳过！
        if vx < min_x or vx > max_x: 
            continue
        vy = vertex_positions[i, 1]
        if vy < min_y or vy > max_y: 
            continue
        vz = vertex_positions[i, 2]
        if vz < min_z or vz > max_z: 
            continue

        # 能活到这里的点，说明已经在这个正方体包围盒里了，命中率极高
        dx = vx - hit_x
        dy = vy - hit_y
        dz = vz - hit_z

        dist_sq = dx * dx + dy * dy + dz * dz

        # 进一步判断是否在精确的球体内部
        if dist_sq <= radius_sq:
            
            # 💥 优化 1：实心笔刷短路计算，直接给 1.0
            if falloff_mode == 2:  # Solid (硬边圆柱体)
                weight = 1.0
            else:
                # 💥 优化 2：预先计算距离平方比 (0.0 到 1.0)，很多高级算法根本不需要开方！
                t2 = dist_sq / radius_sq  
                
                if falloff_mode == 1:    
                    # 🎨 【Airbrush / Smooth - 喷枪】 (极度推荐！)
                    # 算法：(1 - t²)²。完美模拟高斯模糊（钟形曲线）的边缘柔和度。
                    # 💥 性能极高：全都是简单的乘减法，彻底消灭了 sqrt 开方！
                    weight = 1.0 - t2
                    weight = weight * weight
                    
                elif falloff_mode == 0:  
                    # 📐 【Linear - 线性圆锥】
                    # 算法：1 - t。标准的匀速衰减。
                    t = sqrt(t2)
                    weight = 1.0 - t
                    
                elif falloff_mode == 3:  
                    # 🔮 【Dome - 半球形/饱满】
                    # 算法：√(1 - t²)。顶部非常饱满平缓，只在最边缘迅速掉落。适合大面积快速铺色。
                    weight = sqrt(1.0 - t2)
                    
                elif falloff_mode == 4:  
                    # 🗡️ 【Spike - 尖锐/细节】
                    # 算法：(1 - t)³。只有中心极小区域有高强度，四周迅速衰减。适合画细线或毛发权重。
                    t = sqrt(t2)
                    weight = 1.0 - t
                    weight = weight * weight * weight
                
                else:
                    weight = 1.0

            out_hit_indices[hit_count] = i
            out_hit_weights[hit_count] = weight
            hit_count += 1

    return hit_count



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
# fmt:on
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
@cython.ccall
def compute_brush_weights_god_mode(
    center_xyz: tuple,
    vertex_positions: cython.float[:, :],
    tri_indices: cython.int[:, :],
    hit_tri: cython.int,
    adj_offsets: cython.int[:],
    adj_indices: cython.int[:],
    radius: cython.float,
    falloff_mode: cython.int,
    use_surface: cython.bint,
    epoch: cython.int,
    node_epochs: cython.int[:],
    dist: cython.float[:],
    queue: cython.int[:],
    in_queue: cython.char[:],  # 保持签名不变，虽然底层我们已经牛逼到不需要它了
    out_hit_indices: cython.int[:],
    out_hit_weights: cython.float[:],
) -> cython.int:

    i: cython.int
    j: cython.int
    hit_count: cython.int = 0
    num_verts: cython.int = vertex_positions.shape[0]

    hit_x: cython.float = center_xyz[0]
    hit_y: cython.float = center_xyz[1]
    hit_z: cython.float = center_xyz[2]

    radius_sq: cython.float = radius * radius

    vx: cython.float
    vy: cython.float
    vz: cython.float
    dx: cython.float
    dy: cython.float
    dz: cython.float
    dist_sq: cython.float
    weight: cython.float
    t: cython.float
    t2: cython.float

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
                vx = vertex_positions[i, 0]
                if vx < min_x or vx > max_x:
                    continue
                vy = vertex_positions[i, 1]
                if vy < min_y or vy > max_y:
                    continue
                vz = vertex_positions[i, 2]
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
    # 🕸️ 模式 B：圆形表面拓扑衰减 (Topological Mode) - 绝对圆形 + 绝对防穿透
    # ==========================================================================
    if hit_tri < 0:
        return 0

    v0: cython.int = tri_indices[hit_tri, 0]
    v1: cython.int = tri_indices[hit_tri, 1]
    v2: cython.int = tri_indices[hit_tri, 2]

    seed_vertex: cython.int = v0
    min_dist_sq: cython.float = 9999999.0

    dx = vertex_positions[v0, 0] - hit_x
    dy = vertex_positions[v0, 1] - hit_y
    dz = vertex_positions[v0, 2] - hit_z
    dist_sq = dx * dx + dy * dy + dz * dz
    if dist_sq < min_dist_sq:
        min_dist_sq = dist_sq
        seed_vertex = v0

    dx = vertex_positions[v1, 0] - hit_x
    dy = vertex_positions[v1, 1] - hit_y
    dz = vertex_positions[v1, 2] - hit_z
    dist_sq = dx * dx + dy * dy + dz * dz
    if dist_sq < min_dist_sq:
        min_dist_sq = dist_sq
        seed_vertex = v1

    dx = vertex_positions[v2, 0] - hit_x
    dy = vertex_positions[v2, 1] - hit_y
    dz = vertex_positions[v2, 2] - hit_z
    dist_sq = dx * dx + dy * dy + dz * dz
    if dist_sq < min_dist_sq:
        min_dist_sq = dist_sq
        seed_vertex = v2

    with cython.nogil:
        # 💥 种子点贴上最新世代号，并计算它到靶心的纯直线空间距离平方！
        node_epochs[seed_vertex] = epoch
        dist[seed_vertex] = min_dist_sq
        queue[0] = seed_vertex

        head: cython.int = 0
        tail: cython.int = 1  # tail 也是最终摸到的总顶点数

        u: cython.int
        v: cython.int
        edge_start: cython.int
        edge_end: cython.int

        # 极简纯粹的 BFS 洪水泛滥 (删除了 SPFA 松弛，速度起飞)
        while head < tail:
            u = queue[head]
            head += 1

            edge_start = adj_offsets[u]
            edge_end = adj_offsets[u + 1]

            for j in range(edge_start, edge_end):
                v = adj_indices[j]

                # 💥 世代审查：只访问没去过的空房间
                if node_epochs[v] != epoch:
                    node_epochs[v] = epoch

                    # 💥 不再累加边长，直接算直线空间距离的平方！
                    dx = vertex_positions[v, 0] - hit_x
                    dy = vertex_positions[v, 1] - hit_y
                    dz = vertex_positions[v, 2] - hit_z
                    dist_sq = dx * dx + dy * dy + dz * dz

                    # 只要直线距离没超过半径，就把它拉进队列继续泛滥！
                    if dist_sq <= radius_sq:
                        dist[v] = dist_sq  # 存下距离平方，免得下面再算一遍
                        queue[tail] = v
                        tail += 1

        # 3. 收网！所有进入过队列的点，就是我们沿着表面摸到的点
        v_idx: cython.int

        for i in range(tail):
            v_idx = queue[i]
            t2 = dist[v_idx] / radius_sq  # 💥 直接拿存好的距离平方比来算衰减！

            if falloff_mode == 2:
                weight = 1.0  # Solid
            else:
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

            out_hit_indices[hit_count] = v_idx
            out_hit_weights[hit_count] = weight
            hit_count += 1

    return hit_count
