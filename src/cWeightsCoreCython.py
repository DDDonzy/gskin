# cython: language_level=3
import cython
from cython.cimports.libc.stdlib import malloc, free # type:ignore

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def blend_layer_raw_view(
    output_raw_buffer     : cython.float[::1],  
    layer_raw_buffer      : cython.float[::1],  
    layer_mask_raw_buffer : cython.float[::1] = None, 
    layer_alpha           : cython.float      = 1.0, 
    vertex_indices        : cython.int[::1]   = None   
) -> None:
    """
    直接接收 WeightsHandle 的完整连续内存视图，利用 C 指针强转进行原地数据合并。
    100% 满级静态类型标注，无对象创建，绝对零 Python 循环开销。

    Parameters:
        output_raw_buffer : cython.float[::1]
            目标画布 (最终输出) 的完整内存视图。必须包含合法的 Header 和清零后的 Weights。
            计算结果将直接以 C 级别原地覆写到该内存中。
            
        layer_raw_buffer : cython.float[::1]
            当前图层 (输入源) 的完整内存视图。包含 Header 和 Weights。
            
        layer_mask_raw_buffer : cython.float[::1] 或 None
            当前图层的遮罩内存视图。如果该图层没有绘制遮罩，必须传入 `None`。
            
        layer_alpha : cython.float
            当前图层的全局透明度乘数，取值范围通常为 0.0 ~ 1.0。
            
        vertex_indices : cython.int[::1] 或 None
            稀疏更新的顶点 ID 列表。专供画刷的 Undo/Redo 或局部涂抹加速使用。
            如果是全局全量烘焙，必须传入 `None` 以遍历所有顶点。
    """
    
    # ==========================================================
    # 💥 1. 严格 C 变量声明区 (告别一切隐式推导)
    # ==========================================================
    # 1.1 原生指针类型 (Raw Pointers)
    output_float_ptr: cython.p_float
    layer_float_ptr:  cython.p_float
    mask_float_ptr:   cython.p_float
    
    output_int_ptr:   cython.p_int
    layer_int_ptr:    cython.p_int
    mask_int_ptr:     cython.p_int
    
    output_influence_indices: cython.p_int
    layer_influence_indices:  cython.p_int
    layer_to_output_lut:      cython.p_int
    
    output_weights: cython.p_float
    layer_weights:  cython.p_float
    mask_weights:   cython.p_float
    
    # 1.2 整型数据类型 (Integers)
    output_vertex_count:    cython.int
    output_influence_count: cython.int
    layer_influence_count:  cython.int
    mask_influence_count:   cython.int
    
    layer_inf_idx:  cython.int
    output_inf_idx: cython.int
    loop_idx:       cython.int
    vertex_id:      cython.int
    
    output_row_offset: cython.int
    layer_row_offset:  cython.int
    output_col_index:  cython.int
    total_loop_vertices: cython.int
    
    # 1.3 浮点与布尔类型 (Floats & Booleans)
    mask_val: cython.float
    inv_mask: cython.float
    has_mask: cython.bint
    is_sparse: cython.bint

    # ==========================================================
    # 💥 2. 提取原生 C 指针并强转
    # ==========================================================
    output_float_ptr = cython.address(output_raw_buffer[0])
    layer_float_ptr  = cython.address(layer_raw_buffer[0])
    
    output_int_ptr = cython.cast(cython.p_int, output_float_ptr)
    layer_int_ptr  = cython.cast(cython.p_int, layer_float_ptr)

    # ==========================================================
    # 💥 3. 肢解内存，解析 Header 与 Weights
    # ==========================================================
    output_vertex_count      = output_int_ptr[0]
    output_influence_count   = output_int_ptr[1]
    output_influence_indices = cython.address(output_int_ptr[2])
    output_weights           = cython.address(output_float_ptr[2 + output_influence_count])

    layer_influence_count    = layer_int_ptr[1]
    layer_influence_indices  = cython.address(layer_int_ptr[2])
    layer_weights            = cython.address(layer_float_ptr[2 + layer_influence_count])

    # 极速拦截
    if output_influence_count == 0 or layer_influence_count == 0:
        return

    # 安全判断 Mask 是否为 None
    has_mask = layer_mask_raw_buffer is not None
    mask_weights = cython.NULL

    if has_mask:
        # 只有在确认不是 None 时，才安全提取内存地址
        mask_float_ptr = cython.address(layer_mask_raw_buffer[0])
        mask_int_ptr   = cython.cast(cython.p_int, mask_float_ptr)
        mask_influence_count = mask_int_ptr[1]
        mask_weights = cython.address(mask_float_ptr[2 + mask_influence_count])

    # ==========================================================
    # 💥 4. 动态分配 LUT 表并初始化
    # ==========================================================
    layer_to_output_lut = cython.cast(
        cython.p_int, 
        malloc(layer_influence_count * cython.sizeof(cython.int))
    )
    if not layer_to_output_lut:
        raise MemoryError("底层 malloc 分配 LUT 内存失败！")
    
    for layer_inf_idx in range(layer_influence_count):
        layer_to_output_lut[layer_inf_idx] = -1
        for output_inf_idx in range(output_influence_count):
            if output_influence_indices[output_inf_idx] == layer_influence_indices[layer_inf_idx]:
                layer_to_output_lut[layer_inf_idx] = output_inf_idx
                break

    # ==========================================================
    # 💥 5. 极速合并死循环
    # ==========================================================
    is_sparse = vertex_indices is not None
    
    if is_sparse:
        total_loop_vertices = cython.cast(cython.int, vertex_indices.shape[0])
    else:
        total_loop_vertices = output_vertex_count

    for loop_idx in range(total_loop_vertices):
        # 严谨的 if-else 展开，拒绝三元运算符的隐式开销
        if is_sparse:
            vertex_id = vertex_indices[loop_idx]
        else:
            vertex_id = loop_idx

        # 遮罩与透明度混合
        if has_mask:
            mask_val = mask_weights[vertex_id] * layer_alpha
        else:
            mask_val = layer_alpha

        if mask_val <= 0.0:
            continue
        if mask_val > 1.0: 
            mask_val = 1.0

        inv_mask = 1.0 - mask_val
        
        output_row_offset = vertex_id * output_influence_count
        layer_row_offset  = vertex_id * layer_influence_count

        # 阶段 A：衰减底层
        for output_inf_idx in range(output_influence_count):
            output_weights[output_row_offset + output_inf_idx] *= inv_mask

        # 阶段 B：叠加当前层
        for layer_inf_idx in range(layer_influence_count):
            output_col_index = layer_to_output_lut[layer_inf_idx]
            if output_col_index >= 0:
                output_weights[output_row_offset + output_col_index] += layer_weights[layer_row_offset + layer_inf_idx] * mask_val

    # ==========================================================
    # 💥 6. 手动释放 C 内存防泄漏
    # ==========================================================
    free(layer_to_output_lut)