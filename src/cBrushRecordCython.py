import ctypes
import cython
from cython.cimports.libc.math import fabs  # type:ignore
from cython.cimports.libc.stdlib import calloc, free  # type:ignore
from cython.cimports.libc.string import memset  # type:ignore


# ==============================================================================
# 撤销记录器基类 (BrushUndoRecorder)
# ==============================================================================
@cython.cclass
class BrushUndoRecorder:
    """
    笔刷业务的通用撤销/重做基类。

    纯粹只负责提供稀疏数据快照功能 能够备份任意多维目标数据 如权重、位置、法线 。
    绝对不包含任何数学计算和业务逻辑 贯彻单一职责原则。

    Attributes:
        modified_buffer (cython.float[:, ::1]): 需要被修改的目标数据 2D shape(N, channel_count)。
        channel_count (cython.int): 数据的通道数/列宽 (如 XYZ = 3, 骨骼权重 = influencesCount)。

        modified_vtx_count (cython.int): 当前行程实际修改的顶点总数。
        modified_vtx_bool_buffer (cython.uchar[::1]): 防重录掩码 记录顶点在当前行程中是否已生成过快照 1D shape(N,)。
        modified_vtx_indices_buffer (cython.int[::1]): 当前行程涉及的所有被修改的顶点物理索引池 1D shape(N,)。

        undo_buffer (cython.float[:, ::1]): 撤销内存池 存储顶点被修改前的原始快照。

    Methods:
        begin_stroke:
            在鼠标按下时调用。开启一次新的笔刷行程 重置顶点的防重录标记与计数器。
        end_stroke:
            在鼠标松开时调用。结束当前行程 提取目标数据的最新状态作为 Redo 并打包返回完整的 Undo/Redo 稀疏数据切片。
        record_undo_snapshot:
            在运算前调用。接收命中结果 对首次触碰的顶点进行旧数据快照备份。
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
        """初始化撤销系统。

        Args:
            modified_buffer: 需要被修改的目标数据矩阵 [N, channel_count]。必填。
            modified_vtx_indices_buffer: 顶点物理索引池。若为 None 则自动分配。
            modified_vtx_bool_buffer: 防重录掩码。若为 None 则自动分配。
            undo_buffer: 撤销内存池。若为 None 则自动按 [N, channel_count] 分配 2D 连续内存。
        """
        self.modified_buffer = modified_buffer
        self.channel_count = modified_buffer.shape[1]
        self.modified_vtx_count = 0

        vtx_count: cython.int = modified_buffer.shape[0]

        # 1. 处理防重录掩码 (cython.uchar 对应 ctypes.c_uint8)
        if modified_vtx_bool_buffer is None:
            c_bool_arr = (ctypes.c_uint8 * vtx_count)()  # 申请 C 级别的无符号单字节数组 并包装为 memoryview
            self.modified_vtx_bool_buffer = memoryview(c_bool_arr)
        else:
            self.modified_vtx_bool_buffer = modified_vtx_bool_buffer

        # 2. 处理被修改顶点索引池 (cython.int 对应 ctypes.c_int32)
        if modified_vtx_indices_buffer is None:
            c_indices_arr = (ctypes.c_int32 * vtx_count)()  # 申请 C 级别的 32 位整型数组
            self.modified_vtx_indices_buffer = memoryview(c_indices_arr)
        else:
            self.modified_vtx_indices_buffer = modified_vtx_indices_buffer

        # 3. 处理 2D 撤销内存池 (cython.float 对应 ctypes.c_float)
        if undo_buffer is None:
            flat_size = vtx_count * self.channel_count  # 申请 1D 的连续单精度浮点 C 数组
            c_undo_arr = (ctypes.c_float * flat_size)()

            self.undo_buffer = memoryview(c_undo_arr).cast("f", shape=(vtx_count, self.channel_count))  # 将 1D ctypes 数组转换为 memoryview 再 cast 成 2D 视图
        else:
            self.undo_buffer = undo_buffer

    # endregion

    # region ---------- Begin stroke
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def begin_stroke(self) -> tuple:
        """开启绘制 (Stroke) 初始化防重录标记。

        Update:
            - `self.modified_vtx_count`
            - `self.modified_vtx_bool_buffer`
        """

        _mask = self.modified_vtx_bool_buffer
        vtx_count: cython.int = _mask.shape[0]

        self.modified_vtx_count = 0

        memset(
            cython.cast(cython.p_uchar, _mask),
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
        record_indices: cython.int[::1] = None, # 设为可选
        record_count: cython.int = -1,          # 默认为 -1 触发自动计算
    ) -> cython.void:
        
        # 1. 局部变量提取 (性能优化)
        _mod_buf = self.modified_buffer
        _undo_buf = self.undo_buffer
        _mask = self.modified_vtx_bool_buffer
        _idx_pool = self.modified_vtx_indices_buffer
        _channels: cython.int = self.channel_count
        _current_count: cython.int = self.modified_vtx_count

        # 2. 智能逻辑判断
        # 如果没有传入索引 则认为是全量处理
        use_all: cython.bint = (record_indices is None)
        
        # 实际需要遍历的长度
        final_count: cython.int
        if use_all:  # noqa: SIM108
            final_count = _mod_buf.shape[0]
        else:
            final_count = record_indices.shape[0] if record_count < 0 else record_count

        # 3. 核心循环
        i: cython.int
        j: cython.int
        vtx_idx: cython.int

        for i in range(final_count):
            # 获取当前要处理的顶点索引
            vtx_idx = i if use_all else record_indices[i]

            # 防重录检查
            if _mask[vtx_idx] == 0:
                _mask[vtx_idx] = 1

                # 记录旧数据快照 (Copy channels)
                for j in range(_channels):
                    _undo_buf[_current_count, j] = _mod_buf[vtx_idx, j]

                # 存入索引池并累加计数
                _idx_pool[_current_count] = vtx_idx
                _current_count += 1

        self.modified_vtx_count = _current_count

    # endregion

    # region ---------- End Stroke
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.ccall
    def end_stroke(self) -> tuple:
        """结束绘制 打包出最新的 Undo & Redo 状态数据 包含双重稀疏压缩 。

        Returns:
            tuple: 若当前行程没有任何顶点被修改 则返回 None, 否则返回包含以下元素的元组:
                - modified_vertex_indices (ctypes.c_int32 数组): 涉及修改的顶点物理索引池 已截断有效长度 。
                - modified_channel_indices (ctypes.c_int32 数组): 实际发生变动的局部列索引。
                - old_sparse_ary (ctypes.c_float 数组): 极限压缩的 1D 旧快照。
                - new_sparse_ary (ctypes.c_float 数组): 极限压缩的 1D 新状态。
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
        # 用来记录通道是否修改    channel count 长度的 bool 数组
        channel_is_dirty: cython.p_char = cython.cast(cython.p_char, calloc(_channel_count, cython.sizeof(cython.char)))
        # _modified 数组和 _undo 数组逐元素对比 差异大于 1e-6 则为有变化 把 channel_is_dirty 中对应的数据设置为 1
        modified_channel_count: cython.int = 0
        for i in range(_modified_vtx_count):
            vtx_idx = _indices[i]
            for j in range(_channel_count):
                if channel_is_dirty[j] == 0:
                    diff = _modified[vtx_idx, j] - _undo[i, j]
                    if fabs(diff) > 1e-6:
                        channel_is_dirty[j] = 1
                        modified_channel_count += 1
        # 如果没有骨骼修改 代表这次绘制没有任何效果 直接释放内存 结束函数
        if modified_channel_count == 0:
            free(channel_is_dirty)
            return None

        # ----------------------------------------------------------------------------------------------------------------------------
        # 提取并记录脏列 ID
        # 使用 ctypes 申请 c_int32 数组
        modified_channel_indices = (ctypes.c_int32 * modified_channel_count)()
        modified_channel_view: cython.int[::1] = modified_channel_indices

        # 迭代查询channel真实的index,设置到 ctypes 数组中
        write_channel_idx: cython.int = 0
        for j in range(_channel_count):
            if channel_is_dirty[j] == 1:
                modified_channel_view[write_channel_idx] = j
                write_channel_idx += 1
        # 释放临时内存
        free(channel_is_dirty)

        # ----------------------------------------------------------------------------------------------------------------------------
        # 截取有效顶点索引 (原本的 buffer 是全量尺寸 这里切出有效长度)
        # 使用 ctypes 申请 c_int32 数组
        modified_vertex_indices = (ctypes.c_int32 * _modified_vtx_count)()
        modified_vtx_indices_view: cython.int[::1] = modified_vertex_indices

        # 查询 vtx_index 放入数组
        for i in range(_modified_vtx_count):
            modified_vtx_indices_view[i] = _indices[i]

        # ------------------------------------------------------------------------------------------------------------------------------
        # 申请 1D 稀疏数组内存
        sparse_size: cython.int = _modified_vtx_count * modified_channel_count

        # 使用 ctypes 申请 c_float 数组
        old_sparse_ary = (ctypes.c_float * sparse_size)()
        old_sparse_view: cython.float[::1] = old_sparse_ary

        new_sparse_ary = (ctypes.c_float * sparse_size)()
        new_sparse_view: cython.float[::1] = new_sparse_ary

        # 双向提取channel value
        write_idx: cython.int = 0
        channel_idx: cython.int = 0
        for i in range(_modified_vtx_count):
            vtx_idx = _indices[i]
            for j in range(modified_channel_count):
                channel_idx = modified_channel_view[j]

                # 提取点对点的 1D 压缩数据 用于返回给外部 Undo 栈
                old_sparse_view[write_idx] = _undo[i, channel_idx]
                new_sparse_view[write_idx] = _modified[vtx_idx, channel_idx]
                write_idx += 1

        return (modified_vertex_indices, modified_channel_indices, old_sparse_ary, new_sparse_ary)

    # endregion
