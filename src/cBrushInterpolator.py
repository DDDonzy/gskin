"""
======================================================================================
Module: cStrokeInterpolator
======================================================================================
笔刷轨迹插值追踪引擎

核心特性 (Features):
  1. 纯数据驱动: 输入纯坐标 输出纯坐标序列。
  2. 距离累加器 (Distance Accumulator): 保证无论鼠标移动多快 插值点间距永远绝对均匀。
  3. 极致轻量: 剔除了复杂的曲线拟合逻辑 时间复杂度降至最低 适合极高频率的实时涂抹。
  4. 完整的生命周期闭环: 严格保障 begin(起笔) -> drag(运笔) -> end(收笔) 的端点闭合。
======================================================================================
"""

import math


class LinearStrokeInterpolator:
    """
    🖌️ 笔刷轨迹追踪器 (纯线性极速版)

    接管并处理用户鼠标输入的离散坐标 返回经过等距采样后的密集直线坐标列表。

    Attributes:
        spacing (float): 插值点之间的固定像素间距。
    """

    def __init__(self, spacing_pixels: float = 2.0):
        """
        初始化追踪器。

        Args:
            spacing_pixels (float): 笔刷印章之间的间隔距离（通常设为笔刷半径的 10%~25%）。
        """
        self.spacing = spacing_pixels

        # 内部状态寄存器 纯线性只需要记录上一个点
        self._last_raw_pos: tuple | None = None
        self._last_draw_pos: tuple | None = None
        self._leftover_dist: float = 0.0

    def begin_stroke(self, x: float, y: float) -> list[tuple]:
        """
        记录第一笔的起始坐标 并初始化内部状态。
        """
        self._last_raw_pos = (x, y)
        self._last_draw_pos = (x, y)
        self._leftover_dist = 0.0
        return [(x, y)]

    def drag_stroke(self, curr_x: float, curr_y: float) -> list[tuple]:
        """
        接收实时鼠标坐标 进行物理防抖与线性等距采样。
        """
        # 防御：若未起笔直接运笔 强制转为起笔逻辑
        if not self._last_raw_pos:
            return self.begin_stroke(curr_x, curr_y)

        # 物理防抖：剔除极小范围的鼠标抖动噪音
        dist = math.hypot(curr_x - self._last_raw_pos[0], curr_y - self._last_raw_pos[1])
        if dist < 0.1:
            return []

        # 核心逻辑：从上一个真实鼠标点 向当前真实鼠标点连线并采样
        pts = self._sample_line(self._last_raw_pos, (curr_x, curr_y))

        # 更新记忆 将当前点作为下一次运算的起点
        self._last_raw_pos = (curr_x, curr_y)
        return pts

    def end_stroke(self, curr_x: float, curr_y: float) -> list[tuple]:
        """
        强制闭合最后一段缺失的轨迹 并重置追踪器。
        """
        res = []

        # 补齐最后一段滑动距离
        if self._last_raw_pos:
            res.extend(self._sample_line(self._last_raw_pos, (curr_x, curr_y)))

        # 兜底强制闭合：如果上述采样未能精准覆盖到最终释放点 强行补入最后一点
        if not res or math.hypot(res[-1][0] - curr_x, res[-1][1] - curr_y) > 0.1:
            res.append((curr_x, curr_y))

        # 彻底清空寄存状态
        self._last_raw_pos = None
        self._last_draw_pos = None
        self._leftover_dist = 0.0
        return res

    # =====================================================================
    # 底层数学引擎 (Internal Math Engine)
    # =====================================================================

    def _sample_line(
        self,
        p_start: tuple,
        p_end: tuple,
    ) -> list[tuple]:
        """直线采样器 计算两点间距并移交累加器。"""
        dist = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
        return self._accumulate(dist, p_start, p_end)

    def _accumulate(
        self,
        segment_dist: float,
        p_start: tuple,
        p_end: tuple,
    ) -> list[tuple]:
        """
        核心距离累加器。
        将传入的线段距离注入能量槽 满 `spacing` 阈值即吐出一个精确的插值坐标点。
        """
        if segment_dist <= 0:
            return []

        self._leftover_dist += segment_dist
        results = []

        while self._leftover_dist >= self.spacing:
            # 倒推计算刚好达到 spacing 时的 t 值比例
            t = (segment_dist - (self._leftover_dist - self.spacing)) / segment_dist
            t = max(0.0, min(1.0, t))  # 钳制防止浮点数精度溢出

            new_p = (p_start[0] + (p_end[0] - p_start[0]) * t, p_start[1] + (p_end[1] - p_start[1]) * t)

            results.append(new_p)
            self._last_draw_pos = new_p
            self._leftover_dist -= self.spacing

        return results
