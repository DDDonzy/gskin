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

    def begin_stroke(self, x: float, y: float, p: float = 1.0) -> list[tuple]:
        """
        Args:
            x (float): 起笔坐标 X。
            y (float): 起笔坐标 Y。
            p (float): 起笔压感值（默认为 1.0）。
        """
        self._last_raw_pos = (x, y, p)
        self._last_draw_pos = (x, y, p)
        self._leftover_dist = 0.0
        return [(x, y, p)]

    def drag_stroke(self, curr_x: float, curr_y: float, curr_p: float = 1.0) -> list[tuple]:
        """
        Args:
            x (float): 起笔坐标 X。
            y (float): 起笔坐标 Y。
            p (float): 起笔压感值（默认为 1.0）。
        """
        if not self._last_raw_pos:
            return self.begin_stroke(curr_x, curr_y, curr_p)

        dist = math.hypot(curr_x - self._last_raw_pos[0], curr_y - self._last_raw_pos[1])
        if dist < 0.1:
            return []

        pts = self._sample_line(self._last_raw_pos, (curr_x, curr_y, curr_p))
        self._last_raw_pos = (curr_x, curr_y, curr_p)
        return pts

    def end_stroke(self, curr_x: float, curr_y: float, curr_p: float = 1.0) -> list[tuple]:
        """
        Args:
            x (float): 起笔坐标 X。
            y (float): 起笔坐标 Y。
            p (float): 起笔压感值（默认为 1.0）。
        """
        res = []
        if self._last_raw_pos:
            res.extend(self._sample_line(self._last_raw_pos, (curr_x, curr_y, curr_p)))

        if not res or math.hypot(res[-1][0] - curr_x, res[-1][1] - curr_y) > 0.1:
            res.append((curr_x, curr_y, curr_p))

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
        if segment_dist <= 0:
            return []

        self._leftover_dist += segment_dist
        results = []

        while self._leftover_dist >= self.spacing:
            t = (segment_dist - (self._leftover_dist - self.spacing)) / segment_dist
            t = max(0.0, min(1.0, t))

            # X, Y 坐标插值
            nx = p_start[0] + (p_end[0] - p_start[0]) * t
            ny = p_start[1] + (p_end[1] - p_start[1]) * t

            # 🌟 压感(Pressure) 线性插值
            np = p_start[2] + (p_end[2] - p_start[2]) * t

            new_p = (nx, ny, np)
            results.append(new_p)
            self._last_draw_pos = new_p
            self._leftover_dist -= self.spacing

        return results









class SplineStrokeInterpolator:
    """
    🖌️ 笔刷轨迹追踪器 (Catmull-Rom 样条曲线极速版)

    接管用户的离散鼠标坐标，首先使用 Catmull-Rom 算法生成平滑的弧线，
    然后通过距离累加器，在弧线上进行绝对均匀的像素级重采样。
    彻底消除底层降帧时的“折线感”与“多边形化”。

    Attributes:
        spacing (float): 插值点之间的固定像素间距。
    """

    def __init__(self, spacing_pixels: float = 2.0):
        """
        初始化曲线追踪器。
        """
        self.spacing = spacing_pixels

        # 曲线插值需要至少 4 个控制点，因此我们维护一个历史队列
        self._history: list[tuple] = []
        self._last_draw_pos: tuple | None = None
        self._leftover_dist: float = 0.0

    def begin_stroke(self, x: float, y: float, p: float = 1.0) -> list[tuple]:
        """起笔：初始化队列，首点需要被复制以充当曲线的起点约束。"""
        pt = (x, y, p)
        # Catmull-Rom 需要 P0, P1, P2, P3。起笔时我们将首点复制，作为 P0 和 P1
        self._history = [pt, pt]
        self._last_draw_pos = pt
        self._leftover_dist = 0.0
        return [pt]

    def drag_stroke(self, curr_x: float, curr_y: float, curr_p: float = 1.0) -> list[tuple]:
        """运笔：收集点位，当积攒满 4 个点时，生成并吐出中间的平滑曲线。"""
        if not self._history:
            return self.begin_stroke(curr_x, curr_y, curr_p)

        pt = (curr_x, curr_y, curr_p)
        last_pt = self._history[-1]

        # 过滤掉极其微小的鼠标抖动
        if math.hypot(pt[0] - last_pt[0], pt[1] - last_pt[1]) < 0.1:
            return []

        self._history.append(pt)
        results = []

        # 🌟 核心：滑动窗口机制。只要有了 4 个控制点，就算出 P1 到 P2 之间的完美弧线
        while len(self._history) >= 4:
            p0, p1, p2, p3 = self._history[0:4]

            # 动态决定曲线的分段数 (鼠标甩得越快，距离越长，采样的曲线点就越密集，保证绝对圆滑)
            segment_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            steps = max(10, int(segment_dist / 2.0)) # 至少每 2 个像素采样一次曲线

            # 生成高密度的曲线点
            curve_pts = self._generate_catmull_rom_segment(p0, p1, p2, p3, steps)

            # 将高密度的曲线点，喂给匀距累加器进行真正的“盖章”分发
            for i in range(1, len(curve_pts)):
                results.extend(self._sample_line(curve_pts[i-1], curve_pts[i]))

            # 窗口向前滑动，丢弃 P0，等待下一个鼠标点进来充当新的 P3
            self._history.pop(0)

        return results

    def end_stroke(self, curr_x: float, curr_y: float, curr_p: float = 1.0) -> list[tuple]:
        """收笔：强制把历史队列里还没画完的尾巴 (P2 -> P3) 给闭合掉。"""
        pt = (curr_x, curr_y, curr_p)
        
        if self._history:
            last_pt = self._history[-1]
            if math.hypot(pt[0] - last_pt[0], pt[1] - last_pt[1]) >= 0.1:
                self._history.append(pt)

        results = []

        # 强制清空队列 (Flush)。如果还有剩余的点，通过复制末尾点作为虚假控制点来画完最后一段
        while len(self._history) > 2:
            p0, p1, p2 = self._history[0:3]
            p3 = self._history[-1] # 复制末尾点充当结束约束

            segment_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            steps = max(10, int(segment_dist / 2.0))

            curve_pts = self._generate_catmull_rom_segment(p0, p1, p2, p3, steps)
            for i in range(1, len(curve_pts)):
                results.extend(self._sample_line(curve_pts[i-1], curve_pts[i]))

            self._history.pop(0)

        # 保底端点闭合
        if self._last_draw_pos and math.hypot(pt[0] - self._last_draw_pos[0], pt[1] - self._last_draw_pos[1]) > 0.1:
            results.append(pt)

        self._history.clear()
        self._last_draw_pos = None
        self._leftover_dist = 0.0
        return results

    # =====================================================================
    # 底层数学引擎 (Internal Math Engine)
    # =====================================================================

    def _generate_catmull_rom_segment(self, p0: tuple, p1: tuple, p2: tuple, p3: tuple, steps: int) -> list[tuple]:
        """
        Catmull-Rom 样条曲线数学内核。
        不仅平滑 X 和 Y 的坐标，连压感 P 也使用三次方程进行完美过渡！
        """
        pts = []
        for i in range(steps + 1):
            t = i / steps
            t2 = t * t
            t3 = t2 * t

            # 核心张量矩阵计算
            # 形式: 0.5 * ( (2*P1) + (-P0+P2)*t + (2*P0 - 5*P1 + 4*P2 - P3)*t^2 + (-P0 + 3*P1 - 3*P2 + P3)*t^3 )
            
            nx = 0.5 * ( (2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3 )
            ny = 0.5 * ( (2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3 )
            
            # 🌟 压感也享受最高级别的三次曲线平滑过渡！
            np = 0.5 * ( (2 * p1[2]) + (-p0[2] + p2[2]) * t + (2 * p0[2] - 5 * p1[2] + 4 * p2[2] - p3[2]) * t2 + (-p0[2] + 3 * p1[2] - 3 * p2[2] + p3[2]) * t3 )
            
            # 钳制压感防止函数振铃效应(Overshoot)产生负数或超出1.0
            np = max(0.0, min(1.0, np))

            pts.append((nx, ny, np))
            
        return pts

    def _sample_line(self, p_start: tuple, p_end: tuple) -> list[tuple]:
        """直线采样器，计算微小曲线片段的间距并移交累加器。"""
        dist = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
        return self._accumulate(dist, p_start, p_end)

    def _accumulate(self, segment_dist: float, p_start: tuple, p_end: tuple) -> list[tuple]:
        """绝对匀距累加器 (原封不动保留，完美融合)。"""
        if segment_dist <= 0:
            return []

        self._leftover_dist += segment_dist
        results = []

        while self._leftover_dist >= self.spacing:
            t = (segment_dist - (self._leftover_dist - self.spacing)) / segment_dist
            t = max(0.0, min(1.0, t))

            nx = p_start[0] + (p_end[0] - p_start[0]) * t
            ny = p_start[1] + (p_end[1] - p_start[1]) * t
            np = p_start[2] + (p_end[2] - p_start[2]) * t

            new_p = (nx, ny, np)
            results.append(new_p)
            self._last_draw_pos = new_p
            self._leftover_dist -= self.spacing

        return results