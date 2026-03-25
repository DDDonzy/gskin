"""
======================================================================================
Module: cStrokeInterpolator
======================================================================================
笔刷轨迹插值与追踪 (Stroke Interpolation Engine)

绘图软件标准的笔刷轨迹处理能力。主要用于解决鼠标/数位板采样率
不足导致的“轨迹折线感”和“笔触断层”问题。

核心特性 (Features):
  1. 纯数据驱动 (Data-Driven): 无 UI 依赖，输入纯坐标，输出纯坐标序列。
  2. 距离累加器 (Distance Accumulator): 保证无论鼠标移动多快，插值点间距永远绝对均匀等距。
  3. 双模式切换 (Dual Mode):
     - Linear (线性): 适合多边形套索、硬边几何笔刷。
     - B-Spline (二阶B样条): 采用中点贝塞尔算法，保证轨迹 C1 连续，转角丝滑圆润。
  4. 完整的生命周期闭环: 严格保障 begin(起笔) -> drag(运笔) -> end(收笔) 的端点闭合。

======================================================================================
"""

import math

class SmoothStrokeInterpolator:
    """
    🖌️ 笔刷轨迹追踪器 (双模式版)
    
    接管并处理用户鼠标输入的离散坐标，返回经过等距采样和曲线拟合后的密集坐标列表。
    
    Attributes:
        spacing (float): 插值点之间的固定像素间距。
        smooth_path (bool): 是否启用 B-Spline 曲线平滑。True 为曲线，False 为纯直线。
        raw_points (list[tuple[float, float]]): 内部维护的鼠标原始采样点滑动窗口。
    """

    def __init__(self, spacing_pixels: float = 2.0, smooth_path: bool = True):
        """
        初始化追踪器。
        
        Args:
            spacing_pixels (float): 笔刷印章之间的间隔距离（通常设为笔刷半径的 10%~25%）。
            smooth_path (bool): 轨迹模式开关。True 开启二阶曲线拟合，False 使用线性插值。
        """
        self.smooth_path = smooth_path
        self.spacing = spacing_pixels
        
        # 内部状态寄存器
        self.raw_points: list[tuple[float, float]] = []
        self._leftover_dist: float = 0.0
        self._last_draw_pos: tuple[float, float] | None = None

    def begin_stroke(self, x: float, y: float) -> list[tuple[float, float]]:
        """
        【起笔】生命周期起点：记录第一笔的起始坐标，并初始化内部状态。
        
        Args:
            x (float): 鼠标按下的 X 坐标。
            y (float): 鼠标按下的 Y 坐标。
            
        Returns:
            list[tuple[float, float]]: 包含起笔坐标的列表，保证“落地见墨”。
        """
        self.raw_points = [(x, y)]
        self._leftover_dist = 0.0
        self._last_draw_pos = (x, y)
        return [(x, y)]

    def drag_stroke(self, curr_x: float, curr_y: float) -> list[tuple[float, float]]:
        """
        【运笔】生命周期中段：接收实时鼠标坐标，进行防抖过滤、曲线拟合与等距采样。
        
        Args:
            curr_x (float): 鼠标拖拽时的当前 X 坐标。
            curr_y (float): 鼠标拖拽时的当前 Y 坐标。
            
        Returns:
            list[tuple[float, float]]: 生成的等距插值坐标列表。如果距离过短未触发采样，返回空列表 []。
        """
        # 防御：若未起笔直接运笔，强制转为起笔逻辑
        if not self.raw_points:
            return self.begin_stroke(curr_x, curr_y)

        # 物理防抖：剔除极小范围的鼠标抖动噪音
        last_raw = self.raw_points[-1]
        if math.hypot(curr_x - last_raw[0], curr_y - last_raw[1]) < 0.1:
            return []

        self.raw_points.append((curr_x, curr_y))

        # ==========================================
        # 🟢 模式 1: 纯线性插值 (Linear)
        # ==========================================
        if self.smooth_path is False:
            p_start, p_end = self.raw_points[-2], self.raw_points[-1]
            pts = self._sample_line(p_start, p_end)
            
            # 踢掉最老的点，保持滑动窗口只有 1 个起点
            self.raw_points.pop(0)
            return pts

        # ==========================================
        # 🔵 模式 2: 二阶 B 样条曲线 (B-Spline)
        # ==========================================
        if self.smooth_path is True:
            if len(self.raw_points) == 2:
                # 初始段：绘制从 起点 到 第1与第2点【中点】 的前导直线
                p0, p1 = self.raw_points
                mid = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
                return self._sample_line(p0, mid)

            if len(self.raw_points) == 3:
                # 曲线段：绘制 前一中点 到 当前中点 的贝塞尔曲线
                p0, p1, p2 = self.raw_points
                m0 = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
                m1 = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
                pts = self._sample_bezier(m0, p1, m1)

                # 踢掉 P0，保留 P1, P2 等待 P3 的到来，维持 3 点滑动窗口
                self.raw_points.pop(0)
                return pts

        return []

    def end_stroke(self, curr_x: float, curr_y: float) -> list[tuple[float, float]]:
        """
        【收笔】生命周期终点：强制闭合最后一段缺失的轨迹，并重置追踪器以备下次使用。
        
        Args:
            curr_x (float): 鼠标释放时的 X 坐标。
            curr_y (float): 鼠标释放时的 Y 坐标。
            
        Returns:
            list[tuple[float, float]]: 收尾段生成的等距插值坐标列表。
        """
        res = []

        # --- 线性模式收尾 ---
        if self.smooth_path is False:
            if self.raw_points:
                res.extend(self._sample_line(self.raw_points[-1], (curr_x, curr_y)))

        # --- B样条模式收尾 ---
        elif self.smooth_path is True:
            if len(self.raw_points) >= 2:
                # 提取最后两点，补齐由中点射向鼠标释放位置的收尾贝塞尔曲线
                p_prev, p_last = self.raw_points[-2:]
                mid = ((p_prev[0] + p_last[0]) * 0.5, (p_prev[1] + p_last[1]) * 0.5)
                res.extend(self._sample_bezier(mid, p_last, (curr_x, curr_y)))
            elif len(self.raw_points) == 1:
                # 仅有单点拖拽即释放的情况，退化为直线补齐
                res.extend(self._sample_line(self.raw_points[0], (curr_x, curr_y)))

        # 兜底强制闭合：如果上述采样未能精准覆盖到最终释放点，强行补入最后一点
        if not res or math.hypot(res[-1][0] - curr_x, res[-1][1] - curr_y) > 0.1:
            res.append((curr_x, curr_y))

        # 彻底清空寄存状态
        self.raw_points.clear()
        self._last_draw_pos = None
        self._leftover_dist = 0.0
        return res

    # =====================================================================
    # 底层数学引擎 (Internal Math Engine)
    # =====================================================================

    def _sample_line(self, p_start: tuple[float, float], p_end: tuple[float, float]) -> list[tuple[float, float]]:
        """
        直线采样器。
        
        Args:
            p_start (tuple): 线段起点。
            p_end (tuple): 线段终点。
            
        Returns:
            list[tuple[float, float]]: 采样出的坐标列表。
        """
        dist = math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1])
        return self._accumulate(dist, p_start, p_end)

    def _sample_bezier(self, m0: tuple[float, float], ctrl: tuple[float, float], m1: tuple[float, float]) -> list[tuple[float, float]]:
        """
        二次贝塞尔曲线等距采样器。采用步进细分近似法，将曲线切分成微小直线段进行距离累加。
        
        Args:
            m0 (tuple): 曲线起点 (前段中点)。
            ctrl (tuple): 控制点 (真实报点)。
            m1 (tuple): 曲线终点 (后段中点)。
            
        Returns:
            list[tuple[float, float]]: 沿着曲线等距分布的坐标列表。
        """
        steps = 16  # 采样细分精度
        prev_p = m0
        points = []
        
        for i in range(1, steps + 1):
            t = i / steps
            u = 1.0 - t
            # 二次贝塞尔曲线公式展开
            curr_p = (
                u * u * m0[0] + 2 * u * t * ctrl[0] + t * t * m1[0], 
                u * u * m0[1] + 2 * u * t * ctrl[1] + t * t * m1[1]
            )
            dist = math.hypot(curr_p[0] - prev_p[0], curr_p[1] - prev_p[1])
            points.extend(self._accumulate(dist, prev_p, curr_p))
            prev_p = curr_p
            
        return points

    def _accumulate(self, segment_dist: float, p_start: tuple[float, float], p_end: tuple[float, float]) -> list[tuple[float, float]]:
        """
        核心距离累加器。将传入的线段距离注入能量槽，满 `spacing` 阈值即吐出一个精确的插值坐标点。
        
        Args:
            segment_dist (float): 当前运算线段的长度。
            p_start (tuple): 线段起点。
            p_end (tuple): 线段终点。
            
        Returns:
            list[tuple[float, float]]: 在该线段上合法生成的插值点列表。
        """
        if segment_dist <= 0:
            return []
            
        self._leftover_dist += segment_dist
        results = []
        
        while self._leftover_dist >= self.spacing:
            # 倒推计算刚好达到 spacing 时的 t 值比例
            t = (segment_dist - (self._leftover_dist - self.spacing)) / segment_dist
            t = max(0.0, min(1.0, t)) # 钳制防止浮点数精度溢出
            
            new_p = (
                p_start[0] + (p_end[0] - p_start[0]) * t, 
                p_start[1] + (p_end[1] - p_start[1]) * t
            )
            
            results.append(new_p)
            self._last_draw_pos = new_p
            self._leftover_dist -= self.spacing
            
        return results