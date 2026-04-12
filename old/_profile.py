# encoding=utf-8

import time

import cProfile
import pstats
import os

import maya.OpenMaya as om1


class MicroProfiler:
    """纳秒级切片计时器 (专为微秒级优化设计)"""

    _records = {}
    _counter = 0
    _target_runs = 30
    _enabled = True

    def __init__(self, target_runs=30, enable=True):
        MicroProfiler._target_runs = target_runs
        MicroProfiler._enabled = enable
        self.t_start = 0

    def __enter__(self):
        if MicroProfiler._enabled:
            # 记录起始的纳秒时间戳
            self.t_start = time.perf_counter()
        return self

    def step(self, step_name):
        """在你想要打点的地方调用此方法"""
        if not MicroProfiler._enabled:
            return

        t_now = time.perf_counter()
        elapsed = t_now - self.t_start
        self.t_start = t_now  # 重置起点

        # 累加时间 (转换为毫秒)
        if step_name not in MicroProfiler._records:
            MicroProfiler._records[step_name] = 0.0
        MicroProfiler._records[step_name] += elapsed * 1000

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not MicroProfiler._enabled:
            return

        MicroProfiler._counter += 1
        if MicroProfiler._counter >= MicroProfiler._target_runs:
            print(f"\n🔬 [MicroProfiler | 累计 {MicroProfiler._target_runs} 帧平均耗时]")
            print("-" * 50)

            total_time = 0.0
            # 排序并打印
            for name, cum_time in sorted(MicroProfiler._records.items(), key=lambda x: x[1], reverse=True):
                avg_time = cum_time / MicroProfiler._target_runs
                total_time += avg_time
                print(f"{name.ljust(25)}: {avg_time:.4f} ms")

            print("-" * 50)
            print(f"{'总计 (Total)'.ljust(25)}: {total_time:.4f} ms")
            print("=" * 50 + "\n")

            # 清空重置
            MicroProfiler._counter = 0
            MicroProfiler._records.clear()


class DeepProfiler:
    """基于 cProfile 的深度函数调用分析器 (毫秒级高精度输出)"""

    _profiler = cProfile.Profile()
    _counter = 0

    # 💡 在这里集中配置！
    _target_runs = 30  # 累计抓取多少次后打印
    _top_n = 100  # 报告显示前多少名 (-1 表示全部显示)

    def __enter__(self):
        self._profiler.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._profiler.disable()
        DeepProfiler._counter += 1

        if DeepProfiler._counter >= DeepProfiler._target_runs:
            ps = pstats.Stats(self._profiler).sort_stats("tottime")

            # 💡 新增 1：计算整个剖析区块的单帧平均总耗时
            total_time_ms = ps.total_tt * 1000.0
            avg_frame_time = total_time_ms / DeepProfiler._target_runs

            print(f"\n🔬 [DeepProfiler | 累计 {DeepProfiler._target_runs} 帧深度分析 | 显示 Top {DeepProfiler._top_n}]")
            print(f"⏱️  全局单帧平均总耗时 (Avg per frame): {avg_frame_time:.4f} ms")
            print("-" * 120)

            # 💡 新增 2：在表头中间插入 'avg/frm(ms)' 列
            header = f"{'ncalls':>10} {'tottime(ms)':>12} {'avg/frm(ms)':>12} {'percall(ms)':>12} {'cumtime(ms)':>12} {'percall(ms)':>12}  {'filename:lineno(function)'}"
            print(header)

            func_list = ps.fcn_list if DeepProfiler._top_n == -1 else ps.fcn_list[: DeepProfiler._top_n]

            for func in func_list:
                cc, nc, tt, ct, callers = ps.stats[func]

                tt_ms = tt * 1000.0
                ct_ms = ct * 1000.0

                # 💡 新增 3：计算该函数平摊到单帧的平均耗时
                pf_tt_ms = tt_ms / DeepProfiler._target_runs

                pc_tt_ms = (tt_ms / nc) if nc > 0 else 0.0
                pc_ct_ms = (ct_ms / cc) if cc > 0 else 0.0

                call_str = str(nc) if nc == cc else f"{nc}/{cc}"
                file_path, line, func_name = func
                file_name = os.path.basename(file_path)

                if file_name == "~":
                    func_str = func_name
                else:
                    func_str = f"{file_name}:{line} ({func_name})"

                # 💡 输出时将 pf_tt_ms 填入对应列
                print(f"{call_str:>10} {tt_ms:>12.4f} {pf_tt_ms:>12.4f} {pc_tt_ms:>12.4f} {ct_ms:>12.4f} {pc_ct_ms:>12.4f}  {func_str}")

            print("=" * 120 + "\n")

            DeepProfiler._counter = 0
            self._profiler.clear()


# 注册类别
MY_PLUGIN_CATEGORY = om1.MProfiler.addCategory("CythonSkinPlugin", "Cython Nodes")

# ==============================================================================
# 💡 核心修复：全局字符串保活池
# 只要字符串被丢进这个 set 里，它的引用计数永远大于 0，绝对不会被 GC 回收。
# 这样底层 C++ 拿到的指针就永远是有效且清晰的！
# ==============================================================================
_ALIVE_STRINGS = set()


def get_safe_string(text):
    """确保传入的是 str，并强制全局续命"""
    if not isinstance(text, str):
        text = str(text)
    _ALIVE_STRINGS.add(text)
    return text


class MayaNativeProfiler:
    """基于 Maya API 1.0 的原生性能分析器"""

    def __init__(self, event_name, color=5):
        # 1. 把名字丢进保活池，拿到拥有不死之身的 str
        self.event_name = get_safe_string(event_name)

        # 颜色 ID 推荐: 2(橘红), 5(蓝色), 6(红色), 7(绿色)
        self.color = color
        self.event_id = 0

    def __enter__(self):
        # 2. 直接传原汁原味的 Python str (绝不传 bytes)
        self.event_id = om1.MProfiler.eventBegin(MY_PLUGIN_CATEGORY, self.color, self.event_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        om1.MProfiler.eventEnd(self.event_id)
