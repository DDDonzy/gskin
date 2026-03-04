import time


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
        MicroProfiler._records[step_name] += elapsed*1000

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


import cProfile
import pstats
import io


class DeepProfiler:
    """基于 cProfile 的深度函数调用分析器"""

    _profiler = cProfile.Profile()
    _counter = 0
    _target_runs = 30

    def __enter__(self):
        # 进入时开启抓取
        self._profiler.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 退出时暂停抓取
        self._profiler.disable()
        DeepProfiler._counter += 1

        # 达到指定帧数，打印报告
        if DeepProfiler._counter >= DeepProfiler._target_runs:
            print(f"\n🔬 [DeepProfiler | 累计 {DeepProfiler._target_runs} 帧深度函数分析]")
            print("-" * 80)

            s = io.StringIO()
            # 💡 核心：按 'tottime' (自身内部耗时) 排序
            # 如果想看包含子函数的总耗时，可以改成 'cumtime'
            sortby = "tottime"
            ps = pstats.Stats(self._profiler, stream=s).sort_stats(sortby)

            # 去掉冗长的绝对路径，让界面清爽
            ps.strip_dirs()
            # 只打印排名前 20 的“性能刺客”
            ps.print_stats(20)

            print(s.getvalue())
            print("=" * 80 + "\n")

            # 清空缓存，准备下一轮
            DeepProfiler._counter = 0
            self._profiler.clear()


import maya.OpenMaya as om1  # 👈 切回 API 1.0

# 注册类别
MY_PLUGIN_CATEGORY = om1.MProfiler.addCategory("CythonSkinPlugin", "Cython Nodes")

class MayaNativeProfiler:
    """基于 Maya API 1.0 的原生性能分析器"""
    
    def __init__(self, event_name, color=5):
        self.event_name = event_name
        # 颜色 ID 推荐: 2(橘红), 5(蓝色), 6(红色), 7(绿色)
        self.color = color 
        self.event_id = 0

    def __enter__(self):
        # API 1.0 的调用方式
        self.event_id = om1.MProfiler.eventBegin(MY_PLUGIN_CATEGORY, self.color, self.event_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        om1.MProfiler.eventEnd(self.event_id)