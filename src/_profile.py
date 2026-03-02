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
            self.t_start = time.perf_counter_ns()
        return self

    def step(self, step_name):
        """在你想要打点的地方调用此方法"""
        if not MicroProfiler._enabled:
            return

        t_now = time.perf_counter_ns()
        elapsed = t_now - self.t_start
        self.t_start = t_now  # 重置起点

        # 累加时间 (转换为毫秒)
        if step_name not in MicroProfiler._records:
            MicroProfiler._records[step_name] = 0.0
        MicroProfiler._records[step_name] += elapsed / 1_000_000.0

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
