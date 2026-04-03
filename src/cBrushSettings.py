# cBrushSettings.py
from dataclasses import dataclass

@dataclass
class _BrushSettingsData:
    """存放笔刷半径、强度、模式等用户 UI 配置的底层数据结构"""
    radius              : float = 1.0
    strength            : float = 1.0
    iter                : int   = 10
    falloff_type        : int   = 1     # 0:Linear, 1:Airbrush, 2:Solid, 3:Dome, 4:Spike
    mode                : int   = 0     # 0:Add, 1:Sub, 2:Replace, 3:Multiply, 4:Smooth, 5:Sharp
    brush_spacing_ratio : float = 0.1
    use_surface         : bool  = True

# 实例化一个全局唯一的单例对象
BrushSettings = _BrushSettingsData()