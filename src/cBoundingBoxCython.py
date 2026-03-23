# cython_core.py
import cython

# 彻底关闭边界检查和负数索引检查
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.ccall
def compute_bbox_fast(
    points: cython.float[:],    # 💥 输入是一维的 float32 顶点位置数组
    num_verts: cython.int       # 💥 顶点总数
):
    """
    极速核心 遍历一维连续顶点数组 寻找最小和最大边界点 (纯 Python 语法模式)
    """
    # ==========================================
    # 静态类型声明区
    # ==========================================
    _i: cython.int = 0
    idx: cython.int = 0
    
    # 初始包围盒极值 (正负极颠倒 用于确保第一次比较时必定被覆写)
    min_x: cython.float = 9999999.0
    min_y: cython.float = 9999999.0
    min_z: cython.float = 9999999.0
    
    max_x: cython.float = -9999999.0
    max_y: cython.float = -9999999.0
    max_z: cython.float = -9999999.0
    
    x: cython.float = 0.0
    y: cython.float = 0.0
    z: cython.float = 0.0

    # ==========================================
    # 纯 C 级别的极速循环区
    # ==========================================
    for _i in range(num_verts):
        # 提取当前顶点的 XYZ
        x = points[idx]
        y = points[idx + 1]
        z = points[idx + 2]
        idx += 3  # 指针步进 3 个 float

        # X 轴比对
        if x < min_x:
            min_x = x
        elif x > max_x:
            max_x = x
            
        # Y 轴比对
        if y < min_y:
            min_y = y
        elif y > max_y:
            max_y = y
            
        # Z 轴比对
        if z < min_z:
            min_z = z
        elif z > max_z:
            max_z = z

    # 循环结束后 打包成两个普通的 Python Tuple 返回给外界
    return (min_x, min_y, min_z), (max_x, max_y, max_z)