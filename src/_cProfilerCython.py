# cProfiler.py
# cython: language_level=3
# distutils: language=c++

import cython
import functools


if cython.compiled:
    # 从同名的 .pxd 文件中导入 C++ 接口
    from cython.cimports._cProfiler import ProfilingColor, addCategory, eventBegin, eventEnd

    # 导入 C 标准库的 strdup
    from cython.cimports.libc.string import strdup

# 懒加载与内存分配池
_CATEGORY_ID: cython.int = -1
_IMMORTAL_C_STRINGS: dict = {}


@cython.cfunc
def get_category() -> cython.int:
    global _CATEGORY_ID
    if _CATEGORY_ID == -1:
        _CATEGORY_ID = addCategory(b"GSkin_Core", b"GSkin Native Profiler")
        if _CATEGORY_ID < 0:
            _CATEGORY_ID = 0
    return _CATEGORY_ID


@cython.cfunc
def get_immortal_c_string(name: str) -> cython.p_char:
    """在纯 C 堆上分配内存，并进行严谨的类型连环转换"""
    b_name: bytes
    c_ptr: cython.p_char
    ptr_address: cython.Py_ssize_t

    if name not in _IMMORTAL_C_STRINGS:
        b_name = name.encode("utf-8")
        c_ptr = strdup(b_name)
        # 将指针地址转为 Py_ssize_t 整数存入字典
        _IMMORTAL_C_STRINGS[name] = cython.cast(cython.Py_ssize_t, c_ptr)

    # 从字典取出整数，名正言顺地转回 C 指针
    ptr_address = _IMMORTAL_C_STRINGS[name]
    return cython.cast(cython.p_char, ptr_address)


# ==============================================================================
# 3. 对外暴露的 Python 接口
# ==============================================================================
@cython.cclass
class MayaNativeProfiler:
    event_id: cython.int
    color: "ProfilingColor"  # 使用字符串类型提示防止静态检查报错
    c_name: cython.p_char

    def __init__(self, name: str, color_idx: cython.int = 5):
        self.c_name = get_immortal_c_string(name)
        self.color = cython.cast("ProfilingColor", color_idx)

    def __enter__(self):
        self.event_id = eventBegin(get_category(), self.color, self.c_name, self.c_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        eventEnd(self.event_id)


def maya_profile(name=None, color_index=5):
    def decorator(func):
        event_name = name if name else func.__name__

        def wrapper(*args, **kwargs):
            with MayaNativeProfiler(event_name, color_index):
                return func(*args, **kwargs)

        return functools.wraps(func)(wrapper)

    return decorator
