# cProfiler.pxd
# cython: language_level=3

cdef extern from "maya/MProfiler.h" namespace "MProfiler":
    cdef enum ProfilingColor:
        kColor1, kColor2, kColor3, kColor4, kColor5, kColor6, kColor7, kColor8

    int addCategory(const char* name, const char* description)
    int eventBegin(int categoryId, ProfilingColor color, const char* name, const char* description)
    void eventEnd(int eventId)