from __future__ import annotations

import array
import ctypes

import maya.OpenMaya as OpenMaya  # type: ignore


__all__ = [
    "TopologyContext",
]


class TopologyContext:
    """
    Topology 数据上下文
    支持从OpenMaya.MFnMesh解析Topology数据
    支持构建CSR邻边表

    CSR Data:
    ```
    ===================================================================================
    Offsets:
        Index:    [0]            [1]                     [2]            [3] (End bound)
        Data:      0              2                       5              6
    Indices:
        Index:    [0]    [1]     [2]    [3]    [4]       [5]            (Out of bounds)
        Data:      x      x       y      y      y         z
                   \________/     \________________/      \_/
                   (2 items)          (3 items)        (1 item)
    ===================================================================================
    ```
    """

    __slots__ = ("position",
                 
                 "num_vertices",
                 "num_edges",
                 "num_polygons",
                 "num_triangles",
                 
                 "tri_face_indices",
                 "tri_edge_indices",
                 "quad_face_indices",
                 "quad_edge_indices",

                 "v2v_offsets",
                 "v2v_indices",
                 "v2f_offsets",
                 "v2f_indices",
                 
                 "mFnMesh")  # fmt:skip

    # fmt:off
    num_vertices : int
    num_edges    : int
    num_polygons : int
    num_triangles: int
    # position
    position: memoryview
    # 基础拓扑
    tri_face_indices : memoryview
    tri_edge_indices : memoryview
    quad_face_indices: memoryview
    quad_edge_indices: memoryview
    # v2v CSR
    v2v_offsets: memoryview
    v2v_indices: memoryview
    # v2f CSR
    v2f_offsets: memoryview
    v2f_indices: memoryview
    # fmt:on

    def __init__(self, mFnMesh: OpenMaya.MFnMesh = None):
        # fmt:off
        self.init_default()

        if mFnMesh:
            self.mFnMesh:OpenMaya.MFnMesh = mFnMesh

    def init_default(self):
        # fmt:off
        self.mFnMesh      = None
        self.num_vertices = 0
        self.num_edges    = 0
        self.num_polygons = 0
        self.num_triangles= 0
        # position
        self.position = None
        # 基础拓扑
        self.tri_face_indices  = None
        self.tri_edge_indices  = None
        self.quad_face_indices = None
        self.quad_edge_indices = None
        # v2v CSR
        self.v2v_offsets = None
        self.v2v_indices = None
        # v2f CSR
        self.v2f_offsets = None
        self.v2f_indices = None
        # fmt:on

    def update_position(self, mFnMesh: OpenMaya.MFnMesh = None):
        """
        更新顶点位置数据
        Args:
            mFnMesh (OpenMaya.MFnMesh): 输入mFnMesh, 如果不输入, 自动使用`self.mFnMesh`
        Update:
            - `self.position`
            - `self.num_vertices`
        """
        fnMesh = mFnMesh if mFnMesh else self.mFnMesh
        if fnMesh is None:
            raise RuntimeError("mFnMesh is not set.")

        num_vertices: int = fnMesh.numVertices()
        self.num_vertices = num_vertices

        address = int(fnMesh.getRawPoints())
        raw_buffer = (ctypes.c_float * (num_vertices * 3)).from_address(address)

        self.position = memoryview(raw_buffer).cast("B").cast("f")

    def update_topology(self, mFnMesh: OpenMaya.MFnMesh = None, update_csr: bool = True):
        """
        更新顶点位置数据
        TODO polygon edge 查询效率太低了,想办法优化
        Args:
            mFnMesh (OpenMaya.MFnMesh): 输入mFnMesh, 如果不输入, 自动使用`self.mFnMesh`
            update_csr (bool): 是否更新 csr 邻边数据. Default = True.
        Update:
            - `self.tri_face_indices`
            - `self.tri_edge_indices`
            - `self.quad_face_indices`
            - `self.quad_edge_indices`
            - `self.v2v_offsets`
            - `self.v2v_indices`
            - `self.v2f_offsets`
            - `self.v2f_indices`
        """
        mFnMesh = mFnMesh if mFnMesh else self.mFnMesh
        if mFnMesh is None:
            raise RuntimeError("mFnMesh is not set.")

        _new_num_edges = mFnMesh.numEdges()
        _new_num_vertices = mFnMesh.numVertices()
        _new_num_polygons = mFnMesh.numPolygons()

        if (self.num_edges == _new_num_edges and
            self.num_vertices == _new_num_vertices and 
            self.num_polygons == _new_num_polygons):  # fmt:skip
            # 判断每个条件, 如果全部相等, 说明topology 相同无需更新
            return False

        # --- count
        self.num_edges = _new_num_edges
        self.num_vertices = _new_num_vertices
        self.num_polygons = _new_num_polygons

        # --- quad edge indices
        self.quad_face_indices = None  # TODO
        self.quad_edge_indices = memoryview((ctypes.c_int * (self.num_edges * 2))()).cast("B").cast("i")
        ptr = OpenMaya.MScriptUtil().asInt2Ptr()
        for i in range(self.num_edges):
            mFnMesh.getEdgeVertices(i, ptr)
            self.quad_edge_indices[i * 2 + 0] = OpenMaya.MScriptUtil.getInt2ArrayItem(ptr, 0, 0)
            self.quad_edge_indices[i * 2 + 1] = OpenMaya.MScriptUtil.getInt2ArrayItem(ptr, 0, 1)

        # --- tri
        tri_counts = OpenMaya.MIntArray()
        tri_indices = OpenMaya.MIntArray()
        mFnMesh.getTriangles(tri_counts, tri_indices)
        self.num_triangles = tri_indices.length() // 3
        self.tri_face_indices = memoryview(array.array("i", tri_indices)).cast("B").cast("i")
        self.tri_edge_indices = self.get_tri_edge_indices(self.tri_face_indices)

        if update_csr:
            self.v2v_offsets, self.v2v_indices = self.get_v2v_adjacency(self.num_vertices, self.tri_edge_indices)
            self.v2f_offsets, self.v2f_indices = self.get_v2f_adjacency(self.num_vertices, self.tri_face_indices)

        return True

    @staticmethod
    def get_tri_edge_indices(tri_indices):
        """
        MFnMesh 不返回三边面的边信息,
        通过输入三角面的点序号信息,计算出三角面的边信息
        TODO python效率太低了, 想办法优化
        Args:
            tri_indices (list|memoryview|array): 三角面点索引, 通过MFnMesh.getTriangles获取
        Return:
            tri_edge_indices (memoryview): 三角面边索引视图
        """
        unique_edges = set()

        for i in range(0, len(tri_indices), 3):
            v0 = tri_indices[i]
            v1 = tri_indices[i + 1]
            v2 = tri_indices[i + 2]

            unique_edges.add((v0, v1) if v0 < v1 else (v1, v0))
            unique_edges.add((v1, v2) if v1 < v2 else (v2, v1))
            unique_edges.add((v2, v0) if v2 < v0 else (v0, v2))

        ctypes_array = (ctypes.c_int * (len(unique_edges) * 2))()

        tri_edge_indices = memoryview(ctypes_array).cast("B").cast("i")

        idx = 0
        for edge in unique_edges:
            tri_edge_indices[idx] = edge[0]
            tri_edge_indices[idx + 1] = edge[1]
            idx += 2

        return tri_edge_indices

    @staticmethod
    def get_v2v_adjacency(num_vertices, edge_indices):
        """
        构建顶点到顶点 (Vertex-to-Vertex) 的 CSR 格式邻接表
        TODO python效率太低了, 想办法优化
        Args:
            num_vertices (int): 模型的顶点总数
            edge_indices (memoryview): 生成的去重边数组
        Return:
            offset_view (memoryview): 偏移量视图
            indices_view (memoryview): 索引视图
        """

        offsets = (ctypes.c_int * (num_vertices + 1))()
        offset_view = memoryview(offsets).cast("B").cast("i")

        for v in edge_indices:
            offset_view[v + 1] += 1

        for i in range(num_vertices):
            offset_view[i + 1] += offset_view[i]

        indices = (ctypes.c_int * offset_view[num_vertices])()
        indices_view = memoryview(indices).cast("B").cast("i")

        cur = offset_view[:-1].tolist()

        for i in range(0, len(edge_indices), 2):
            u, v = edge_indices[i], edge_indices[i + 1]
            # 双向填充
            cur[u] += 1
            indices_view[cur[u] - 1] = v
            cur[v] += 1
            indices_view[cur[v] - 1] = u

        return offset_view, indices_view

    @staticmethod
    def get_v2f_adjacency(num_vertices, tri_face_indices):
        """
        构建顶点到面 (Vertex-to-Face) 的 CSR 格式邻接表
        TODO python效率太低了, 想办法优化
        Args:
            num_vertices (int): 模型的顶点总数
            tri_face_indices (memoryview): 三角面点索引, 通过MFnMesh.getTriangles获取
        Return:
            offset_view (memoryview): 偏移量视图
            indices_view (memoryview): 索引视图
        """

        offsets = (ctypes.c_int * (num_vertices + 1))()
        offset_view = memoryview(offsets).cast("B").cast("i")

        for v in tri_face_indices:
            offset_view[v + 1] += 1

        for i in range(num_vertices):
            offset_view[i + 1] += offset_view[i]

        indices = (ctypes.c_int * offset_view[num_vertices])()
        indices_view = memoryview(indices).cast("B").cast("i")
        cur = offset_view[:-1].tolist()

        for i, v in enumerate(tri_face_indices):
            indices_view[cur[v]] = i // 3
            cur[v] += 1

        return offset_view, indices_view

    def update_fnMesh(self, mFnMesh: OpenMaya.MFnMesh):
        """
        更新mFnMesh

        Args:
            mFnMesh (OpenMaya.MFnMesh): 输入mFnMesh
        Update:
            - `self.mFnMesh`
        """
        self.mFnMesh = mFnMesh

    def update_fnMesh_from_string(self, input_string: str):
        """
        更新mFnMesh

        Args:
            input_string (str): 输入字符串
        Update:
            - `self.mFnMesh`
        """
        try:
            sel = OpenMaya.MGlobal.getSelectionListByName(input_string)
        except RuntimeError as e:
            raise RuntimeError(f"Invalid input_string '{input_string}'.") from e

        dag_path = OpenMaya.MDagPath()
        sel.getDagPath(0, dag_path)
        fnMesh = OpenMaya.MFnMesh(dag_path)
        self.mFnMesh = fnMesh

    @classmethod
    def from_mFnMesh(cls, mFnMesh: OpenMaya.MFnMesh):
        return cls(mFnMesh)

    @classmethod
    def from_string(cls, input_string: str):
        try:
            sel = OpenMaya.MSelectionList()
            sel.add(input_string)
        except RuntimeError as e:
            raise RuntimeError(f"Invalid input_string '{input_string}'.") from e

        dag_path = OpenMaya.MDagPath()
        sel.getDagPath(0, dag_path)
        fnMesh = OpenMaya.MFnMesh(dag_path)
        return cls(fnMesh)

    def __repr__(self) -> str:
        status = "Loaded" if self.position is not None else "Empty"
        return (
            f"<{self.__class__.__name__} [{status}]>\n"
            f"    Vertices:  {self.num_vertices}\n"
            f"    Edges:     {self.num_edges}\n"
            f"    Polygons:  {self.num_polygons}\n"
            f"    Triangles: {self.num_triangles}\n"
            f"    CSR_V2V:   {'Yes' if self.v2v_indices is not None else 'No'}\n"
            f"    CSR_V2F:   {'Yes' if self.v2f_indices is not None else 'No'}\n"
        )


if __name__ == "__main__":
    a = TopologyContext()
    a.update_fnMesh_from_string("pCube1")
    a.update_position()
    a.update_topology()
    print(list(a.position))
    print(list(a.tri_edge_indices))
    print(list(a.v2v_offsets))
    print(list(a.v2v_indices))
    print(a)
