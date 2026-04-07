from __future__ import annotations

import maya.OpenMaya as OpenMaya  # type: ignore


class MeshTopologyContext:
    """
    CSR Data:
    ```
    ===================================================================================
    Offsets:
        Index:    [0]            [1]                     [2]            [3] (End bound)
        Data:      0              2                       5              6
                   |              |                       |              |
                   |              |                       |              |
    Indices:       |              |                       |              |
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

    def __init__(self, mFnMesh: OpenMaya.MFnMesh = None):
        # fmt:off
        if mFnMesh:
            self.mFnMesh:OpenMaya.MFnMesh = mFnMesh

        self.num_vertices : int = 0
        self.num_edges    : int = 0
        self.num_polygons : int = 0
        self.num_triangles: int = 0
        # position
        self.position: memoryview = None
        # 基础拓扑
        self.tri_face_indices : memoryview = None
        self.tri_edge_indices : memoryview = None
        self.quad_face_indices: memoryview = None
        self.quad_edge_indices: memoryview = None
        # v2v CSR
        self.v2v_offsets: memoryview = None
        self.v2v_indices: memoryview = None
        # v2f CSR
        self.v2f_offsets: memoryview = None
        self.v2f_indices: memoryview = None
        # fmt:on

    def clear(self):
        # fmt:off
        self.num_vertices : int = 0
        self.num_edges    : int = 0
        self.num_polygons : int = 0
        self.num_triangles: int = 0
        # position
        self.position: memoryview = None
        # 基础拓扑
        self.tri_face_indices : memoryview = None
        self.tri_edge_indices : memoryview = None
        self.quad_face_indices: memoryview = None
        self.quad_edge_indices: memoryview = None
        # v2v CSR
        self.v2v_offsets: memoryview = None
        self.v2v_indices: memoryview = None
        # v2f CSR
        self.v2f_offsets: memoryview = None
        self.v2f_indices: memoryview = None
        # fmt:on

    def update_fnMesh():
        raise NotImplementedError("not supped")

    def update_position(self):
        raise NotImplementedError("not supped")

    def update_csr(self):
        raise NotImplementedError("not supped")

    def update_topology(self, update_csr=False):
        raise NotImplementedError("not supped")


a = MeshTopologyContext()
