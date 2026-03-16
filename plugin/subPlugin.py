import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui
import maya.api.OpenMayaRender as omr
import ctypes


PLUGIN_VENDOR = "DDDonzy"
PLUGIN_VERSION = "1.0"
PLUGIN_API_VERSION = "Any"


def maya_useNewAPI():
    pass


class RenderData:
    def __init__(self):
        self.line_width = 5.0
        self.point_size = 15.0
        self.draw_faces = True
        self.draw_lines = True
        self.draw_points = True

        # fmt:off
        self.dirty_vertices_pos   = True

        self.dirty_face_colors    = True
        self.dirty_line_colors    = True
        self.dirty_point_colors   = True

        self.dirty_face_indices   = True
        self.dirty_line_indices   = True
        self.dirty_point_indices  = True
        # fmt:on

        self.vertices_pos   = (ctypes.c_float * 9)(00.0, 00.0, 00.0,
                                                   10.0, 00.0, 00.0, 
                                                   05.0, 10.0, 00.0)  # fmt:skip
        self.face_colors    = (ctypes.c_float * 12)(1.0, 0.0, 0.0, 0.1,
                                                    1.0, 0.0, 1.0, 0.1, 
                                                    0.0, 1.0, 0.0, 0.1)  # fmt:skip
        self.line_colors    = (ctypes.c_float * 12)(1.0, 1.0, 1.0, 1.0,
                                                    0.0, 1.0, 1.0, 1.0, 
                                                    0.0, 0.0, 1.0, 1.0)  # fmt:skip
        self.point_colors   = (ctypes.c_float * 12)(0.0, 1.0, 0.0, 1.0,
                                                    1.0, 0.0, 0.0, 1.0, 
                                                    1.0, 1.0, 0.0, 1.0)  # fmt:skip

        self.point_indices = (ctypes.c_uint32 * 3)(0, 1, 2)  # fmt:skip
        self.face_indices  = (ctypes.c_uint32 * 3)(0, 1, 2)  # fmt:skip
        self.line_indices  = (ctypes.c_uint32 * 6)(0, 1,
                                                   1, 2, 
                                                   2, 0)  # fmt:skip


class TriangleShapeUI(omui.MPxSurfaceShapeUI):
    def __init__(self):
        omui.MPxSurfaceShapeUI.__init__(self)

    @classmethod
    def creator(cls):
        return TriangleShapeUI()


# 🌟 修改点 1：基类改为 om2.MPxSurfaceShape，并重命名为 TriangleShape
class TriangleShape(om.MPxSurfaceShape):
    TYPE_ID = om.MTypeId(0x80089)
    NODE_NAME = "triangleShape"
    DRAW_REGISTRANT_ID = "TriangleShapeOverride"
    DRAW_DB_CLASSIFICATION = "drawdb/subscene/triangleShape"

    lineWidthAttr = om.MObject()
    pointSizeAttr = om.MObject()
    drawFacesAttr = om.MObject()
    drawLinesAttr = om.MObject()
    drawPointsAttr = om.MObject()

    # 🌟 伪输出属性，专门用来触发 compute
    outDummyAttr = om.MObject()

    def __init__(self):
        om.MPxSurfaceShape.__init__(self)
        self.render_data = RenderData()  # 挂载数据中心

    @classmethod
    def creator(cls):
        return TriangleShape()

    @classmethod
    def initialize(cls):
        # 🌟 2. 创建真正的 Maya 节点属性，暴露在通道盒里！
        nAttr = om.MFnNumericAttribute()

        # 创建“线宽”属性 (默认 5.0)
        cls.lineWidthAttr = nAttr.create("lineWidth", "lw", om.MFnNumericData.kFloat, 5.0)
        nAttr.keyable = True  # 允许在通道盒显示并做动画
        nAttr.storable = True  # 允许保存在 Maya 文件里
        nAttr.setMin(0.0)  # 最小 1 个像素
        om.MPxNode.addAttribute(cls.lineWidthAttr)

        # 创建“点大小”属性 (默认 15.0)
        cls.pointSizeAttr = nAttr.create("pointSize", "ps", om.MFnNumericData.kFloat, 15.0)
        nAttr.keyable = True
        nAttr.storable = True
        om.MPxNode.addAttribute(cls.pointSizeAttr)

        # 创建“显示面”开关属性 (默认 True)
        cls.drawFacesAttr = nAttr.create("drawFaces", "df", om.MFnNumericData.kBoolean, True)
        nAttr.keyable = True
        nAttr.storable = True
        om.MPxNode.addAttribute(cls.drawFacesAttr)

        # 创建“显示边”开关属性 (默认 True)
        cls.drawLinesAttr = nAttr.create("drawLines", "dl", om.MFnNumericData.kBoolean, True)
        nAttr.keyable = True
        nAttr.storable = True
        om.MPxNode.addAttribute(cls.drawLinesAttr)

        # 创建“显示点”开关属性 (默认 True)
        cls.drawPointsAttr = nAttr.create("drawPoints", "dp", om.MFnNumericData.kBoolean, True)
        nAttr.keyable = True
        nAttr.storable = True
        om.MPxNode.addAttribute(cls.drawPointsAttr)

        # 🌟 3. 创建 Dummy 输出属性并建立依赖图脏数据传播 (Dirty Propagation)
        cls.outDummyAttr = nAttr.create("outDummy", "od", om.MFnNumericData.kInt, 0)
        nAttr.writable = False
        nAttr.storable = False
        nAttr.hidden = True
        om.MPxNode.addAttribute(cls.outDummyAttr)

        # 将所有渲染输入属性与 dummy 输出绑定
        om.MPxNode.attributeAffects(cls.lineWidthAttr, cls.outDummyAttr)
        om.MPxNode.attributeAffects(cls.pointSizeAttr, cls.outDummyAttr)
        om.MPxNode.attributeAffects(cls.drawFacesAttr, cls.outDummyAttr)
        om.MPxNode.attributeAffects(cls.drawLinesAttr, cls.outDummyAttr)
        om.MPxNode.attributeAffects(cls.drawPointsAttr, cls.outDummyAttr)

    def compute(self, plug, dataBlock):
        # 🌟 4. 当属性发生改变被拉取时，触发 compute，在此更新数据中心！
        if plug == TriangleShape.outDummyAttr:
            self.render_data.line_width = dataBlock.inputValue(TriangleShape.lineWidthAttr).asFloat()
            self.render_data.point_size = dataBlock.inputValue(TriangleShape.pointSizeAttr).asFloat()
            self.render_data.draw_faces = dataBlock.inputValue(TriangleShape.drawFacesAttr).asBool()
            self.render_data.draw_lines = dataBlock.inputValue(TriangleShape.drawLinesAttr).asBool()
            self.render_data.draw_points = dataBlock.inputValue(TriangleShape.drawPointsAttr).asBool()

            # 🌟 注意：如果以后你在这里加入了控制坐标/颜色的输入属性，
            # 并在 compute 里修改了 self.render_data.vertices 数组，
            # 你只需要将对应的阀门打开即可！例如：
            # self.render_data.dirty_vertices = True
            # 当前这里只拉取了开关和尺寸，不需要动显存，所以什么都不用标脏。

            # 标记伪输出属性为 Clean
            # dataBlock.outputValue(TriangleShape.outDummyAttr).setInt(1)
            dataBlock.outputValue(TriangleShape.outDummyAttr).setClean()
        else:
            return om.kUnknownParameter

    def isBounded(self):
        return True

    def boundingBox(self):
        return om.MBoundingBox(om.MPoint(-100, -100, -100), om.MPoint(100, 100, 100))


class TriangleOverride(omr.MPxSubSceneOverride):
    def __init__(self, obj):
        super(TriangleOverride, self).__init__(obj)
        self.node_obj = obj

        self.item_name_face = "my_triangle_face"  # 面的名字
        self.item_name_line = "my_triangle_line"  # 线的名字
        self.item_name_point = "my_triangle_point"  # 点的名字

        self.vertex_buffer = None
        self.color_buffer_face = None
        self.index_buffer_face = None
        self.vertex_buffer_array_face = None

        # 为线和点准备独立的显存池
        self.color_buffer_line = None
        self.color_buffer_point = None
        self.index_buffer_line = None
        self.index_buffer_point = None
        self.vertex_buffer_array_line = None
        self.vertex_buffer_array_point = None

    @classmethod
    def creator(cls, obj):
        return TriangleOverride(obj)

    def supportedDrawAPIs(self):
        return omr.MRenderer.kAllDevices

    def requiresUpdate(self, container, frameContext):
        return True

    def update(self, container, frameContext):
        render_item_face = container.find(self.item_name_face)
        render_item_line = container.find(self.item_name_line)
        render_item_point = container.find(self.item_name_point)

        shader_mgr = omr.MRenderer.getShaderManager()

        # ==========================================
        # 🌟 1. 面
        # ==========================================
        if render_item_face is None:
            render_item_face = omr.MRenderItem.create(self.item_name_face, omr.MRenderItem.MaterialSceneItem, omr.MGeometry.kTriangles)
            render_item_face.setSelectionMask(om.MSelectionMask("polymesh"))
            render_item_face.setDrawMode(omr.MGeometry.kShaded | omr.MGeometry.kTextured)
            shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVSolidShader).clone()
            render_item_face.setShader(shader)
            container.add(render_item_face)

        # ==========================================
        # 🌟 2. 线
        # ==========================================
        if render_item_line is None:
            # 关键：使用 DecorationItem，Maya射线会直接无视它！
            render_item_line = omr.MRenderItem.create(self.item_name_line, omr.MRenderItem.DecorationItem, omr.MGeometry.kLines)
            render_item_line.setDrawMode(omr.MGeometry.kAll)
            render_item_line.setDepthPriority(omr.MRenderItem.sActiveWireDepthPriority)  # 防闪烁
            # 🌟 4. 极其关键：必须调用 clone()！
            # 否则你改了这里的线宽，Maya 里所有的线都会变成这么粗！
            shader_line = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVThickLineShader).clone()

            render_item_line.setShader(shader_line)
            container.add(render_item_line)

        # ==========================================
        # 🌟 3. 点
        # ==========================================
        if render_item_point is None:
            # 关键：使用 DecorationItem！
            render_item_point = omr.MRenderItem.create(self.item_name_point, omr.MRenderItem.DecorationItem, omr.MGeometry.kPoints)
            render_item_point.setDrawMode(omr.MGeometry.kAll)
            render_item_point.setDepthPriority(omr.MRenderItem.sActiveWireDepthPriority)
            # 🌟 5. 同样必须 clone() 胖点着色器
            shader_point = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVFatPointShader).clone()
            render_item_point.setShader(shader_point)
            container.add(render_item_point)

        # ==========================================================
        # 🌟 阶段 A：提取渲染状态
        # ==========================================================
        # 1. 轻量级拉取 dummy 属性触发 DG
        om.MPlug(self.node_obj, TriangleShape.outDummyAttr).asInt()
        render_data: RenderData = om.MFnDependencyNode(self.node_obj).userNode().render_data

        # ==========================================
        # 显存开辟
        # ==========================================
        if self.vertex_buffer is None:
            # 1. 共享的顶点坐标显存池
            self.vertex_buffer = omr.MVertexBuffer(omr.MVertexBufferDescriptor("", omr.MGeometry.kPosition, omr.MGeometry.kFloat, 3))

            # 2. 面 (Face) 专属显存及阵列打包
            self.color_buffer_face = omr.MVertexBuffer(omr.MVertexBufferDescriptor("", omr.MGeometry.kColor, omr.MGeometry.kFloat, 4))
            self.index_buffer_face = omr.MIndexBuffer(omr.MGeometry.kUnsignedInt32)
            self.vertex_buffer_array_face = omr.MVertexBufferArray()
            self.vertex_buffer_array_face.append(self.vertex_buffer, "")
            self.vertex_buffer_array_face.append(self.color_buffer_face, "")

            # 3. 线 (Line) 专属显存及阵列打包
            self.color_buffer_line = omr.MVertexBuffer(omr.MVertexBufferDescriptor("", omr.MGeometry.kColor, omr.MGeometry.kFloat, 4))
            self.index_buffer_line = omr.MIndexBuffer(omr.MGeometry.kUnsignedInt32)
            self.vertex_buffer_array_line = omr.MVertexBufferArray()
            self.vertex_buffer_array_line.append(self.vertex_buffer, "")
            self.vertex_buffer_array_line.append(self.color_buffer_line, "")

            # 4. 点 (Point) 专属显存及阵列打包
            self.color_buffer_point = omr.MVertexBuffer(omr.MVertexBufferDescriptor("", omr.MGeometry.kColor, omr.MGeometry.kFloat, 4))
            self.index_buffer_point = omr.MIndexBuffer(omr.MGeometry.kUnsignedInt32)
            self.vertex_buffer_array_point = omr.MVertexBufferArray()
            self.vertex_buffer_array_point.append(self.vertex_buffer, "")
            self.vertex_buffer_array_point.append(self.color_buffer_point, "")

            # 🌟 核心修复：如果是全新开辟的显存(如撤销删除重建Override、或开启新视口时)
            # 此时的 GPU 缓冲是空的！必须强制将节点的阀门全部打开，保证基础数据能拷入新缓冲！
            render_data.dirty_vertices_pos = True
            render_data.dirty_face_colors = True
            render_data.dirty_line_colors = True
            render_data.dirty_point_colors = True
            render_data.dirty_face_indices = True
            render_data.dirty_line_indices = True
            render_data.dirty_point_indices = True

        # ==========================================================
        # 更新顶点坐标缓冲 (面、线、点共享同一套动画变形)
        # ==========================================================

        # 仅当对应的 dirty 为 True 时，才向 GPU 拷贝！
        # --- 点位置显存同步 ---
        if render_data.dirty_vertices_pos:
            vertex_count = len(render_data.vertices_pos) // 3
            vertex_addr = self.vertex_buffer.acquire(vertex_count, True)
            ctypes.memmove(vertex_addr, ctypes.addressof(render_data.vertices_pos), ctypes.sizeof(render_data.vertices_pos))
            self.vertex_buffer.commit(vertex_addr)
            render_data.dirty_vertices_pos = False

        # --- 面专属显存同步 ---
        if render_data.dirty_face_colors:
            color_count = len(render_data.face_colors) // 4
            color_addr_face = self.color_buffer_face.acquire(color_count, True)
            ctypes.memmove(color_addr_face, ctypes.addressof(render_data.face_colors), ctypes.sizeof(render_data.face_colors))
            self.color_buffer_face.commit(color_addr_face)
            render_data.dirty_face_colors = False

        if render_data.dirty_face_indices:
            index_count = len(render_data.face_indices)
            index_addr_face = self.index_buffer_face.acquire(index_count, True)
            ctypes.memmove(index_addr_face, ctypes.addressof(render_data.face_indices), ctypes.sizeof(render_data.face_indices))
            self.index_buffer_face.commit(index_addr_face)
            render_data.dirty_face_indices = False

        # --- 线专属显存同步 ---
        if render_data.dirty_line_colors:
            color_count = len(render_data.line_colors) // 4
            color_addr_line = self.color_buffer_line.acquire(color_count, True)
            ctypes.memmove(color_addr_line, ctypes.addressof(render_data.line_colors), ctypes.sizeof(render_data.line_colors))
            self.color_buffer_line.commit(color_addr_line)
            render_data.dirty_line_colors = False

        if render_data.dirty_line_indices:
            index_count = len(render_data.line_indices)
            index_addr_line = self.index_buffer_line.acquire(index_count, True)
            ctypes.memmove(index_addr_line, ctypes.addressof(render_data.line_indices), ctypes.sizeof(render_data.line_indices))
            self.index_buffer_line.commit(index_addr_line)
            render_data.dirty_line_indices = False

        # --- 点专属显存同步 ---
        if render_data.dirty_point_colors:
            color_count = len(render_data.point_colors) // 4
            color_addr_point = self.color_buffer_point.acquire(color_count, True)
            ctypes.memmove(color_addr_point, ctypes.addressof(render_data.point_colors), ctypes.sizeof(render_data.point_colors))
            self.color_buffer_point.commit(color_addr_point)
            render_data.dirty_point_colors = False

        if render_data.dirty_point_indices:
            index_count = len(render_data.point_indices)
            index_addr_point = self.index_buffer_point.acquire(index_count, True)
            ctypes.memmove(index_addr_point, ctypes.addressof(render_data.point_indices), ctypes.sizeof(render_data.point_indices))
            self.index_buffer_point.commit(index_addr_point)
            render_data.dirty_point_indices = False

        # ==========================================================
        # 🌟 阶段 B：应用渲染状态到 RenderItem
        # ==========================================================
        # 动态修改专属着色器参数及渲染开关！
        if render_item_face:
            render_item_face.enable(render_data.draw_faces)

        if render_item_line:
            render_item_line.getShader().setParameter("lineWidth", (render_data.line_width, render_data.line_width))
            render_item_line.enable(render_data.draw_lines)

        if render_item_point:
            render_item_point.getShader().setParameter("pointSize", (render_data.point_size, render_data.point_size))
            render_item_point.enable(render_data.draw_points)

        # 终极绑定通知
        bbox = om.MBoundingBox(om.MPoint(-100, -100, -100), om.MPoint(100, 100, 100))
        self.setGeometryForRenderItem(render_item_face, self.vertex_buffer_array_face, self.index_buffer_face, bbox)
        self.setGeometryForRenderItem(render_item_line, self.vertex_buffer_array_line, self.index_buffer_line, bbox)
        self.setGeometryForRenderItem(render_item_point, self.vertex_buffer_array_point, self.index_buffer_point, bbox)

    def areControlsAllocated(self):
        return False

    def freeControls(self):
        pass


def initializePlugin(mObject: om.MObject):
    plugin: om.MFnPlugin = om.MFnPlugin(
        mObject,
        PLUGIN_VENDOR,
        PLUGIN_VERSION,
        PLUGIN_API_VERSION,
    )

    # 🌟 修改点 4：改为使用 registerShape，传入 UI 类和表面形状类型
    plugin.registerShape(
        TriangleShape.NODE_NAME,
        TriangleShape.TYPE_ID,
        TriangleShape.creator,
        TriangleShape.initialize,
        TriangleShapeUI.creator,
        TriangleShape.DRAW_DB_CLASSIFICATION,
    )

    omr.MDrawRegistry.registerSubSceneOverrideCreator(
        TriangleShape.DRAW_DB_CLASSIFICATION,
        TriangleShape.DRAW_REGISTRANT_ID,
        TriangleOverride.creator,
    )


def uninitializePlugin(mObject):
    plugin: om.MFnPlugin = om.MFnPlugin(mObject)

    omr.MDrawRegistry.deregisterSubSceneOverrideCreator(
        TriangleShape.DRAW_DB_CLASSIFICATION,
        TriangleShape.DRAW_REGISTRANT_ID,
    )
    # 卸载时注销对应的 ID
    plugin.deregisterNode(TriangleShape.TYPE_ID)
