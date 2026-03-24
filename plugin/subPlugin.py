import ctypes
from contextlib import contextmanager

import maya.api.OpenMaya as om
import maya.api.OpenMayaUI as omui
import maya.api.OpenMayaRender as omr

from gskin.src._cRegistry import SkinRegistry
from gskin.src import cColorCython as cColor
from gskin.src._cProfilerCython import MayaNativeProfiler, maya_profile

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gskin.src.cSkinDeform import CythonSkinDeformer

PLUGIN_VENDOR = "DDDonzy"
PLUGIN_VERSION = "1.0"
PLUGIN_API_VERSION = "Any"


def maya_useNewAPI():
    pass


@contextmanager
def gpu_write_session(gpu_buffer, element_count, dimension=4):
    if not gpu_buffer or element_count <= 0:
        yield None
        return

    gpu_ptr = gpu_buffer.acquire(element_count, True)
    try:
        if gpu_ptr:
            float_count = element_count * dimension
            ArrayType = ctypes.c_float * float_count
            buffer_array = ArrayType.from_address(int(gpu_ptr))
            cv_view = memoryview(buffer_array).cast("B").cast("f", shape=(element_count, dimension))
            yield cv_view
        else:
            yield None
    finally:
        if gpu_ptr:
            gpu_buffer.commit(gpu_ptr)


class RenderData:
    def __init__(self):
        # fmt:off
        self.line_width  = 1
        self.point_size  = 4

        self.draw_faces  = True
        self.draw_lines  = True
        self.draw_points = True

        self.draw_default_faces  = False
        self.draw_default_lines  = False
        self.draw_default_points = False

        # Default Solid Colors
        self.default_face_color    = (0.5, 0.5, 0.5, 1.0) # 单色面色
        self.default_line_color    = (0.0, 1.0, 0.0, 1.0) # 单色线色
        self.default_point_color   = (0.0, 0.0, 1.0, 1.0) # 单色点色

        self.wire_color            = (0.0, 1.0, 1.0, 1.0) # 彩色线色
        self.vertex_color          = (1.0, 0.0, 0.0, 1.0) # 彩色点色
        self.mask_remap_a_color    = (0.1, 0.1, 0.1, 1.0) # 遮罩渐变色 A
        self.mask_remap_b_color    = (0.1, 1.0, 0.1, 1.0) # 遮罩渐变色 B
        self.weights_remap_a_color = (0.0, 0.0, 0.0, 1.0) # 权重渐变色 A
        self.weights_remap_b_color = (1.0, 1.0, 1.0, 1.0) # 权重渐变色 B
        self.brush_remap_a_color   = (1.0, 0.0, 0.0, 1.0) # 笔刷渐变色 A
        self.brush_remap_b_color   = (1.0, 1.0, 0.0, 1.0) # 笔刷渐变色 B

        # dirty
        self.dirty_vertices_pos   = True

        self.dirty_face_indices   = True
        self.dirty_line_indices   = True
        self.dirty_point_indices  = True

        self.dirty_face_colors    = True
        self.dirty_line_colors    = True
        self.dirty_point_colors   = True

        # 🌟 架构升级:存储 compute 阶段拿到的计算上下文,供 update 阶段的 GPU 直写使用
        self.vtx_count = 0
        self.render_mode = 0
        self.is_mask = False
        self.paint_weights_view = None
        self.brush_hit_indices = None
        self.brush_hit_weights = None
        self.brush_hit_count = 0

        """
        绘制一个普通的三角面
        下面是需要用到的参数
        在这里作为渲染的初始数据,
        后续开发替换数据即可。

            - point_indices (vtx_count): 点索引信息
            - face_indices (tri_count * 3): 面索引信息
            - line_indices (line_count * 2): 线索引信息

            - vertices_pos (vtx_count * 3): 位置信息
            - face_colors (vtx_count * 4):  面色信息
            - line_colors (vtx_count * 4):  线色信息
            - point_colors (vtx_count * 4): 点色信息


        """
        # 位置 buffer
        self.vertices_pos   = (ctypes.c_float * 9 )(00.0, 00.0, 00.0,
                                                    10.0, 00.0, 00.0, 
                                                    05.0, 10.0, 00.0)

        # 绘制的点 indices, 从 vertices_pos 读取位置信息
        self.point_indices = (ctypes.c_uint32 * 3)(0, 1, 2) 
        # 绘制的三角面 indices, 从 vertices_pos 读取位置信息,数组是 tri_count * 3
        self.face_indices  = (ctypes.c_uint32 * 3)(0, 1, 2)
        # 绘制的边 indices, 从 vertices_pos 读取位置信息,数组是 line_count * 2
        self.line_indices  = (ctypes.c_uint32 * 6)(0, 1,
                                                   1, 2, 
                                                   2, 0)

        # 面色 buffer
        self.face_colors    = (ctypes.c_float * 12)(1.0, 0.0, 0.0, 0.1,
                                                    1.0, 0.0, 1.0, 0.1, 
                                                    0.0, 1.0, 0.0, 0.1) 
        # 线色 buffer
        self.line_colors    = (ctypes.c_float * 12)(1.0, 1.0, 1.0, 1.0,
                                                    0.0, 1.0, 1.0, 1.0, 
                                                    0.0, 0.0, 1.0, 1.0)  
        # 点色 buffer
        self.point_colors   = (ctypes.c_float * 12)(0.0, 1.0, 0.0, 1.0,
                                                    1.0, 0.0, 0.0, 1.0, 
                                                    1.0, 1.0, 0.0, 1.0) 
        # fmt:on


class TriangleShapeUI(omui.MPxSurfaceShapeUI):
    def __init__(self):
        omui.MPxSurfaceShapeUI.__init__(self)

    @classmethod
    def creator(cls):
        return TriangleShapeUI()


# 🌟 修改点 1:基类改为 om2.MPxSurfaceShape,并重命名为 TriangleShape
class TriangleShape(om.MPxSurfaceShape):
    TYPE_ID = om.MTypeId(0x80089)
    NODE_NAME = "triangleShape"
    DRAW_REGISTRANT_ID = "TriangleShapeOverride"
    DRAW_DB_CLASSIFICATION = "drawdb/subscene/triangleShape"

    # region
    paintLayerAttr = om.MObject()
    paintInfluenceAttr = om.MObject()
    paintMaskAttr = om.MObject()
    renderModeAttr = om.MObject()
    inMeshAttr = om.MObject()

    lineWidthAttr = om.MObject()
    pointSizeAttr = om.MObject()
    drawFacesAttr = om.MObject()
    drawLinesAttr = om.MObject()
    drawPointsAttr = om.MObject()
    wireColorAttr = om.MObject()
    vertexColorAttr = om.MObject()
    maskRemapAColorAttr = om.MObject()
    maskRemapBColorAttr = om.MObject()
    weightsRemapAColorAttr = om.MObject()
    weightsRemapBColorAttr = om.MObject()
    brushRemapAColorAttr = om.MObject()
    brushRemapBColorAttr = om.MObject()

    # 🌟 Default Solid Items Attributes
    defaultDrawFacesAttr = om.MObject()
    defaultDrawLinesAttr = om.MObject()
    defaultDrawPointsAttr = om.MObject()
    defaultFaceColorAttr = om.MObject()
    defaultLineColorAttr = om.MObject()
    defaultPointColorAttr = om.MObject()

    # 🌟 伪输出属性,专门用来触发 compute
    outDummyAttr = om.MObject()
    # endregion

    def __init__(self):
        om.MPxSurfaceShape.__init__(self)
        self.render_data = RenderData()  # 挂载数据中心
        # 预先储存好包围盒,避免每帧重复创建对象
        self._boundingBox = om.MBoundingBox(om.MPoint(-100, -100, -100), om.MPoint(100, 100, 100))

    @classmethod
    def creator(cls):
        return TriangleShape()

    @classmethod
    def initialize(cls):
        # 🌟 2. 创建真正的 Maya 节点属性,暴露在通道盒里
        nAttr = om.MFnNumericAttribute()
        eAttr = om.MFnEnumAttribute()
        tAttr = om.MFnTypedAttribute()
        # 🌟 3. 创建 Dummy 输出属性并建立依赖图脏数据传播 (Dirty Propagation)
        TriangleShape.outDummyAttr = nAttr.create("outDummy", "od", om.MFnNumericData.kInt, 0)
        nAttr.writable = False
        nAttr.storable = False
        nAttr.hidden = True
        TriangleShape.addAttribute(TriangleShape.outDummyAttr)

        # region Input Attr
        # 创建“输入网格”属性,用于替代以前的 message 连接查找 cSkinDeform
        TriangleShape.inMeshAttr = tAttr.create("inputMesh", "ipm", om.MFnData.kMesh)
        tAttr.storable = False
        tAttr.writable = True
        TriangleShape.addAttribute(TriangleShape.inMeshAttr)
        # endregion

        # region Render Attr
        # 渲染模式
        TriangleShape.renderModeAttr = eAttr.create("renderMode", "rm", 0)
        eAttr.addField("Alpha", 0)
        eAttr.addField("Heatmap", 1)
        eAttr.channelBox = True
        TriangleShape.addAttribute(TriangleShape.renderModeAttr)

        # 创建“线宽”属性 (默认 5.0)
        TriangleShape.lineWidthAttr = nAttr.create("lineWidth", "lw", om.MFnNumericData.kFloat, 1.0)
        nAttr.keyable = True  # 允许在通道盒显示并做动画
        nAttr.setMin(0.0)  # 最小 1 个像素
        TriangleShape.addAttribute(TriangleShape.lineWidthAttr)

        # 创建“点大小”属性 (默认 15.0)
        TriangleShape.pointSizeAttr = nAttr.create("pointSize", "ps", om.MFnNumericData.kFloat, 5.0)
        nAttr.keyable = True
        TriangleShape.addAttribute(TriangleShape.pointSizeAttr)

        # 创建“显示面”开关属性 (默认 True)
        TriangleShape.drawFacesAttr = nAttr.create("drawFaces", "df", om.MFnNumericData.kBoolean, True)
        nAttr.keyable = True
        TriangleShape.addAttribute(TriangleShape.drawFacesAttr)

        # 创建“显示边”开关属性 (默认 True)
        TriangleShape.drawLinesAttr = nAttr.create("drawLines", "dl", om.MFnNumericData.kBoolean, True)
        nAttr.keyable = True
        TriangleShape.addAttribute(TriangleShape.drawLinesAttr)

        # 创建“显示点”开关属性 (默认 True)
        TriangleShape.drawPointsAttr = nAttr.create("drawPoints", "dp", om.MFnNumericData.kBoolean, True)
        nAttr.keyable = True
        TriangleShape.addAttribute(TriangleShape.drawPointsAttr)

        # 创建 Default 渲染项开关
        TriangleShape.defaultDrawFacesAttr = nAttr.create("defaultDrawFaces", "ddf", om.MFnNumericData.kBoolean, False)
        nAttr.keyable = True
        nAttr.storable = True
        TriangleShape.addAttribute(TriangleShape.defaultDrawFacesAttr)

        TriangleShape.defaultDrawLinesAttr = nAttr.create("defaultDrawLines", "ddl", om.MFnNumericData.kBoolean, False)
        nAttr.keyable = True
        TriangleShape.addAttribute(TriangleShape.defaultDrawLinesAttr)

        TriangleShape.defaultDrawPointsAttr = nAttr.create("defaultDrawPoints", "ddp", om.MFnNumericData.kBoolean, False)
        nAttr.keyable = True
        TriangleShape.addAttribute(TriangleShape.defaultDrawPointsAttr)
        # endregion

        # region Color Attr
        def add_color(long_name, short_name, default_rgb):
            attr = nAttr.createColor(long_name, short_name)
            nAttr.default = default_rgb
            nAttr.storable = True
            nAttr.keyable = True
            TriangleShape.addAttribute(attr)
            return attr

        # New color attributes
        TriangleShape.wireColorAttr = add_color("renderWireColor", "wcl", (0.0, 1.0, 1.0))
        TriangleShape.vertexColorAttr = add_color("renderVertexColor", "vcl", (1.0, 0.0, 0.0))
        TriangleShape.maskRemapAColorAttr = add_color("maskRemapAColor", "mra", (0.1, 0.1, 0.1))
        TriangleShape.maskRemapBColorAttr = add_color("maskRemapBColor", "mrb", (0.1, 1.0, 0.1))
        TriangleShape.weightsRemapAColorAttr = add_color("weightsRemapAColor", "wra", (0.0, 0.0, 0.0))
        TriangleShape.weightsRemapBColorAttr = add_color("weightsRemapBColor", "wrb", (1.0, 1.0, 1.0))
        TriangleShape.brushRemapAColorAttr = add_color("brushRemapAColor", "bra", (1.0, 0.0, 0.0))
        TriangleShape.brushRemapBColorAttr = add_color("brushRemapBColor", "brb", (1.0, 1.0, 0.0))

        # New Default color attributes
        TriangleShape.defaultFaceColorAttr = add_color("defaultFaceColor", "dfc", (0.5, 0.5, 0.5))
        TriangleShape.defaultLineColorAttr = add_color("defaultLineColor", "dlc", (0.0, 1.0, 0.0))
        TriangleShape.defaultPointColorAttr = add_color("defaultPointColor", "dpc", (0.0, 0.0, 1.0))
        # endregion

        # region AttrAffect
        TriangleShape.attributeAffects(TriangleShape.wireColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.vertexColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.maskRemapAColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.maskRemapBColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.weightsRemapAColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.weightsRemapBColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.brushRemapAColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.brushRemapBColorAttr, TriangleShape.outDummyAttr)

        # 将所有渲染输入属性与 dummy 输出绑定
        TriangleShape.attributeAffects(TriangleShape.inMeshAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.renderModeAttr, TriangleShape.outDummyAttr)

        TriangleShape.attributeAffects(TriangleShape.lineWidthAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.pointSizeAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.drawFacesAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.drawLinesAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.wireColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.vertexColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.maskRemapAColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.maskRemapBColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.weightsRemapAColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.weightsRemapBColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.brushRemapAColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.brushRemapBColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.drawPointsAttr, TriangleShape.outDummyAttr)

        TriangleShape.attributeAffects(TriangleShape.defaultDrawFacesAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.defaultDrawLinesAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.defaultDrawPointsAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.defaultFaceColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.defaultLineColorAttr, TriangleShape.outDummyAttr)
        TriangleShape.attributeAffects(TriangleShape.defaultPointColorAttr, TriangleShape.outDummyAttr)
        # endregion

    def compute(self, plug, dataBlock):
        with MayaNativeProfiler("render-compute", 1):
            with MayaNativeProfiler("render-compute-get-attr", 2):
                # fmt:off
                self.render_data.line_width                      = dataBlock.inputValue(TriangleShape.lineWidthAttr).asFloat()
                self.render_data.point_size                      = dataBlock.inputValue(TriangleShape.pointSizeAttr).asFloat()

                self.render_data.draw_faces                      = dataBlock.inputValue(TriangleShape.drawFacesAttr).asBool()
                self.render_data.draw_lines                      = dataBlock.inputValue(TriangleShape.drawLinesAttr).asBool()
                self.render_data.draw_points                     = dataBlock.inputValue(TriangleShape.drawPointsAttr).asBool()

                self.render_data.draw_default_faces              = dataBlock.inputValue(TriangleShape.defaultDrawFacesAttr).asBool()
                self.render_data.draw_default_lines              = dataBlock.inputValue(TriangleShape.defaultDrawLinesAttr).asBool()
                self.render_data.draw_default_points             = dataBlock.inputValue(TriangleShape.defaultDrawPointsAttr).asBool()

                self.render_data.wire_color            = (*dataBlock.inputValue(TriangleShape.wireColorAttr).asFloat3(), 1.0)
                self.render_data.vertex_color          = (*dataBlock.inputValue(TriangleShape.vertexColorAttr).asFloat3(), 1.0)

                self.render_data.mask_remap_a_color    = (*dataBlock.inputValue(TriangleShape.maskRemapAColorAttr).asFloat3(), 1.0)
                self.render_data.mask_remap_b_color    = (*dataBlock.inputValue(TriangleShape.maskRemapBColorAttr).asFloat3(), 1.0)

                self.render_data.weights_remap_a_color = (*dataBlock.inputValue(TriangleShape.weightsRemapAColorAttr).asFloat3(), 1.0)
                self.render_data.weights_remap_b_color = (*dataBlock.inputValue(TriangleShape.weightsRemapBColorAttr).asFloat3(), 1.0)

                self.render_data.brush_remap_a_color   = (*dataBlock.inputValue(TriangleShape.brushRemapAColorAttr).asFloat3(), 1.0)
                self.render_data.brush_remap_b_color   = (*dataBlock.inputValue(TriangleShape.brushRemapBColorAttr).asFloat3(), 1.0)

                self.render_data.default_face_color    = (*dataBlock.inputValue(TriangleShape.defaultFaceColorAttr).asFloat3(), 1.0)
                self.render_data.default_line_color    = (*dataBlock.inputValue(TriangleShape.defaultLineColorAttr).asFloat3(), 1.0)
                self.render_data.default_point_color   = (*dataBlock.inputValue(TriangleShape.defaultPointColorAttr).asFloat3(), 1.0)
                # fmt:on

            # 2. 扁平化数据同步与渲染调用
            with MayaNativeProfiler("render-compute-get-cSkinData", 3):
                self._update_from_cSkin(dataBlock)
            with MayaNativeProfiler("render-compute-get-setClean", 4):
                dataBlock.outputValue(TriangleShape.outDummyAttr).setClean()

    def _update_from_cSkin(self, dataBlock):
        """管线接线员:获取上游内存并刷新渲染上下文数据,使用前置判定消灭深层嵌套"""

        with MayaNativeProfiler("render-update_cSkin_datablock", 1):
            dataBlock.inputValue(TriangleShape.inMeshAttr)

        # region Get cSkin
        cSkin_plug: om.MPlug = om.MPlug(self.thisMObject(), TriangleShape.inMeshAttr)
        if not cSkin_plug.isConnected:
            return
        conns = cSkin_plug.connectedTo(True, False)
        if not conns:
            return
        cSkin_node = conns[0].node()
        cSkin: CythonSkinDeformer = SkinRegistry.get_instance_by_api2(cSkin_node)
        # endregion

        if not cSkin or getattr(cSkin, "mesh_context", None) is None:
            return

        mesh_ctx = cSkin.mesh_context
        vtx_count = mesh_ctx.vertex_count
        if vtx_count <= 0:
            return

        render_data = self.render_data
        render_data.vtx_count = vtx_count  # 存下顶点数供 update 使用

        if mesh_ctx.vertex_positions:
            render_data.vertices_pos = mesh_ctx.vertex_positions.ctypes

        if mesh_ctx.triangle_indices:
            render_data.face_indices = mesh_ctx.triangle_indices.ctypes

        if mesh_ctx.quad_edge_indices:
            render_data.line_indices = mesh_ctx.quad_edge_indices.ctypes

        # point_indices
        if not hasattr(render_data, "point_indices") or len(render_data.point_indices) != vtx_count:
            render_data.point_indices = (ctypes.c_uint32 * vtx_count)(*range(vtx_count))

        # 获取要渲染的权重信息
        render_data.render_mode = dataBlock.inputValue(TriangleShape.renderModeAttr).asInt()
        _, render_data.is_mask, _, render_data.paint_weights_view = cSkin.get_active_paint_weights()

        # 获取笔刷上下文
        brush_ctx = getattr(cSkin, "brush_context", None)
        if brush_ctx and brush_ctx.is_valid:
            render_data.brush_hit_indices = brush_ctx.hit_indices
            render_data.brush_hit_weights = brush_ctx.hit_weights
            render_data.brush_hit_count = brush_ctx.hit_count
        else:
            render_data.brush_hit_count = 0

        # 全面打上脏标,通知 update 阶段去向 GPU 借用显存并直写
        render_data.dirty_face_colors = True
        render_data.dirty_line_colors = True
        render_data.dirty_point_colors = True

    def isBounded(self):
        return True

    def boundingBox(self):
        return self._boundingBox


class TriangleOverride(omr.MPxSubSceneOverride):
    def __init__(self, obj):
        super().__init__(obj)
        self.node_obj = obj

        self.item_name_face = "my_triangle_face"
        self.item_name_line = "my_triangle_line"
        self.item_name_point = "my_triangle_point"

        self.item_name_default_face = "my_default_solid_face"
        self.item_name_default_line = "my_default_solid_line"
        self.item_name_default_point = "my_default_solid_point"

        self.vertex_buffer: omr.MVertexBuffer = None

        self.color_buffer_face: omr.MVertexBuffer = None
        self.index_buffer_face: omr.MIndexBuffer = None
        self.vertex_buffer_array_face: omr.MVertexBufferArray = None

        # 为线和点准备独立的显存池
        self.color_buffer_line: omr.MVertexBuffer = None
        self.color_buffer_point: omr.MVertexBuffer = None
        self.index_buffer_line: omr.MIndexBuffer = None
        self.index_buffer_point: omr.MIndexBuffer = None
        self.vertex_buffer_array_line: omr.MVertexBufferArray = None
        self.vertex_buffer_array_point: omr.MVertexBufferArray = None

        self.vertex_buffer_array_default: omr.MVertexBufferArray = None

    @classmethod
    def creator(cls, obj):
        return TriangleOverride(obj)

    def supportedDrawAPIs(self):
        return omr.MRenderer.kAllDevices

    def requiresUpdate(self, container, frameContext):
        return True

    def _init_render_items(self, container, shader_mgr):
        """渲染物品初始装载器"""
        # 面
        render_item_face = container.find(self.item_name_face)
        if render_item_face is None:
            render_item_face = omr.MRenderItem.create(self.item_name_face, omr.MRenderItem.MaterialSceneItem, omr.MGeometry.kTriangles)
            render_item_face.setSelectionMask(om.MSelectionMask("polymesh"))
            render_item_face.setDrawMode(omr.MGeometry.kShaded | omr.MGeometry.kTextured)
            shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVSolidShader).clone()
            render_item_face.setShader(shader)
            container.add(render_item_face)

        # 线
        render_item_line = container.find(self.item_name_line)
        if render_item_line is None:
            render_item_line = omr.MRenderItem.create(self.item_name_line, omr.MRenderItem.DecorationItem, omr.MGeometry.kLines)
            render_item_line.setDrawMode(omr.MGeometry.kAll)
            render_item_line.setDepthPriority(omr.MRenderItem.sActiveWireDepthPriority)  # 防闪烁
            shader_line = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVThickLineShader).clone()
            render_item_line.setShader(shader_line)
            container.add(render_item_line)

        # 点
        render_item_point = container.find(self.item_name_point)
        if render_item_point is None:
            render_item_point = omr.MRenderItem.create(self.item_name_point, omr.MRenderItem.DecorationItem, omr.MGeometry.kPoints)
            render_item_point.setDrawMode(omr.MGeometry.kAll)
            render_item_point.setDepthPriority(omr.MRenderItem.sActiveWireDepthPriority)
            shader_point = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVFatPointShader).clone()
            render_item_point.setShader(shader_point)
            container.add(render_item_point)

        # Default 纯色面
        render_item_default_face = container.find(self.item_name_default_face)
        if render_item_default_face is None:
            render_item_default_face = omr.MRenderItem.create(self.item_name_default_face, omr.MRenderItem.MaterialSceneItem, omr.MGeometry.kTriangles)
            render_item_default_face.setSelectionMask(om.MSelectionMask("polymesh"))
            render_item_default_face.setDrawMode(omr.MGeometry.kShaded | omr.MGeometry.kTextured)
            shader_default_face = shader_mgr.getStockShader(omr.MShaderManager.k3dSolidShader).clone()
            render_item_default_face.setShader(shader_default_face)
            container.add(render_item_default_face)

        # Default 纯色线
        render_item_default_line = container.find(self.item_name_default_line)
        if render_item_default_line is None:
            render_item_default_line = omr.MRenderItem.create(self.item_name_default_line, omr.MRenderItem.DecorationItem, omr.MGeometry.kLines)
            render_item_default_line.setDrawMode(omr.MGeometry.kAll)
            render_item_default_line.setDepthPriority(omr.MRenderItem.sActiveWireDepthPriority)
            shader_default_line = shader_mgr.getStockShader(omr.MShaderManager.k3dThickLineShader).clone()
            render_item_default_line.setShader(shader_default_line)
            container.add(render_item_default_line)

        # Default 纯色点
        render_item_default_point = container.find(self.item_name_default_point)
        if render_item_default_point is None:
            render_item_default_point = omr.MRenderItem.create(self.item_name_default_point, omr.MRenderItem.DecorationItem, omr.MGeometry.kPoints)
            render_item_default_point.setDrawMode(omr.MGeometry.kAll)
            render_item_default_point.setDepthPriority(omr.MRenderItem.sActiveWireDepthPriority)
            shader_default_point = shader_mgr.getStockShader(omr.MShaderManager.k3dFatPointShader).clone()
            render_item_default_point.setShader(shader_default_point)
            container.add(render_item_default_point)

    def _init_gpu_buffers(self, render_data: RenderData):
        """GPU 专属显存结构建立"""
        if self.vertex_buffer is None:
            # vertices_position_gpu_buffer
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

            # 5. 纯坐标专属阵列打包 (为 default 单色项目准备,不挂载颜色缓冲)
            self.vertex_buffer_array_default = omr.MVertexBufferArray()
            self.vertex_buffer_array_default.append(self.vertex_buffer, "")

            # 此时的 GPU 缓冲是空的 必须强制将节点的阀门全部打开,保证基础数据能拷入新缓冲
            render_data.dirty_vertices_pos = True
            render_data.dirty_face_colors = True
            render_data.dirty_line_colors = True
            render_data.dirty_point_colors = True
            render_data.dirty_face_indices = True
            render_data.dirty_line_indices = True
            render_data.dirty_point_indices = True

    def _sync_topology_buffers(self, render_data: RenderData):
        """负责同步客观物理数据 (位置与索引)的显存"""
        # --- 点位置显存同步 ---
        if render_data.dirty_vertices_pos:
            vertex_count = len(render_data.vertices_pos) // 3
            if vertex_count > 0:
                vertex_addr = self.vertex_buffer.acquire(vertex_count, True)
                if vertex_addr:
                    ctypes.memmove(vertex_addr, ctypes.addressof(render_data.vertices_pos), ctypes.sizeof(render_data.vertices_pos))
                    self.vertex_buffer.commit(vertex_addr)

        # --- 拓补结构显存同步 ---
        if render_data.dirty_face_indices:
            index_count = len(render_data.face_indices)
            if index_count > 0:
                index_addr_face = self.index_buffer_face.acquire(index_count, True)
                if index_addr_face:
                    ctypes.memmove(index_addr_face, ctypes.addressof(render_data.face_indices), ctypes.sizeof(render_data.face_indices))
                    self.index_buffer_face.commit(index_addr_face)
            render_data.dirty_face_indices = False

        if render_data.dirty_line_indices:
            index_count = len(render_data.line_indices)
            if index_count > 0:
                index_addr_line = self.index_buffer_line.acquire(index_count, True)
                if index_addr_line:
                    ctypes.memmove(index_addr_line, ctypes.addressof(render_data.line_indices), ctypes.sizeof(render_data.line_indices))
                    self.index_buffer_line.commit(index_addr_line)
            render_data.dirty_line_indices = False

        if render_data.dirty_point_indices:
            index_count = len(render_data.point_indices)
            if index_count > 0:
                index_addr_point = self.index_buffer_point.acquire(index_count, True)
                if index_addr_point:
                    ctypes.memmove(index_addr_point, ctypes.addressof(render_data.point_indices), ctypes.sizeof(render_data.point_indices))
                    self.index_buffer_point.commit(index_addr_point)
            render_data.dirty_point_indices = False

    def _calculate_colors_direct(self, render_data: RenderData, face_view, line_view, point_view):
        """🌟 专门负责颜色计算与显存直写,由 update 阶段的指挥官直接调用"""
        # 线和点基础底色填充
        if line_view is not None:
            cColor.render_fill(line_view, render_data.wire_color)
        if point_view is not None:
            cColor.render_fill(point_view, render_data.vertex_color)

        # 面色热力图/遮罩计算
        if face_view is not None:
            if render_data.paint_weights_view is not None:
                if render_data.is_mask:
                    # 遮罩专属配色
                    cColor.render_gradient(
                        render_data.paint_weights_view,
                        face_view,
                        render_data.mask_remap_a_color,
                        render_data.mask_remap_b_color,
                    )
                elif render_data.render_mode == 0:
                    # 黑白色
                    cColor.render_gradient(
                        render_data.paint_weights_view,
                        face_view,
                        render_data.weights_remap_a_color,
                        render_data.weights_remap_b_color,
                    )
                else:
                    # 热色
                    cColor.render_heatmap(
                        render_data.paint_weights_view,
                        face_view,
                    )
            else:
                # 无数据输入,填充蓝底
                cColor.render_fill(face_view, (0.0, 0.0, 1.0, 1.0))

        # 笔刷高亮叠加直写
        if point_view is not None and render_data.brush_hit_count > 0:
            cColor.render_brush_gradient(
                point_view,
                render_data.brush_hit_indices,
                render_data.brush_hit_weights,
                render_data.brush_hit_count,
                render_data.brush_remap_a_color,
                render_data.brush_remap_b_color,
            )

    def _sync_color_buffers(self, render_data: RenderData):
        """利用 Context Manager 为GPU Buffer开锁,Cython 直接计算到GPU buffer"""
        vtx_count = render_data.vtx_count
        if vtx_count <= 0:
            return

        if render_data.dirty_face_colors:
            with gpu_write_session(self.color_buffer_face, vtx_count) as face_view:
                # 只传 face_view,线和点传 None,计算引擎会自动跳过它们
                self._calculate_colors_direct(render_data, face_view=face_view, line_view=None, point_view=None)
            render_data.dirty_face_colors = False

        if render_data.dirty_line_colors:
            with gpu_write_session(self.color_buffer_line, vtx_count) as line_view:
                self._calculate_colors_direct(render_data, face_view=None, line_view=line_view, point_view=None)
            render_data.dirty_line_colors = False

        if render_data.dirty_point_colors:
            with gpu_write_session(self.color_buffer_point, vtx_count) as point_view:
                self._calculate_colors_direct(render_data, face_view=None, line_view=None, point_view=point_view)
            render_data.dirty_point_colors = False

    def _update_render_items_state(self, render_data: RenderData, container):
        """动态修改专属着色器参数及渲染开关"""
        # 注意:现已改写为 GPU 缓冲,所以只要 vtx_count 足够,就不需要校验 len() 了
        vertex_count = render_data.vtx_count
        has_vertices = vertex_count > 0

        # 获取组件缓存
        render_item_face = container.find(self.item_name_face)
        render_item_line = container.find(self.item_name_line)
        render_item_point = container.find(self.item_name_point)
        render_item_default_face = container.find(self.item_name_default_face)
        render_item_default_line = container.find(self.item_name_default_line)
        render_item_default_point = container.find(self.item_name_default_point)

        # 🛡️ 安全锁:颜色数组里的点数,必须大于等于顶点数组里的点数
        # (现已改为直写显存,GPU缓冲容量必然与点数一致,因此直接通过)
        is_color_safe = True

        if render_item_face:
            has_faces = has_vertices and (len(render_data.face_indices) > 0)
            render_item_face.enable(render_data.draw_faces and has_faces and is_color_safe)

        if render_item_line:
            render_item_line.getShader().setParameter("lineWidth", (render_data.line_width, render_data.line_width))
            has_lines = has_vertices and (len(render_data.line_indices) > 0)
            render_item_line.enable(render_data.draw_lines and has_lines and is_color_safe)

        if render_item_point:
            render_item_point.getShader().setParameter("pointSize", (render_data.point_size, render_data.point_size))
            has_points = has_vertices and (len(render_data.point_indices) > 0)
            render_item_point.enable(render_data.draw_points and has_points and is_color_safe)

        # Default 纯色渲染项控制
        if render_item_default_face:
            render_item_default_face.getShader().setParameter("solidColor", render_data.default_face_color)
            has_default_faces = has_vertices and (len(render_data.face_indices) > 0)
            render_item_default_face.enable(render_data.draw_default_faces and has_default_faces)

        if render_item_default_line:
            render_item_default_line.getShader().setParameter("solidColor", render_data.default_line_color)
            render_item_default_line.getShader().setParameter("lineWidth", (render_data.line_width, render_data.line_width))
            has_default_lines = has_vertices and (len(render_data.line_indices) > 0)
            render_item_default_line.enable(render_data.draw_default_lines and has_default_lines)

        if render_item_default_point:
            render_item_default_point.getShader().setParameter("solidColor", render_data.default_point_color)
            render_item_default_point.getShader().setParameter("pointSize", (render_data.point_size, render_data.point_size))
            has_default_points = has_vertices and (len(render_data.point_indices) > 0)
            render_item_default_point.enable(render_data.draw_default_points and has_default_points)

    def update(self, container, frameContext):
        with MayaNativeProfiler("render-update", 2):
            with MayaNativeProfiler("render-update-dg", 4):
                # 1. 拉取 dummy 属性触发 DG 依赖脏标更新
                om.MPlug(self.node_obj, TriangleShape.outDummyAttr).asInt()
            with MayaNativeProfiler("render-update-userNode-data", 4):
                shape_inst: TriangleShape = om.MFnDependencyNode(self.node_obj).userNode()
                render_data: RenderData = shape_inst.render_data

            with MayaNativeProfiler("render-update-buffer", 4):
                # 2. 初始 RenderItem 与 GPU缓冲 创建
                self._init_render_items(container, omr.MRenderer.getShaderManager())
                self._init_gpu_buffers(render_data)

            # 3. 拦截非法状态
            if render_data.vertices_pos is None or render_data.vtx_count <= 0:
                return

            with MayaNativeProfiler("render-update-set-data", 4):
                # 4. 核心任务分发 (领域解耦)
                with MayaNativeProfiler("render-update-set-data-topology", 5):
                    self._sync_topology_buffers(render_data)
                with MayaNativeProfiler("render-update-set-data-color", 6):
                    self._sync_color_buffers(render_data)
                with MayaNativeProfiler("render-update-set-data-items-state", 7):
                    self._update_render_items_state(render_data, container)

            # 5. 绑定终极渲染状态通知
            bbox = shape_inst.boundingBox()
            self.setGeometryForRenderItem(container.find(self.item_name_face), self.vertex_buffer_array_face, self.index_buffer_face, bbox)
            self.setGeometryForRenderItem(container.find(self.item_name_line), self.vertex_buffer_array_line, self.index_buffer_line, bbox)
            self.setGeometryForRenderItem(container.find(self.item_name_point), self.vertex_buffer_array_point, self.index_buffer_point, bbox)
            self.setGeometryForRenderItem(container.find(self.item_name_default_face), self.vertex_buffer_array_default, self.index_buffer_face, bbox)
            self.setGeometryForRenderItem(container.find(self.item_name_default_line), self.vertex_buffer_array_default, self.index_buffer_line, bbox)
            self.setGeometryForRenderItem(container.find(self.item_name_default_point), self.vertex_buffer_array_default, self.index_buffer_point, bbox)

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

    # 🌟 修改点 4:改为使用 registerShape,传入 UI 类和表面形状类型
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
