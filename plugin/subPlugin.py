import maya.api.OpenMaya as om2
import maya.api.OpenMayaUI as omui
import maya.api.OpenMayaRender as omr
import maya.api.OpenMayaAnim as oma  # 🌟 新增：用于获取时间
import ctypes
import math  # 🌟 新增：用于计算正弦波


def maya_useNewAPI():
    pass


# 🌟 修改点 1：基类改为 om2.MPxSurfaceShape，并重命名为 TriangleShape
class TriangleShape(om2.MPxSurfaceShape):
    id = om2.MTypeId(0x80089)
    drawDbClassification = "drawdb/subscene/triangleShape"
    registrantId = "TriangleShapePlugin"

    lineWidthAttr = om2.MObject()
    pointSizeAttr = om2.MObject()

    def __init__(self):
        om2.MPxSurfaceShape.__init__(self)

    @classmethod
    def creator(cls):
        return TriangleShape()

    @classmethod
    def initialize(cls):
        # 🌟 2. 创建真正的 Maya 节点属性，暴露在通道盒里！
        nAttr = om2.MFnNumericAttribute()

        # 创建“线宽”属性 (默认 5.0)
        cls.lineWidthAttr = nAttr.create("lineWidth", "lw", om2.MFnNumericData.kFloat, 5.0)
        nAttr.keyable = True  # 允许在通道盒显示并做动画
        nAttr.storable = True  # 允许保存在 Maya 文件里
        nAttr.setMin(1.0)  # 最小 1 个像素
        om2.MPxNode.addAttribute(cls.lineWidthAttr)

        # 创建“点大小”属性 (默认 15.0)
        cls.pointSizeAttr = nAttr.create("pointSize", "ps", om2.MFnNumericData.kFloat, 15.0)
        nAttr.keyable = True
        nAttr.storable = True
        nAttr.setMin(1.0)
        om2.MPxNode.addAttribute(cls.pointSizeAttr)

    def isBounded(self):
        return True

    def boundingBox(self):
        return om2.MBoundingBox(om2.MPoint(-100, -100, -100), om2.MPoint(100, 100, 100))


# 🌟 修改点 2：必须添加配套的 UI 类 (属于 omui)
class TriangleShapeUI(omui.MPxSurfaceShapeUI):
    def __init__(self):
        omui.MPxSurfaceShapeUI.__init__(self)

    @classmethod
    def creator(cls):
        return TriangleShapeUI()


class TriangleOverride(omr.MPxSubSceneOverride):
    def __init__(self, obj):
        super(TriangleOverride, self).__init__(obj)
        self.node_obj = obj

        self.item_name = "my_triangle_item"
        self.item_name_line = "my_triangle_line"  # 🌟 新增：线的名字
        self.item_name_point = "my_triangle_point"  # 🌟 新增：点的名字

        self.vtx_buffer = None
        self.norm_buffer = None
        self.clr_buffer = None
        self.idx_buffer = None
        self.vtx_buffer_array = None

        # 🌟 新增：为线和点准备独立的显存池
        self.clr_buffer_line = None
        self.clr_buffer_point = None
        self.idx_buffer_line = None
        self.idx_buffer_point = None
        self.vtx_buffer_array_line = None
        self.vtx_buffer_array_point = None

    @classmethod
    def creator(cls, obj):
        return TriangleOverride(obj)

    def supportedDrawAPIs(self):
        return omr.MRenderer.kAllDevices

    def requiresUpdate(self, container, frameContext):
        return True

    def update(self, container, frameContext):
        render_item = container.find(self.item_name)
        # 🌟 新增：查找线和点
        render_item_line = container.find(self.item_name_line)
        render_item_point = container.find(self.item_name_point)

        shader_mgr = omr.MRenderer.getShaderManager()

        # ==========================================
        # 1. 核心面 (保持你完美的代码不变！)
        # ==========================================
        if render_item is None:
            render_item = omr.MRenderItem.create(self.item_name, omr.MRenderItem.MaterialSceneItem, omr.MGeometry.kTriangles)
            render_item.setSelectionMask(om2.MSelectionMask("polymesh"))
            render_item.setDrawMode(omr.MGeometry.kAll)
            shader = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVSolidShader)
            render_item.setShader(shader)
            container.add(render_item)

        # ==========================================
        # 🌟 2. 装饰线 (免疫点击闪退！)
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
        # 🌟 3. 装饰点 (免疫点击闪退！)
        # ==========================================
        if render_item_point is None:
            # 关键：使用 DecorationItem！
            render_item_point = omr.MRenderItem.create(self.item_name_point, omr.MRenderItem.DecorationItem, omr.MGeometry.kPoints)
            render_item_point.setDrawMode(omr.MGeometry.kAll)
            render_item_point.setDepthPriority(omr.MRenderItem.sActiveWireDepthPriority)
            # 🌟 5. 同样必须 clone() 胖点着色器
            shader_pt = shader_mgr.getStockShader(omr.MShaderManager.k3dCPVFatPointShader).clone()
            render_item_point.setShader(shader_pt)
            container.add(render_item_point)

        # ==========================================
        # 显存开辟
        # ==========================================
        if self.vtx_buffer is None:
            self.vtx_buffer = omr.MVertexBuffer(omr.MVertexBufferDescriptor("", omr.MGeometry.kPosition, omr.MGeometry.kFloat, 3))
            self.norm_buffer = omr.MVertexBuffer(omr.MVertexBufferDescriptor("", omr.MGeometry.kNormal, omr.MGeometry.kFloat, 3))
            self.clr_buffer = omr.MVertexBuffer(omr.MVertexBufferDescriptor("", omr.MGeometry.kColor, omr.MGeometry.kFloat, 4))
            self.idx_buffer = omr.MIndexBuffer(omr.MGeometry.kUnsignedInt32)

            self.vtx_buffer_array = omr.MVertexBufferArray()
            self.vtx_buffer_array.append(self.vtx_buffer, "")
            self.vtx_buffer_array.append(self.norm_buffer, "")
            self.vtx_buffer_array.append(self.clr_buffer, "")

            # 🌟 新增：为线和点开辟颜色与索引显存
            self.clr_buffer_line = omr.MVertexBuffer(omr.MVertexBufferDescriptor("", omr.MGeometry.kColor, omr.MGeometry.kFloat, 4))
            self.clr_buffer_point = omr.MVertexBuffer(omr.MVertexBufferDescriptor("", omr.MGeometry.kColor, omr.MGeometry.kFloat, 4))
            self.idx_buffer_line = omr.MIndexBuffer(omr.MGeometry.kUnsignedInt32)
            self.idx_buffer_point = omr.MIndexBuffer(omr.MGeometry.kUnsignedInt32)

            # 🌟 新增：打包线和点的阵列 (它们共享你的 vtx_buffer 坐标！)
            self.vtx_buffer_array_line = omr.MVertexBufferArray()
            self.vtx_buffer_array_line.append(self.vtx_buffer, "")
            self.vtx_buffer_array_line.append(self.norm_buffer, "")
            self.vtx_buffer_array_line.append(self.clr_buffer_line, "")

            self.vtx_buffer_array_point = omr.MVertexBufferArray()
            self.vtx_buffer_array_point.append(self.vtx_buffer, "")
            self.vtx_buffer_array_point.append(self.norm_buffer, "")
            self.vtx_buffer_array_point.append(self.clr_buffer_point, "")

        # ==========================================================
        # 动态写入顶点坐标 (完全没变，共享变形！)
        # ==========================================================
        current_time = oma.MAnimControl.currentTime().value
        dynamic_y = 10.0 + math.sin(current_time * 5.0) * 5.0

        vtx_addr = self.vtx_buffer.acquire(3, True)
        raw_vtx = (ctypes.c_float * 9)(0.0, 0.0, 0.0,   
                                       10.0, 0.0, 0.0, 
                                       5.0, dynamic_y, 0.0)  # fmt:off
        ctypes.memmove(vtx_addr, ctypes.addressof(raw_vtx), 9 * 4)
        self.vtx_buffer.commit(vtx_addr)

        # 强写法线
        norm_addr = self.norm_buffer.acquire(3, True)
        raw_norm = (ctypes.c_float * 9)(0.0, 0.0, 1.0, 
                                        0.0, 0.0, 1.0, 
                                        0.0, 0.0, 1.0)  # fmt:skip
        ctypes.memmove(norm_addr, ctypes.addressof(raw_norm), 9 * 4)
        self.norm_buffer.commit(norm_addr)

        # 强写面的颜色
        clr_addr = self.clr_buffer.acquire(3, True)
        raw_clr = (ctypes.c_float * 12)(1.0, 0.0, 0.0, 0.10, 
                                        1.0, 0.0, 1.0, 0.10, 
                                        0.0, 1.0, 0.0, 0.10)  # fmt:skip
        ctypes.memmove(clr_addr, ctypes.addressof(raw_clr), 12 * 4)
        self.clr_buffer.commit(clr_addr)

        # 🌟 强写线的颜色 (例如：全青色)
        clr_addr_l = self.clr_buffer_line.acquire(3, True)
        raw_clr_l = (ctypes.c_float * 12)(1.0, 1.0, 1.0, 1.0, 
                                          0.0, 1.0, 1.0, 1.0, 
                                          0.0, 0.0, 1.0, 1.0)  # fmt:skip
        ctypes.memmove(clr_addr_l, ctypes.addressof(raw_clr_l), 12 * 4)
        self.clr_buffer_line.commit(clr_addr_l)

        # 🌟 强写点的颜色 (例如：全黄色)
        clr_addr_p = self.clr_buffer_point.acquire(3, True)
        raw_clr_p = (ctypes.c_float * 12)(0.0, 1.0, 0.0, 1.0, 
                                          1.0, 0.0, 0.0, 1.0, 
                                          1.0, 1.0, 0.0, 1.0)  # fmt:skip
        ctypes.memmove(clr_addr_p, ctypes.addressof(raw_clr_p), 12 * 4)
        self.clr_buffer_point.commit(clr_addr_p)

        # 强写面的图纸 (3 个索引)
        idx_addr = self.idx_buffer.acquire(3, True)
        raw_idx = (ctypes.c_uint32 * 3)(0, 1, 2)
        ctypes.memmove(idx_addr, ctypes.addressof(raw_idx), 3 * 4)
        self.idx_buffer.commit(idx_addr)

        # 🌟 强写线的图纸 (6 个索引 = 3 条边)
        idx_addr_l = self.idx_buffer_line.acquire(6, True)
        raw_idx_l = (ctypes.c_uint32 * 6)(0, 1, 
                                          1, 2, 
                                          2, 0)  # fmt:skip
        ctypes.memmove(idx_addr_l, ctypes.addressof(raw_idx_l), 6 * 4)
        self.idx_buffer_line.commit(idx_addr_l)

        # 🌟 强写点的图纸 (3 个索引)
        idx_addr_p = self.idx_buffer_point.acquire(3, True)
        raw_idx_p = (ctypes.c_uint32 * 3)(0, 1, 2)
        ctypes.memmove(idx_addr_p, ctypes.addressof(raw_idx_p), 3 * 4)
        self.idx_buffer_point.commit(idx_addr_p)

        # ==========================================================
        # 🌟 阶段 B：每帧实时更新着色器参数！
        # ==========================================================
        # 获取最新的 render item 引用
        ri_line_now = container.find(self.item_name_line)
        ri_point_now = container.find(self.item_name_point)

        # 读取通道盒(Channel Box)里当前的数值
        # 🌟 修改点 3：这里读取属性时的类名改为 TriangleShape
        plug_lw = om2.MPlug(self.node_obj, TriangleShape.lineWidthAttr)
        plug_ps = om2.MPlug(self.node_obj, TriangleShape.pointSizeAttr)

        current_lw = plug_lw.asFloat()
        current_ps = plug_ps.asFloat()

        # 动态修改专属着色器参数！
        if ri_line_now:
            ri_line_now.getShader().setParameter("lineWidth", (current_lw, current_lw))

        if ri_point_now:
            ri_point_now.getShader().setParameter("pointSize", (current_ps, current_ps))

        # 终极绑定通知
        bbox = om2.MBoundingBox(om2.MPoint(-100, -100, -100), om2.MPoint(100, 100, 100))
        self.setGeometryForRenderItem(render_item, self.vtx_buffer_array, self.idx_buffer, bbox)
        self.setGeometryForRenderItem(render_item_line, self.vtx_buffer_array_line, self.idx_buffer_line, bbox)
        self.setGeometryForRenderItem(render_item_point, self.vtx_buffer_array_point, self.idx_buffer_point, bbox)

    def areControlsAllocated(self):
        return False

    def freeControls(self):
        pass


def initializePlugin(obj):
    plugin = om2.MFnPlugin(obj, "VP2_ZeroCopy_Core", "1.0", "Any")

    # 🌟 修改点 4：改为使用 registerShape，传入 UI 类和表面形状类型
    plugin.registerShape("triangleShape", TriangleShape.id, TriangleShape.creator, TriangleShape.initialize, TriangleShapeUI.creator, TriangleShape.drawDbClassification)

    omr.MDrawRegistry.registerSubSceneOverrideCreator(TriangleShape.drawDbClassification, TriangleShape.registrantId, TriangleOverride.creator)


def uninitializePlugin(obj):
    plugin = om2.MFnPlugin(obj)
    omr.MDrawRegistry.deregisterSubSceneOverrideCreator(TriangleShape.drawDbClassification, TriangleShape.registrantId)
    # 🌟 修改点 5：卸载时注销对应的 ID
    plugin.deregisterNode(TriangleShape.id)
