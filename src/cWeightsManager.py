from __future__ import annotations


import typing
import array


from maya import cmds
import maya.OpenMaya as om1  # type:ignore


from . import cBrushCore2Cython as cBrushCoreCython
from ._cRegistry import SkinRegistry
from .cBufferManager import BufferManager
from .cWeightsHandle import WeightsHandle


if typing.TYPE_CHECKING:
    from .cSkinDeform import CythonSkinDeformer  # type: ignore


class WeightsLayerItem:
    def __init__(self, cSkin: CythonSkinDeformer, mDataHandle: om1.MDataHandle, logical_idx: int = -1):
        self.cSkin = cSkin
        self.logical_idx = logical_idx

        _handle_weights = mDataHandle.child(cSkin.aLayerWeights)
        _handle_enabled = mDataHandle.child(cSkin.aLayerEnabled)
        _handle_name = mDataHandle.child(cSkin.aLayerName)

        self.enabled = _handle_enabled.asBool()
        self.weights = WeightsHandle(self.cSkin, self.mPlug_weights, _handle_weights)
        self.name = _handle_name.asString()

    # ==========================================
    # 安全的属性获取:直接找 cSkin 要 MObject,完全不碰短命的 MDataHandle
    # ==========================================
    @property
    def mPlug_weights(self):
        mPlug = om1.MPlug(self.cSkin.mObject, self.cSkin.aLayerWeights)
        mPlug.selectAncestorLogicalIndex(self.logical_idx, self.cSkin.aLayerCompound)
        return mPlug

    @property
    def mPlug_enabled(self):
        mPlug = om1.MPlug(self.cSkin.mObject, self.cSkin.aLayerEnabled)
        mPlug.selectAncestorLogicalIndex(self.logical_idx, self.cSkin.aLayerCompound)
        return mPlug

    @property
    def mPlug_name(self):
        mPlug = om1.MPlug(self.cSkin.mObject, self.cSkin.aLayerName)
        mPlug.selectAncestorLogicalIndex(self.logical_idx, self.cSkin.aLayerCompound)
        return mPlug

    def set_name(self, new_name: str):
        self.mPlug_name.setString(new_name)
        self.name = new_name


class WeightsManager:
    def __init__(self, cSkin: CythonSkinDeformer):
        super().__init__()

        self.cSkin = cSkin
        self.mObject_node = cSkin.mObject
        self.mFnDepend_node = cSkin.mFnDep

        self.layers: dict[int, WeightsLayerItem] = {}

        self.plug_refresh: om1.MPlug = cSkin.plug_refresh
        self.plug_weights: om1.MPlug = om1.MPlug(cSkin.mObject, cSkin.aWeights)
        self.plug_mask: om1.MPlug = om1.MPlug(cSkin.mObject, cSkin.aMaskWeights)

        self.mask_handle: WeightsHandle = None
        self.weights_handle: WeightsHandle = None

    @property
    def layer_indices(self) -> list[int]:
        return list(self.layers.keys())

    @classmethod
    def from_node(cls, node_name: str):
        cSkin: CythonSkinDeformer = SkinRegistry.from_instance_by_string(node_name)
        return cSkin.weights_manager

    def sync_layer_cache(self, mDataBlock: om1.MDataBlock):
        """
        [状态同步器]
        一次性扫描 Maya 节点,刷新所有 Plug 缓存与底层内存池。
        当你在 UI 层面添加、删除了图层,或改变了节点连接后,手动调用此函数。
        """
        # weights
        weights_dataHandle = mDataBlock.outputValue(self.cSkin.aWeights)
        self.weights_handle = WeightsHandle(self.cSkin, self.plug_weights, weights_dataHandle)
        # mask
        mask_dataHandle = mDataBlock.outputValue(self.cSkin.aMaskWeights)
        self.mask_handle = WeightsHandle(self.cSkin, self.plug_mask, mask_dataHandle)

        # layer
        self.layers.clear()
        mArrayDataHandle: om1.MArrayDataHandle = mDataBlock.outputArrayValue(self.cSkin.aLayerCompound)
        for idx in range(mArrayDataHandle.elementCount()):
            mArrayDataHandle.jumpToArrayElement(idx)
            logical_idx = mArrayDataHandle.elementIndex()
            element_handle: om1.MDataHandle = mArrayDataHandle.outputValue()

            self.layers[logical_idx] = WeightsLayerItem(self.cSkin, element_handle, logical_idx)

    def updateDG(self):
        self.cSkin.forceRefresh()

    def get_layer(self, index: int, logicalIndex: bool = True) -> WeightsLayerItem:
        """
        获取图层实例,支持通过逻辑索引或物理索引进行查询。
        """
        if logicalIndex:
            return self.layers.get(index, None)
        try:
            return list(self.layers.values())[index]
        except IndexError:
            return None

    def get_weights_handle(self, layer_logical_idx: int, is_mask: bool = False) -> WeightsHandle:
        """
        终极路由：通过逻辑 ID 和 Mask 标记，精确返回物理句柄。
        """
        if is_mask:
            return self.mask_handle

        if layer_logical_idx == -1:
            return self.weights_handle

        if layer_logical_idx in self.layers:
            return self.layers[layer_logical_idx].weights

        return None

    def paint_stroke_coroutine(self, layer_idx: int, is_mask: bool, backup: bool = True) -> typing.Generator[bool, cBrushCoreCython.BrushStrokeContext, None]:
        """
        [笔刷涂抹协程]
        这是一个专为笔刷涂抹设计的协程函数，提供了一个高性能的上下文环境，让用户可以在其中安全地调用 Cython 引擎进行实时权重修改。
        """
        handle = self.get_weights_handle(layer_idx, is_mask)
        if not handle:
            yield False
            return

        with handle.processor_session(backup=backup) as processor:
            if not processor:
                yield False
                return

            base_processor = None
            if layer_idx >= 0:
                base_handle = self.get_weights_handle(-1, is_mask=False)
                if base_handle:
                    base_v_cnt, base_s_cnt, _, base_raw = base_handle.parse_raw_weights()
                    base_w2d = memoryview(base_raw).cast("B").cast("f", (base_v_cnt, base_s_cnt))
                    base_processor = self.cSkin.get_processor(base_w2d)

            try:
                while True:
                    ctx: cBrushCoreCython.BrushStrokeContext = yield True
                    if ctx:
                        hit_count, hit_indices, _ = processor.process_stroke(ctx, normalize=True)
                        if hit_count > 0:
                            dirty_vtx_view = hit_indices[:hit_count]
                            if base_processor:
                                self.update_composite(dirty_vtx_view, out_processor=base_processor)

            except GeneratorExit:
                if base_processor:
                    self.updateDG()

    def add_layer(self, name: str = "NewLayer", weights_value=0.0, mask_weights_value=1.0) -> int:
        """
        添加一个新图层。
        所有底层物理内存分配 (Layer & Mask) 全部推入延迟队列，确保 DG 绝对安全！
        """
        node_name = self.mFnDepend_node.name()
        compound_plug = om1.MPlug(self.mObject_node, self.cSkin.aLayerCompound)

        # 1. 计算可用的 Logical Index
        max_idx = -1
        for i in range(compound_plug.numElements()):
            elem = compound_plug.elementByPhysicalIndex(i)
            if elem.logicalIndex() > max_idx:
                max_idx = elem.logicalIndex()
        new_idx = max_idx + 1

        plug_base = f"{node_name}.layers[{new_idx}]"

        # 2. 触发 Maya 属性创建
        cmds.setAttr(f"{plug_base}.layerName", name, type="string")
        cmds.setAttr(f"{plug_base}.layerEnabled", True)

        self.cSkin.forceRefresh()  # 强制 Maya 刷新 DG,确保新属性生效并可访问
        # 3. 提取 Base 层数据
        base_handle = self.get_weights_handle(-1, False)
        if not base_handle or not base_handle.is_valid:
            return -1

        vtx_count, channel_count, channel_indices, _ = base_handle.parse_raw_weights()

        if vtx_count <= 0 or channel_count <= 0:
            return -1

        safe_bones = list(channel_indices)

        handle = self.get_weights_handle(new_idx, False)
        if handle:
            handle.allocate_and_set_weights(
                vtx_count=vtx_count,
                influence_indices=safe_bones,
                backup=False,  # 新建图层的物理内存会被 Maya 撤销属性时直接销毁，无需额外录制
            )

        # 扩容并初始化 Mask
        _, mask_channel_count, mask_channel_indices, old_mask_weights_1d = self.mask_handle.parse_raw_weights()

        new_mask_indices = [*list(mask_channel_indices), new_idx] if mask_channel_indices is not None else [new_idx]
        new_mask_channel_count = len(new_mask_indices)

        new_mask_weights = array.array("f", [mask_weights_value]) * (vtx_count * new_mask_channel_count)

        if old_mask_weights_1d:
            old_view = memoryview(old_mask_weights_1d).cast("B").cast("f")
            new_view = memoryview(new_mask_weights).cast("B").cast("f")
            for j, _ in enumerate(mask_channel_indices):
                new_view[j::new_mask_channel_count] = old_view[j::mask_channel_count]

        # 提交 Mask 覆写 (Mask 是全图层共享的，它的扩容必须录制 Undo)
        self.mask_handle.allocate_and_set_weights(
            vtx_count=vtx_count,
            influence_indices=new_mask_indices,
            weights_1d=new_mask_weights,
            backup=True,
        )

        self.cSkin.forceRefresh()
        return new_idx

    def delete_layer(self, layer_idx: int) -> bool:
        """
        删除指定图层并释放关联资源。
        图层自身的 weights 随原生属性销毁；共享的 mask 需要手动执行切片缩容。
        """
        if layer_idx == -1:
            return False

        node_name = self.mFnDepend_node.name()
        plug_base = f"{node_name}.layers[{layer_idx}]"

        if not cmds.objExists(plug_base):
            return False

        vtx_count, mask_channel_count, mask_channel_indices, old_mask_weights_1d = self.mask_handle.parse_raw_weights()

        # 确保被删图层确实在我们的 Mask 骨骼列表里
        if mask_channel_indices is not None and layer_idx in mask_channel_indices:
            col_idx = list(mask_channel_indices).index(layer_idx)  # 找到要剔除的列索引

            new_mask_ids = list(mask_channel_indices)
            new_mask_ids.pop(col_idx)  # 拔出那根废弃的“骨骼”
            new_mask_channel_count = len(new_mask_ids)

            if new_mask_channel_count > 0 and old_mask_weights_1d:
                # 申请一块小一点的新内存 (降维)
                new_mask_weights = array.array("f", [0.0]) * (vtx_count * new_mask_channel_count)

                old_view = memoryview(old_mask_weights_1d).cast("B").cast("f")
                new_view = memoryview(new_mask_weights).cast("B").cast("f")

                # 魔法步长拷贝：跳过被删掉的 col_idx 那一列，把其余的列严丝合缝地拼接起来
                new_col = 0
                for old_col in range(mask_channel_count):
                    if old_col == col_idx:
                        continue
                    new_view[new_col::new_mask_channel_count] = old_view[old_col::mask_channel_count]
                    new_col += 1
            else:
                new_mask_weights = None

            self.mask_handle.allocate_and_set_weights(
                vtx_count=vtx_count,
                influence_indices=new_mask_ids,
                weights_1d=new_mask_weights,
                backup=True,
            )

        cmds.removeMultiInstance(plug_base, b=True)
        if layer_idx in self.layers:
            del self.layers[layer_idx]
        self.cSkin.forceRefresh()
        return True

    def update_composite(self, vtx_indices=None, out_processor=None):
        """图层混合器"""
        is_sparse = vtx_indices is not None and len(vtx_indices) > 0
        v_view = BufferManager.auto(vtx_indices, "i").view if is_sparse else None

        active_layers = [idx for idx in sorted(self.layer_indices) if self.get_layer(idx).enabled]
        if not active_layers:
            return

        def _do_blend(processor_inst: cBrushCoreCython.SkinWeightProcessor):
            processor_inst.clear_buffer_sparse(vertex_indices=v_view)

            # 安全提取全局 Mask，供内部切片
            mask_v, mask_ch, mask_w = self.mask_handle.get_weights()
            mask_full_view = memoryview(mask_w).cast("B").cast("f") if mask_w else None

            for layer_idx in active_layers:
                handle_w = self.get_weights_handle(layer_idx)
                if not handle_w:
                    continue

                _, _, w_in_1d = handle_w.get_weights()
                if not w_in_1d:
                    continue
                w_in_view = memoryview(w_in_1d).cast("B").cast("f")

                # 🚀 从全局 Mask 视图中，切出当前 layer_idx 的那一列！
                m_in_view = None
                if mask_full_view and mask_ch is not None:
                    try:
                        col_idx = list(mask_ch).index(layer_idx)
                        m_in_view = mask_full_view[col_idx :: len(mask_ch)]
                    except ValueError:
                        pass

                # 🚀 绝杀：将底层指针直接抛给 C 引擎执行循环，全过程纯 C 级运算！
                processor_inst.add_layer_weights(layer_weights=w_in_view, layer_mask=m_in_view, vertex_indices=v_view)

            processor_inst.normalize_weights(vertex_indices=v_view, priority_influence=-1)

        if out_processor:
            _do_blend(out_processor)
        else:
            base_handle = self.get_weights_handle(-1, is_mask=False)
            with base_handle.processor_session(backup=False) as temp_processor:
                if temp_processor:
                    _do_blend(temp_processor)
