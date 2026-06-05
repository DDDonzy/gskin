# gskin · API 速查手册

> 给二次开发者用：插件清单、节点、对外操作 API、Cython 内核函数签名。

---

## 1. 插件清单

| 插件文件 | 插件名 | 注册产出 | API 版本 |
| --- | --- | --- | --- |
| `plugin/cSkinPlugin.py` | `cSkinPlugin` | Node `cSkinDeformer`（`MTypeId 0x00080033`，`kDeformerNode`） | OM1 |
| `plugin/cBrushPlugin.py` | `cBrushPlugin` | ContextCommand `cBrushCtx` | OM2 |
| `plugin/subPlugin.py` | （vendor `DDDonzy`） | Shape `triangleShape`（`MTypeId 0x80089`） + `MPxSubSceneOverride` | OM2 |
| `plugin/testPlugin.py` | `testPlugin` | Node `MemTestNode`（`MTypeId 0x87001`） | OM1 |

加载方式（任选）：
```python
import maya.cmds as cmds
cmds.loadPlugin(r"E:\d_maya\gskin\plugin\cSkinPlugin.py")
cmds.loadPlugin(r"E:\d_maya\gskin\plugin\cBrushPlugin.py")
cmds.loadPlugin(r"E:\d_maya\gskin\plugin\subPlugin.py")

# 或者用一键脚本
from gskin._debug.gskinReload import reload_all_plugins
reload_all_plugins()
```

---

## 2. `cSkinDeformer` 节点属性

来自 `src/cSkinDeform2.py::CSkinDeform.nodeInitializer`。

### 2.1 物理输入

| 长名 | 短名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `geomMatrix` | `gm` | `kMatrix` | 模型 transform 矩阵（含 inverse 加速） |
| `bindPreMatrix` | `bpm` | `kMatrix[]` | 骨骼绑定姿态逆矩阵数组 |
| `matrix` | `bm` | `kMatrix[]` | 骨骼当前世界矩阵数组 |
| `weightsData` | `wd` | `kVectorArray` | 全量权重（通过 `MWeightsHandle` 访问） |
| `envelope` | — | `float` | 蒙皮整体强度（继承自 `MPxGeometryFilter`） |

### 2.2 Layer 树

| 长名 | 短名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `layers` | `lyd` | compound | 层父节点 |
| `layers.layersMaskData` | `lmd` | `kVectorArray` | 各 Layer 的 mask 权重 |
| `layers.layersLockMask` | `llm` | `kIntArray` | 各 Layer 是否锁定 mask |
| `layers.layerData` | `lds` | compound[] | Layer 数据数组 |
| `layers.layerData[i].layerName` | `ln` | `kString` | Layer 名 |
| `layers.layerData[i].layerEnabled` | `le` | `bool` | Layer 启用状态 |
| `layers.layerData[i].layerWeightsData` | `lwd` | `kVectorArray` | Layer 内部权重 |
| `layers.layerData[i].layerLockInfluences` | `lli` | `kIntArray` | Layer 内部锁定的骨骼索引列表 |

### 2.3 Paint 上下文

| 长名 | 短名 | 类型 | 说明 |
| --- | --- | --- | --- |
| `currentPaintLayer` | `cpl` | `int` | 当前绘制的 Layer 索引（-1 = 无） |
| `currentPaintInfluence` | `cpi` | `int` | 当前绘制的骨骼索引 |
| `currentPaintMask` | `cpm` | `bool` | 是否绘制 mask 而非权重 |
| `forceDirty` | `di` | `int` | 手动触发求值（外部 `set_dirty` 用） |

> 全部属性都通过 `attributeAffects` 指向 `outputGeometry`。

---

## 3. `triangleShape` 节点属性

来自 `plugin/subPlugin.py::TriangleShape.initialize`。

| 类别 | 属性 | 短名 | 默认 |
| --- | --- | --- | --- |
| Source | `inputMesh` | `ipm` | — |
| Mode | `renderMode`（Alpha/Heatmap） | `rm` | 0 |
| Style | `lineWidth` / `pointSize` | `lw` / `ps` | 1.0 / 5.0 |
| Toggle | `drawFaces` / `drawLines` / `drawPoints` | `df` / `dl` / `dp` | True |
| Default Toggle | `defaultDrawFaces/Lines/Points` | `ddf/ddl/ddp` | False |
| Color × 8 | `renderWireColor` | `wcl` | (0,1,1) |
|  | `renderVertexColor` | `vcl` | (1,0,0) |
|  | `maskRemapAColor / B` | `mra/mrb` | (0,0,0) / (1,1,1) |
|  | `weightsRemapAColor / B` | `wra/wrb` | (0,0,0) / (1,1,1) |
|  | `brushRemapAColor / B` | `bra/brb` | (1,0,0) / (1,1,0) |
| Default Color × 3 | `defaultFace/Line/PointColor` | `dfc/dlc/dpc` | (0.5,0.5,0.5)/(0,1,0)/(0,0,1) |
| 隐藏 | `outDummy` | `od` | （所有上述属性都 attributeAffects 到这里） |

---

## 4. `FnCSkinDeform` 公共操作 API

`src/cSkinDeform2.py::FnCSkinDeform`，外部脚本入口。

### 4.1 构造

```python
FnCSkinDeform.from_string(node_name)            # 用节点名
FnCSkinDeform.from_mObject(mObj)                # 用 OM1 / OM2 MObject
FnCSkinDeform(cSkinDeform_instance)             # 直接传 Python 实例（自动转 weakref.proxy）
```

### 4.2 求值控制

| 方法 | 说明 |
| --- | --- |
| `set_dirty()` | 自增 `forceDirty`，标脏但不求值 |
| `pull_output()` | 强制 `outputGeometry[0].asMObject()`，触发 deform |
| `ensure_node()` | 校验绑定的节点仍存在 |

### 4.3 Layer 操作

```python
fn.add_layer()
fn.delete_layer(index)
fn.set_layer_name(index, name)
fn.set_layer_enabled(index, enabled)
fn.set_layer_weights(index, weights, num_influences)
fn.set_layer_lock_influences(index, lock_list)

fn.get_layer_name(index)
fn.get_layer_enabled(index)
fn.get_layer_lock_influences(index)
```

### 4.4 权重 / Mask

```python
fn.set_weights(weights, num_influences)
fn.set_weights_from_skinCluster(name)
fn.set_mask_weights(weights, num_layers)
fn.set_mask_weights_lock(lock_list)
```

### 4.5 一键创建

```python
cSkin_name, fn = FnCSkinDeform.create_cSkinDeform_from_skinCluster("skinCluster1")
# 自动: 添加 deformer + 接 matrix/bindPreMatrix + 拷贝权重 + 关闭原 skinCluster.envelope
```

---

## 5. `MRegistry` 全局注册表

`src/MRegistry.py` — 解决 OM1 节点实例无法被 OM2 拿到的问题。

```python
MRegistry.register(python_instance)             # 在 postConstructor 里注册
MRegistry.get_instance(node_name | mObject)     # 任意 API 取出 weakref.proxy
MRegistry.get_hash(mObject)                     # 通过 MObjectHandle 拿稳定哈希
```

存储用 `weakref.WeakValueDictionary`，节点删除自动失效。

---

## 6. `DirtyEvent`

`src/MDirtyEvent.py` — 节点 dirty 标记 + 回调融合体。

```python
event = DirtyEvent(triggers=(self.aWeights, self.aLayer),
                   functions=self._update_weights_cache)

# DG 模式
def setDependentsDirty(self, plug, arr):
    event.sync_from_plug(plug)

# Parallel 模式
def preEvaluation(self, ctx, evalNode):
    event.sync_from_evaluation(evalNode)

# compute / deform
event.execute(dataBlock)            # 只在 is_dirty=True 时触发回调
event.set_dirty(True)               # 主动标脏
```

回调用 `weakref.WeakMethod`，节点删除时静默失效。

---

## 7. `MFloatArrayProxy` / `MWeightsHandle`

`src/MFloatArrayProxy.py`、`src/MWeightsHandle.py`。

```python
# 从 MPlug 取
proxy = MFloatArrayProxy.from_mPlug(plug)
proxy = MFloatArrayProxy.from_string("node.attr")

# 节点内部
proxy = MFloatArrayProxy(dataBlock.inputValue(attr))

proxy.set_array(list_or_array)
proxy.resize(float_count)
proxy.tolist()
proxy.tobytes()
proxy.view                          # memoryview, dtype=float32
proxy.__array_interface__           # 喂给 numpy 直接零拷贝
proxy.set_to_mPlug(plug, copy=False)
```

`MWeightsHandle`（继承）多出 2D 形状：
```python
h = MWeightsHandle.from_string("cSkin.weightsData")
h.set_weights(weights_flat, num_influences)
h.get_weights() / get_weights_raw()
h.get_influence_weights(idx) / get_influence_weights_raw(idx)
h.resize(num_vertices, num_influences)
h.remap_influences(source_indices, target_indices)
h.num_vertices, h.num_influences
```

---

## 8. `TopologyContext`

`src/MTopologyContext.py`。

```python
tc = TopologyContext(mFnMesh)               # 直接绑定
tc = TopologyContext.from_string("pCube1")  # 通过 mesh 名

tc.update_position()                        # 更新顶点坐标视图
tc.update_topology(update_csr=True)         # 重建拓扑（带 CSR）
tc.update_fnMesh(mFnMesh)
tc.update_fnMesh_from_string(name)

# 字段（全部 memoryview）
tc.position, tc.tri_face_indices, tc.tri_edge_indices,
tc.quad_edge_indices, tc.v2v_offsets, tc.v2v_indices,
tc.v2f_offsets, tc.v2f_indices
```

> CSR 计算的 Python 实现性能差，正在被 `cTopologyCython.build_v2v_adjacency` / `build_v2f_adjacency` 替代。

---

## 9. `BufferManager`

`src/cBufferManager.py`。

```python
mgr = BufferManager.auto(data, format_char="f", shape=None)   # 智能路由
mgr = BufferManager.allocate("f", (n, 3))                     # 全新内存
mgr = BufferManager.from_ptr(ptr, "f", (n,))                  # 裸指针
mgr = BufferManager.from_buffer(arr, "f")                     # 共享内存
mgr = BufferManager.from_list(py_list, "f")                   # 显式拷贝
mgr = BufferManager.from_ctypes(ctypes_arr, "f")              # 接管所有权

mgr.reshape(new_shape)
mgr.slice(start, end)                       # 0 拷贝切片，独立 ptr
mgr.fill(value)                             # memset / 倍增填充
mgr.copy_to(dest_ptr)
mgr.ptr / mgr.view / mgr.ctypes / mgr.shape / mgr.nbytes
```

---

## 10. Cython 内核函数签名

### 10.1 `cSkinDeformCython`

```python
cal_deform_matrices(
    out_rotate_matrix_view:  float[:,:],   # [B, 9]
    out_translate_vec_view:  float[:,:],   # [B, 3]
    influences_matrix_view:  double[:,:],  # [B, 16]
    bind_pre_matrix_view:    double[:,:],  # [B, 16]
    geo_matrix:              double[:],    # [16]
    geo_matrix_i:            double[:],    # [16]
    geo_matrix_is_identity:  bint,
)

run_skinning_core(
    out_position_view:      float[:],      # [V*3]
    original_position_view: float[:],      # [V*3]
    weights_view:           float[:],      # [V*B]
    rotate_matrix_view:     float[:,:],    # [B, 9]
    translate_vector_view:  float[:,:],    # [B, 3]
    envelope:               float,
)

run_partial_skinning_core(
    out_position_view:      float[:],
    orig_position_view:     float[:],
    vertex_indices_view:    int[:],        # 待计算的顶点 ID 列表
    weights_view:           float[:],
    rotate_matrix_view:     float[:,:],
    translate_vector_view:  float[:,:],
    envelope:               float,
)
```

### 10.2 `cBrushCore2Cython.CoreBrushEngine`

```python
CoreBrushEngine(
    vtx_positions2D:    float[:,::1],   # [V, 3]
    triangle_indices2D: int[:,::1],     # [T, 3]
    v2v_offset:         int[::1],
    v2v_indices:        int[::1],
    v2f_offset:         int[::1],
    v2f_indices:        int[::1],
)

# 核心方法
engine.update_vertex_positions(new_positions2D)
engine.lock_mesh()                       # 拍快照
engine.unlock_mesh()
engine.raycast(ray_pos: tuple, ray_dir: tuple, cull_backface: bool=True)
    -> (hit, pos, normal, tri_idx, t, u, v)
engine.calc_brush_falloff(
    hit_position:       tuple,
    prev_hit_position:  tuple,
    hit_tri_idx:        int,
    radius:             float,
    falloff_mode:       int,    # 0:Linear 1:Airbrush 2:Solid 3:Dome 4:Spike
    use_surface:        bool,
) -> (count, indices_view, falloff_view)
```

### 10.3 `BrushStrokeContext`

```python
BrushStrokeContext(
    brush_mode:      int,      # 0:Add 1:Sub 2:Replace 3:Multiply 4:Smooth 5:Sharp
    values:          float[::1],
    channel_indices: int[::1],
    pressure:        float = 1.0,
    clamp_min:       float = 0.0,
    clamp_max:       float = 1.0,
    iterations:      int   = 1,
    normalize:       bool  = True,
)
```

### 10.4 `BrushUndoRecorder`

```python
recorder = BrushUndoRecorder(
    modified_buffer,          # float[:,::1]
    modified_vtx_indices_buffer = None,  # int[::1]
    modified_vtx_bool_buffer    = None,  # uchar[::1]
    undo_buffer                 = None,  # float[:,::1]
)

recorder.begin_stroke()
recorder.record_snapshot(record_indices)   # 仅首次触碰备份
recorder.end_stroke()
    -> (modified_indices, modified_channel_indices,
        old_sparse_ary,    new_sparse_ary)
```

### 10.5 `BrushMathEngine`（一般通过 Processor 调用）

| 方法 | 笔刷模式 |
| --- | --- |
| `_math_standard_stroke` | 0/1/2/3（Add/Sub/Replace/Multi），带防重叠 max_falloff |
| `_math_smooth` | 4 Smooth（拓扑邻居均值） |
| `_math_sharp` | 5 Sharp（远离 0.5） |
| `get_custom_array(verts, channels)` | 提取稀疏数据（零填充时返回全量） |
| `set_custom_array(values, blend_mode, alpha, weights, ...)` | 反向写回 |

### 10.6 `UtilBrushProcessor` / `SkinWeightProcessor`

```python
proc = UtilBrushProcessor(core, modified_buffer, idx_buf, bool_buf, undo_buf)
proc.begin_stroke()
proc.process_stroke(brush_stroke_context) -> (count, indices, modified_buffer)
proc.end_stroke() -> snapshot tuple

proc.get_custom_array(...) / set_custom_array(...)
proc.clear_buffer_sparse(vertex_indices=None)
proc.add_layer_weights(layer_weights, layer_mask, vertex_indices=None)

# 蒙皮专属
weight_proc = SkinWeightProcessor(core, mod_buf, idx_buf, bool_buf, locks_buf, undo_buf)
weight_proc.process_stroke(ctx, normalize=True)
weight_proc.normalize_weights(vertex_indices, priority_influence=-1)
```

### 10.7 `cTopologyCython`

```python
unique_edges_ctypes = compute_unique_edge_indices(tri_indices1D)
v2v_offsets, v2v_indices = build_v2v_adjacency(num_verts, edge_indices1D)
v2f_offsets, v2f_indices = build_v2f_adjacency(num_verts, tri_indices1D)
```

### 10.8 `cColorCython`

```python
render_heatmap(weights_1d, color_view)
render_gradient(weights_1d, color_view, color_a, color_b)
render_fill(color_view, color)
render_brush_gradient(color_view, hit_indices, hit_weights, hit_count, color_a, color_b)
offset_indices_direct(src_addr, dst_addr, count, offset)
```

### 10.9 `cBoundingBoxCython`

```python
(min_xyz, max_xyz) = compute_bbox_fast(points_1d, num_verts)
```

---

## 11. 笔刷参数（`BrushSettings`）

`src/cBrushSettings.py`，全局单例：

```python
from gskin.src.cBrushSettings import BrushSettings

BrushSettings.radius                = 1.0
BrushSettings.strength              = 1.0
BrushSettings.iter                  = 10
BrushSettings.falloff_type          = 1     # 0:Linear 1:Airbrush 2:Solid 3:Dome 4:Spike
BrushSettings.mode                  = 0     # 0:Add 1:Sub 2:Replace 3:Multi 4:Smooth 5:Sharp
BrushSettings.brush_spacing_ratio   = 0.1
BrushSettings.use_surface           = True
```

UI 改这一处即可影响整个笔刷链路。

---

## 12. 编译指引

```bat
:: 在 src 目录执行（脚本会自己 cd 进 SRC_DIR）
mayapy E:\d_maya\gskin\_buildCython\build.py
```

脚本会：
1. 探测 `mayapy.exe` 作为编译器；
2. 自动收集 `*.pyx` + `*Cython.py`；
3. `setuptools + cythonize` 产出 `.cp310-win_amd64.pyd` 到 `src/`；
4. 清理 `.c / .cpp / .html / build/`。

`pyproject.toml` 通过 `uv` 管理依赖：

```toml
[project]
name = "gskin"
requires-python = ">=3.10"
dependencies = ["cython>=3.2.5", "maya-stubs>=0.4.2"]
```
