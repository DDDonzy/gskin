# gskin · 架构详解

> 阐述本仓库的分层模型、关键数据流、内存桥接黑魔法，以及节点属性图。

---

## 1. 分层架构

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 5 · Maya Plugin Registry (plugin/)                        │
│  cSkinPlugin │ cBrushPlugin │ subPlugin │ testPlugin             │
└────────────────────────────┬─────────────────────────────────────┘
                             │ initializePlugin / registerNode
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  Layer 4 · Maya Node / Context Orchestration (src/c*.py)         │
│  CSkinDeform     · WeightBrushContext · TriangleShape            │
│  FnCSkinDeform   · WeightsLayerManager · TriangleOverride        │
└────────────┬───────────────────────┬─────────────────────────────┘
             │                       │
             ▼                       ▼
┌─────────────────────────┐  ┌──────────────────────────────────────┐
│  Layer 3 · Maya Bridge  │  │  Layer 3 · Brush Sub-system          │
│  M*.py                  │  │  cBrushSettings / Interpolator /     │
│  MRegistry              │  │  TabletInput                         │
│  MDirtyEvent            │  └──────────────────┬───────────────────┘
│  MTopologyContext       │                     │
│  MFloatArrayProxy       │                     │
│  MWeightsHandle         │                     ▼
│  MProfiler              │  ┌──────────────────────────────────────┐
└──────────┬──────────────┘  │  Layer 2 · Cython Compute Kernels    │
           │                 │  cSkinDeformCython · cBrushCore2     │
           │                 │  cTopologyCython · cColorCython      │
           │                 │  cBoundingBoxCython · _cProfiler     │
           ▼                 └──────────────────┬───────────────────┘
┌──────────────────────────────────────────────┐│
│  Layer 1 · Memory Foundation                 ││
│  cBufferManager (ctypes router)              ◄┘
│  ctypes · memoryview · Maya raw pointers     │
└──────────────────────────────────────────────┘
```

各层只能向下依赖（从上到下：Plugin → Node → Bridge → Kernel → Memory）。
Brush 子系统额外依赖 Maya Bridge 的 `MRegistry` 拿到 `CSkinDeform` 实例。

---

## 2. 核心数据流

### 2.1 蒙皮求值（Maya DG → Python → Cython → 顶点输出）

```
Bone Matrices ─┐
BindPreMatrix ─┼─ DirtyEvent ─► CSkinContext ─► cSkinDeformCython.cal_deform_matrices
GeomMatrix    ─┘                     │                    │
                                     │             rotate_matrix  (3x3)
input geom ───► MFnMesh ─► TopologyContext         translate_vec  (1x3)
                                     │                    ▼
weights (MVectorArray) ─► MWeightsHandle ──► cSkinDeformCython.run_skinning_core
                                                              │
                                                              ▼
                                                  output geom (MFnMesh)
```

- 所有上游数据都通过 `DirtyEvent.execute()` **按需更新**到 `CSkinContext.*`。
- `CSkinContext` 字段都是 `memoryview`，内存所有权在 Python，但指针给 Cython。

### 2.2 笔刷涂抹

```
Mouse / Tablet ─► WeightBrushContext.doDrag
                     │
                     ▼
       LinearStrokeInterpolator (等距重采样)
                     │
                     ▼  for each (x, y, p)
       _process_single_point ─► engine.raycast (双轨制)
                                    │ hit_pos
                                    ▼
                            engine.calc_brush_falloff
                                    │ active_indices, falloff
                                    ▼
                            BrushStrokeContext (打包)
                                    │
                                    ▼
                            paint_stroke_coroutine.send(ctx)
                                    │
                                    ▼  iterations 次
                            BrushMathEngine._execute_math_step
                            (Add / Sub / Replace / Multi / Smooth / Sharp)
                                    │
                                    ▼
                            cSkin.fast_preview_deform(active_indices)
                                    │
                                    ▼
                            isDirty_brushFastPreview = True
                                    │
                                    ▼
                            TriangleOverride.update (走 fast-path 直渲)
```

### 2.3 渲染（VP2 直写）

```
TriangleShape.compute (DG dummy 求值)
   │
   ├─ 读 channelBox 属性 → render_data.line_width / colors / mode...
   │
   └─ _update_from_cSkin (非求值, 拿 MRegistry 实例)
        │
        ▼
    render_data.vertices_pos / face_indices / line_indices ← cSkin.mesh_context
    render_data.paint_weights_view                         ← cSkin.get_active_paint_weights
    render_data.brush_hit_indices / brush_hit_weights       ← cSkin.brush_context

TriangleOverride.update
   │
   ├─ _init_render_items     (6 个 RenderItem)
   ├─ _init_gpu_buffers      (vertex + 3 套 color + 3 套 index)
   │
   ├─ _sync_topology_buffers (memmove 拷顶点位置 + 索引)
   │
   ├─ _sync_color_buffers
   │     └─ gpu_write_session (acquire) ─► _calculate_colors_direct
   │           ├─ render_fill        (线/点底色)
   │           ├─ render_gradient    (mask / weights 双色)
   │           ├─ render_heatmap     (热力图)
   │           └─ render_brush_gradient  (笔刷高亮散点)
   │
   ├─ _update_render_items_state (开关 + 着色器参数)
   │
   └─ setGeometryForRenderItem (绑定 buffer)
```

---

## 3. 内存桥接黑魔法

### 3.1 `MFloatArrayProxy` 内存布局

Maya 节点属性不直接暴露连续 `float[]`，作者借 `MVectorArray`（连续 double）做载体。
每个 `MVector` 占 24B，可承载 6 个 `float`，第一个 vector 拿来当 Header：

```
┌──────────────────────────────────────────────────────────────────────┐
│  Header[0] (size_t)  │  Header[1] (size_t)  │  Header[2]  │  DATA    │
│  float_count         │  data_byte_size      │  reserved   │  …       │
│       8B             │        8B            │     8B      │ 4B × N   │
│ ◄─── 24-byte Header (1st MVector occupied) ──────────────► │ Payload │
└──────────────────────────────────────────────────────────────────────┘
```

`MWeightsHandle` 进一步在 Header[2] 的低 8B 写入 `(num_vertices, num_influences)`，
做到一次 `setMObject` 同时携带 `(count, shape)` 元数据。

### 3.2 `BufferManager` 路由

```
auto(data) ──────┬──► from_ptr     (int address)         零拷贝
                 ├──► from_buffer  (array / memoryview)  零拷贝
                 ├──► from_list    (list / Maya 数组)    显式拷贝
                 └──► from_ctypes  (Cython 返回的数组)   零拷贝接管
```

所有路径最终统一为：
- `instance.ctypes` 持有 `(c_type * N)` 数组保证 Python GC 不回收；
- `instance.ptr` 是 `ctypes.addressof()`，可喂给 Cython 或 GPU；
- `instance.view` 是按 `format_char` 重塑的 `memoryview`。

### 3.3 GPU 写显存

```python
gpu_ptr = gpu_buffer.acquire(element_count, True)       # Maya 给一段 GPU 映射地址
ArrayType = ctypes.c_float * (element_count * 4)
buf = ArrayType.from_address(int(gpu_ptr))              # 强转 ctypes
view = memoryview(buf).cast("B").cast("f", shape=(N,4)) # 包成 2D memoryview
yield view                                              # 喂给 Cython 染色
gpu_buffer.commit(gpu_ptr)                              # 提交
```

`gpu_write_session` 把上述四步包进 `contextmanager`，调用方无感：

```python
with gpu_write_session(self.color_buffer_face, vtx_count) as face_view:
    cColor.render_heatmap(weights_view, face_view)
```

---

## 4. `CSkinDeform` 节点属性图

```
                     ┌──────────────────────────────────┐
                     │       cSkinDeformer 节点          │
                     │       (CSkinDeform.NODE_ID)       │
                     └──────────────────────────────────┘
                                    │
   ┌────────────────────────────────┼─────────────────────────────────────┐
   ▼                                ▼                                     ▼
[Inputs]                       [Layer Tree]                          [Paint Cursor]
- input[].inputGeometry        - layers (compound)                   - currentPaintLayer
- envelope                       └ layersMaskData                    - currentPaintInfluence
- matrix[]      (bones)          └ layersLockMask                    - currentPaintMask
- bindPreMatrix[]                └ layerData[] (compound array)
- geomMatrix                          ├ layerName
- weightsData                         ├ layerEnabled
- forceDirty                          ├ layerWeightsData
                                      └ layerLockInfluences
                                    │
                                    ▼
                           [Output]: outputGeometry[]

属性影响关系 (attributeAffects):
  几乎所有 input + paint + layer 子属性 ⟶ outputGeometry
```

### 4.1 求值入口的两条岔路

```
┌─ Maya DG 求值 outputGeometry ─► compute() ─► deform() ─► event.execute() x N
│                                                     ─► run_skinning_core (全量)
│
└─ Brush 实时预览 ─► fast_preview_deform(indices)
                            └─► run_partial_skinning_core (稀疏)
                            └─► isDirty_brushFastPreview = True
                            └─► TriangleOverride.update fast-path 渲染
```

---

## 5. `triangleShape` 节点属性图

```
                           ┌────────────────────────────┐
                           │   triangleShape 节点       │
                           │   (TriangleShape.TYPE_ID)  │
                           └────────────────────────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬───────────────┐
        ▼              ▼               ▼               ▼               ▼
   [Source Link]  [Render Mode]   [Style]         [Color (8)]    [Default Solid (3)]
   - inputMesh    - renderMode    - lineWidth     - wireColor    - defaultDrawFaces/Lines/Points
                  (Alpha/Heat)    - pointSize     - vertexColor  - defaultFace/Line/PointColor
                                  - drawFaces     - mask/weights
                                  - drawLines       /brush
                                  - drawPoints     RemapA/B color
                                       │
                                       ▼
                                 outDummy (隐藏触发输出)
                                       │ 所有上面属性都 attributeAffects
                                       ▼
                                  compute()
                                       │
                                       ▼
                              TriangleOverride.update
```

---

## 6. 关键类关系图

```
            ┌──────────────────┐
            │   FnCSkinDeform  │ ◄── 用户公共 API（外部脚本）
            └────────┬─────────┘
                     │ proxy
                     ▼
            ┌──────────────────┐         ┌────────────────────┐
            │   CSkinDeform    │◀────────│   MRegistry        │
            │  (MPxDeformer)   │ register│ (WeakValueDict)    │
            └────────┬─────────┘         └────────────────────┘
                     │ owns
        ┌────────────┼────────────────────┬─────────────────────┐
        ▼            ▼                    ▼                     ▼
 ┌────────────┐ ┌────────────┐  ┌──────────────────┐  ┌────────────────┐
 │ CSkinContext│ │ DirtyEvent │  │ WeightsLayerMgr │  │ MWeightsHandle │
 │  (slots)   │ │  (×8)      │  │ + WeightLayerItem│  │ ◄ MFloatProxy  │
 └────────────┘ └────────────┘  └──────────────────┘  └────────────────┘
        │
        │ memoryviews
        ▼
 ┌────────────────────────────────────────────────────────────────────┐
 │            cSkinDeformCython (LBS kernels, nogil)                  │
 │   cal_deform_matrices · run_skinning_core · run_partial_skinning   │
 └────────────────────────────────────────────────────────────────────┘


            ┌──────────────────────────┐
            │  WeightBrushContext      │  ◄── Maya MPxContext
            │  (MPxContext)            │
            └──────────┬───────────────┘
                       │ uses
        ┌──────────────┼──────────────┬─────────────────┐
        ▼              ▼              ▼                 ▼
  ┌─────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────┐
  │ CSkinDeform │ │TabletTrack │ │StrokeInterp│ │ BrushSettings  │
  │  (via       │ │  (Wacom    │ │ (line /    │ │   (dataclass   │
  │   MRegistry)│ │   filter)  │ │  Catmull)  │ │    singleton)  │
  └──────┬──────┘ └────────────┘ └────────────┘ └────────────────┘
         │
         ▼  cSkin.brush_engine (CoreBrushEngine, in cBrushCore2Cython)
   ┌───────────────────────────────────────────────────────────────────┐
   │                    cBrushCore2Cython.py                           │
   │ CoreBrushEngine ─► raycast / calc_brush_falloff                   │
   │ BrushStrokeContext (UI 配置 + 物理内存)                            │
   │ BrushUndoRecorder ─► begin / record_snapshot / end                │
   │ BrushMathEngine ─► _math_standard / _math_smooth / _math_sharp    │
   │ UtilBrushProcessor / SkinWeightProcessor (调度 + 归一化)            │
   └───────────────────────────────────────────────────────────────────┘
```

---

## 7. Cython 内核 API 一览

### 7.1 `cSkinDeformCython`

| 函数 | 用途 |
| --- | --- |
| `cal_deform_matrices` | 由 4×4 骨骼矩阵 → 3×3 旋转 + 1×3 平移（含 GeoMatrix 路径） |
| `run_skinning_core` | 全量 LBS（遍历所有顶点） |
| `run_partial_skinning_core` | 局部 LBS（遍历给定 vertex_indices） |

### 7.2 `cBrushCore2Cython`

| 类 / 函数 | 用途 |
| --- | --- |
| `CoreBrushEngine.lock_mesh / unlock_mesh` | 涂抹时冻结网格快照 |
| `CoreBrushEngine.raycast` | 双轨 + 盲扫 三层兜底的射线求交 |
| `CoreBrushEngine.calc_brush_falloff` | 体积模式（胶囊体） / 表面模式（BFS）二选一，五种衰减曲线 |
| `BrushStrokeContext` | 一次涂抹的所有静态/动态参数 |
| `BrushUndoRecorder` | 三件套：bool 掩码 / 索引池 / undo buffer |
| `BrushMathEngine` | 6 种笔刷模式数学：Add/Sub/Replace/Multi/Smooth/Sharp |
| `UtilBrushProcessor` | 通用调度器（process_stroke / get/set_custom_array / clear/add layer） |
| `SkinWeightProcessor` | 蒙皮专属（继承 + 归一化 + lockInfluences） |

### 7.3 `cTopologyCython`

| 函数 | 用途 |
| --- | --- |
| `compute_unique_edge_indices` | 三角面 → 全局去重无向边 |
| `build_v2v_adjacency` | v2v CSR（顶点-顶点） |
| `build_v2f_adjacency` | v2f CSR（顶点-三角面） |

### 7.4 `cColorCython`

| 函数 | 用途 |
| --- | --- |
| `render_heatmap` | 5 段静态梯度（蓝→绿→黄→橙→红） |
| `render_gradient` | 任意双色线性插值 |
| `render_fill` | 单色填充 |
| `render_brush_gradient` | 按 hit_indices 散点高亮 |
| `offset_indices_direct` | 索引平移（ctypes 直写） |

### 7.5 `cBoundingBoxCython`

| 函数 | 用途 |
| --- | --- |
| `compute_bbox_fast` | 一维 float 顶点数组的极速 AABB |

---

## 8. 性能要点对照表

| 优化策略 | 对应代码 |
| --- | --- |
| nogil + cdivision + boundscheck=False | 所有 Cython 内核 |
| 世代掩码代替 memset | `vertices_epochs` / `faces_epochs` / `raycast_epoch` / `brush_epoch` |
| 双轨 Raycast | `CoreBrushEngine.raycast` Plan 1/2/3 |
| 内存零拷贝 | `MFloatArrayProxy` / `BufferManager.from_*` |
| GPU 显存直写 | `gpu_write_session` + `cColor.*` |
| 局部蒙皮 | `run_partial_skinning_core` + `fast_preview_deform` |
| 通道压缩 Undo | `BrushUndoRecorder.end_stroke` |
| 倍增式 fill | `BufferManager.fill` 非零路径 |
| AABB 早裁 | `calc_brush_falloff` Volume 模式 |
| 拓扑 BFS | `calc_brush_falloff` Surface 模式 |
| 防重叠 max_falloff | `BrushMathEngine._math_standard_stroke` |
