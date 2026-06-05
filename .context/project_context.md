# gskin · 项目核心蓝图

> Maya 高性能蒙皮变形器 + 实时权重笔刷工具集
> 基于 Cython 加速、零拷贝内存桥接、GPU 直写渲染。

---

## 1. 项目定位

`gskin` 是一个 Maya 自研插件，用来**替代/增强** Maya 原生 `skinCluster`，
聚焦于权重编辑工作流的**性能极限**与**编辑体验**：

- **替代 LBS 内核**：用 Cython + nogil 实现的线性混合蒙皮（Linear Blend Skinning），
  支持全量和局部稀疏两种模式。
- **图层化权重**：原生支持 `Layer` + `Mask` + `LockInfluences`，每层有独立权重数据，
  通过 mask 加权合成最终权重。
- **实时权重笔刷**：自定义 `MPxContext`，包含射线检测、表面/体积衰减、
  数位板压感、轨迹插值、Undo 快照、局部预览等。
- **VP2 直写渲染**：自定义 `MPxSurfaceShape` + `MPxSubSceneOverride`，
  Cython 直接写入 GPU 显存，用于热力图/遮罩/笔刷高亮可视化。

---

## 2. 技术栈

| 层 | 技术 |
| --- | --- |
| 宿主 | Maya 2024（Python 3.10、API 1.0 + API 2.0 混用） |
| 语言 | Python 3.10+、Cython 3.x（Pure Python 模式） |
| 编译 | `uv` 管理依赖、`mayapy.exe` 驱动 setuptools + cythonize |
| 并发 | OpenMP（`/openmp`）、`@cython.nogil` |
| 渲染 | Maya VP2 (`MPxSubSceneOverride`)、`MVertexBuffer` 直写 |
| UI | PySide2（仅用于压感事件过滤） |

---

## 3. 关键术语速查

| 术语 | 含义 |
| --- | --- |
| **OM1 / OM2** | Maya Python API 1.0（`maya.OpenMaya`）/ 2.0（`maya.api.OpenMaya`） |
| **DG / Parallel** | Maya 节点求值的两种调度模式（脏数据传播 vs. 并行评估） |
| **DirtyEvent** | 本项目对脏数据触发回调的封装（`MDirtyEvent.py`） |
| **MRegistry** | 全局节点实例注册表，绕过 OM1/OM2 互不相通的 `userNode()` |
| **CSR** | Compressed Sparse Row 邻接表（`v2v` / `v2f`） |
| **Layer** | 权重图层。每层有 `weights` + `enabled` + `lockInfluences`，被 `mask` 加权 |
| **Falloff** | 笔刷衰减模式（Linear/Airbrush/Solid/Dome/Spike） |
| **Raycast 双轨制** | 微缓存命中 → V2F 局部扫描 → 全网格盲扫，三层兜底 |
| **世代掩码** | `epoch` 自增替代 `memset`，O(1) 清空大数组 |
| **MFloatArrayProxy** | 把 `MVectorArray` 重新解释为 `float[]`，写入 24-byte Header |

---

## 4. 模块清单（一句话职责）

### `plugin/` — Maya 插件注册入口

| 文件 | 职责 |
| --- | --- |
| `cSkinPlugin.py` | 注册 `cSkinDeformer` 变形器节点（OM1） |
| `cBrushPlugin.py` | 注册 `cBrushCtx` 笔刷上下文命令（OM2） |
| `subPlugin.py` | 注册 `triangleShape` 自定义形状 + `MPxSubSceneOverride`（OM2） |
| `testPlugin.py` | 内存调试用最小节点 |

### `src/` — 实现层

#### Maya 工具基础设施（`M*.py`）

| 文件 | 职责 |
| --- | --- |
| `MRegistry.py` | 全局 `WeakValueDictionary` 注册节点实例，OM1 ↔ OM2 互访 |
| `MDirtyEvent.py` | 封装脏标记与回调，统一 DG/Parallel 模式入口 |
| `MTopologyContext.py` | 解析 mesh 拓扑（顶点/边/面/三角化/CSR 邻接） |
| `MFloatArrayProxy.py` | 把 `MVectorArray` 当作连续 `float[]` 内存使用（核心黑魔法） |
| `MWeightsHandle.py` | 在 `MFloatArrayProxy` 上挂 `(num_vertices, num_influences)` 元数据 |
| `MProfiler.py` / `_cProfiler*` | Maya 原生 Profiler 接口的封装 |

#### Cython 算法内核（`c*Cython.py`/`.pyd`）

| 文件 | 职责 |
| --- | --- |
| `cSkinDeformCython.py` | LBS 解算内核（全量 + 局部稀疏） + 矩阵预计算 |
| `cBrushCore2Cython.py` | 笔刷引擎全套：Raycast / Falloff / Recorder / MathEngine / Processor |
| `cTopologyCython.py` | 极速构建 v2v / v2f CSR 邻接表 |
| `cColorCython.py` | 热力图、双色渐变、笔刷散点高亮的 nogil 染色器 |
| `cBoundingBoxCython.py` | 一维 float 数组的极速 AABB 计算 |
| `_cProfilerCython.py` | Maya Native Profiler 的 Cython 封装 |

#### 业务编排层

| 文件 | 职责 |
| --- | --- |
| `cSkinDeform2.py` | `CSkinDeform` 节点主体 + `CSkinContext` 数据中心 + `WeightsLayerManager` + `FnCSkinDeform` 公共 API |
| `cBrush.py` | `WeightBrushContext` 视口交互 + `WeightBrushContextCmd` |
| `cBrushSettings.py` | 笔刷半径/强度/模式等 UI 参数单例 |
| `cBrushInterpolator.py` | 笔刷轨迹插值（线性 / Catmull-Rom，等距重采样） |
| `cBrushTabletInput.py` | PySide2 全局事件过滤器，拦截 Wacom 压感 |
| `cBufferManager.py` | ctypes 万能内存路由（auto/from_ptr/from_buffer/from_list/allocate） |
| `cWeightsManager.py` | 旧版 layer 数据结构占位（实际 active 在 `cSkinDeform2.py`） |

### `_buildCython/`、`_debug/`、`old/`

| 目录 | 作用 |
| --- | --- |
| `_buildCython/build.py` | mayapy 子进程驱动 cythonize，产出 `.pyd` |
| `_debug/gskinReload.py` | 一键重载所有 src 模块和插件 |
| `_debug/test_skin.py` | 从 maya 文件初始化 cSkinDeform 的 demo |
| `old/` | 上一代实现（`CythonSkinDeformer` 等），仍被 `cBrush.py` 经 `subPlugin.py` 引用 |

---

## 5. 关键设计决策

### 5.1 零拷贝内存桥接
- `MFloatArrayProxy` 把 `MVectorArray` 强转为连续 `float`，在前 24B 写入 `(count, byte_size)` Header；
  上层 `MWeightsHandle` 再额外用第 16 字节起的 8 字节存 `(num_vertices, num_influences)`。
- `BufferManager` 提供 `from_ptr / from_buffer / from_ctypes / from_list / allocate` 多路径，
  统一 ctypes、array、memoryview、Maya API 数组。
- Cython memoryview 经 `cython.address(view[0])` 直接拿到 C 指针，全程 `nogil`。

### 5.2 双调度脏标记
`DirtyEvent` 同时支持 `setDependentsDirty(plug)` 和 `preEvaluation(evaluationNode)`，
DG 与并行模式都能命中。`compute()` / `deform()` 里逐个 `event.execute()`，
未脏的事件直接跳过——把 Maya 节点的属性变化转译为**只在变化时刷新缓存**。

### 5.3 双轨 Raycast
笔刷 raycast 优先用上一帧命中的三角面 + V2F 邻接做**微缓存扫描**，
无果再扫一帧 falloff 命中点的 V2F，最后才**全网格盲扫**。
配合**世代掩码**（`raycast_epoch++`）在 nogil 下避免重复测试。

### 5.4 Layer 合成
`CSkinDeform.layers` 是一个 compound 数组，每元素含
`name / enabled / weights / lockInfluences`，外层还有 `layersMaskData / layersLockMask`。
`WeightsLayerManager` 同时支持节点内部（`outputArrayValue`）和外部（`MPlug.asMDataHandle`）两种取数路径。

### 5.5 Undo 快照
`BrushUndoRecorder` 用「掩码 + 索引池 + undo buffer」三件套：
- `record_snapshot` 仅在第一次触碰时备份 → O(1) 重复进入；
- `end_stroke` 二次扫描压缩出实际变动的 channels（**通道压缩**），返回稀疏增量。

### 5.6 GPU 直写
`subPlugin.py` 在 `update()` 用 `MVertexBuffer.acquire/commit` 拿到 GPU 指针，
通过 ctypes 包成 memoryview 后直接喂给 Cython 染色器，**全程不经 numpy 不经拷贝**。
渲染热力图/遮罩/权重渐变/笔刷高亮共 6 个 RenderItem。

### 5.7 局部预览路径
笔刷涂抹时 `cSkin.fast_preview_deform(active_indices)` 走 `_run_partial_skinning_core`，
只解算被笔刷影响的顶点，并设 `isDirty_brushFastPreview` 让渲染节点直接走快路径，
**完全绕开 Maya DG**。

---

## 6. 已知 TODO（来自代码注释）

| 位置 | 内容 |
| --- | --- |
| `cSkinDeform2._update_mesh` | 避免每次更新 topology |
| `cSkinDeform2._update_*_matrix` | 避免每帧申请新内存池 |
| `cSkinDeform2._update_deform_matrices` | 同上，复用 rotate/translate 内存 |
| `cSkinDeform2.fast_preview_deform` | 局部蒙皮算法尚未填充实现体 |
| `cSkinDeform2.create_cSkinDeform_from_skinCluster` | 权重 list 转换太慢 |
| `MTopologyContext.get_*` | Python 实现的 CSR 太慢，已被 `cTopologyCython` 替代但旧调用未清 |
| `cWeightsManager.py` | 仅占位实现，未集成 |
| `FnCSkinDeform.set_weights` | 暂不支持 undo / redo |

---

## 7. 入口点速查

| 操作 | 命令 / 入口 |
| --- | --- |
| 编译 Cython | `mayapy _buildCython/build.py`（自动 build_ext --inplace） |
| 重载所有 | `gskin._debug.gskinReload.reload_all_plugins()` |
| 创建蒙皮 | `FnCSkinDeform.create_cSkinDeform_from_skinCluster("skinCluster1")` |
| 启用笔刷 | `cmds.setToolTo(cmds.cBrushCtx())` |
| 节点 ID | `CSkinDeform = 0x00080033`、`triangleShape = 0x80089`、`MemTestNode = 0x87001` |
