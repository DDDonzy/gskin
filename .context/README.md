# `.context/` · gskin 项目知识库

按 `.clinerules` 约定生成的项目档案。

## 文档清单

| 文件 | 说明 |
| --- | --- |
| `project_context.md` | 核心业务蓝图：定位、技术栈、术语、模块清单、关键设计决策、TODO |
| `architecture.md` | 架构详解：分层模型、数据流、内存桥接、节点属性图、性能要点 |
| `api_reference.md` | API 速查：插件、节点、`FnCSkinDeform`、Cython 内核签名 |

## 流程图清单

| 文件 | 说明 |
| --- | --- |
| `project_flow.mmd` | 总体业务流（启动 → 创建蒙皮 → 求值 → 编辑 → 渲染） |
| `flow_deform.mmd` | 蒙皮 deform 求值（DirtyEvent → 矩阵预计算 → LBS 全量/局部） |
| `flow_brush.mmd` | 笔刷绘制（Hover/Press/Drag/Release，Raycast → Falloff → Math → Undo） |
| `flow_render.mmd` | VP2 直写渲染（compute → GPU buffer 直写 → 6 个 RenderItem） |

## 阅读建议

- 第一次接触项目：先读 `project_context.md`，再扫一遍 `project_flow.mmd`。
- 二次开发：跳到 `api_reference.md`，按需求查公共 API。
- 性能调优 / 排错：`architecture.md` 第 8 节性能要点 + 三张专题流程图。

> 流程图使用 `mermaid` 语法，VS Code 可装 `Markdown Preview Mermaid Support` 直接预览。
