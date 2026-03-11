# 项目学习题库（按知识点展开）

> 用于 `project-learner` 的“问答检验”模式。  
> 每个知识点提供 2 个主问题角度：一个偏事实/实现，一个偏设计/取舍。  
> 用户重复练同一知识点时，优先换题角度。

## D1 项目定位与入口

| ID | 主问题 A | 主问题 B | 追问方向 |
|----|----------|----------|----------|
| D1.1 | 你会怎么解释这个仓库为什么已经不是单纯 MCP/RAG demo？ | 如果你要向面试官讲这个项目的“演进”，主线会怎么讲？ | 旧叙事为什么不够准确 |
| D1.2 | `run_server.py` 和 `src/server/app.py` 分别负责什么？ | 为什么真正的主入口必须从 app 装配层去看，而不是只看脚本名？ | 入口和装配边界 |
| D1.3 | `main.py` 和真实 MCP 实现的关系是什么？ | 为什么 `pyproject.toml` 里的 `mcp-server` 仍然会误导初学者？ | 历史残留如何识别 |
| D1.4 | 默认 collection 和 auto_ingest_dir 是在哪里配置的？ | 自动 ingest 为什么放在启动阶段而不是第一次提问时才触发？ | 启动时机与体验 |
| D1.5 | `docs/`、`data/`、`config/` 在当前项目里分别承载什么？ | 如果你第一次接手这个仓库，为什么先看目录现实而不是先看 README？ | 代码与运行态的关系 |

## D2 Agent 运行时与流式链路

| ID | 主问题 A | 主问题 B | 追问方向 |
|----|----------|----------|----------|
| D2.1 | ConversationStore 和 SystemPromptBuilder 分别做什么？ | 为什么 prompt 组装不直接塞在 `Agent.chat()` 里写死？ | 职责拆分 |
| D2.2 | `Agent.chat()` 到 `_tool_loop()` 的调用链是怎样的？ | 这个 Agent 为什么适合用 ReAct 风格，而不是一次性把所有逻辑塞进单个 prompt？ | 多轮工具编排 |
| D2.3 | StreamEvent 有哪些核心类型？ | 为什么 SSE 在这里是合适的默认方案？ | 事件设计与协议选择 |
| D2.4 | hooks 和 middleware 在代码里是怎么串起来的？ | 为什么要把生命周期 hook 和 LLM middleware 分成两种扩展点？ | 可扩展性边界 |
| D2.5 | `_post_message_tasks()` 和 `flush()` 各自解决什么问题？ | 为什么会话保存放到后台任务里，而不是在主链路同步阻塞？ | 一致性与响应速度 |

## D3 学习工具与课程闭环

| ID | 主问题 A | 主问题 B | 追问方向 |
|----|----------|----------|----------|
| D3.1 | `knowledge_query` 的入参与出参大致是什么？ | 为什么它是课程学习工具链里最核心的检索工具？ | 工具边界 |
| D3.2 | `document_ingest` 在对话型系统里扮演什么角色？ | 为什么导入工具要和 semantic cache 有联动？ | 导入后系统一致性 |
| D3.3 | `review_summary` 和普通问答的输出目标有什么差别？ | 为什么复习摘要必须基于检索结果而不是完全依赖模型先验？ | 结构化复习 |
| D3.4 | `quiz_generator` 和 `quiz_evaluator` 是怎么形成练习闭环的？ | 为什么出题要优先题库，再退到 RAG，再退到 LLM？ | 质量优先级 |
| D3.5 | ToolRegistry 在 app 装配里是怎么被用起来的？ | 为什么工具 schema 要显式暴露给 Agent？ | tool calling 机制 |

## D4 检索查询链路

| ID | 主问题 A | 主问题 B | 追问方向 |
|----|----------|----------|----------|
| D4.1 | QueryRouter 和 Semantic Cache 分别在哪一层介入？ | 为什么路由和缓存都放在 `knowledge_query` 这一侧更合理？ | 编排层与检索层分界 |
| D4.2 | rewrite、conversation-aware rewrite、HyDE、multi-query 在这个项目里怎么组合？ | 哪些增强手段应该默认开，哪些更适合按需打开？ | 质量与成本平衡 |
| D4.3 | `HybridSearch` 的主链路是什么？ | 为什么它更像“检索编排器”，而不是完整 RAG 的全部？ | 模块定位 |
| D4.4 | rerank、MMR、min_score、冲突检测各自处理的是什么问题？ | 如果这些后处理串得不对，最容易出现什么副作用？ | 排序与过滤顺序 |
| D4.5 | `use_parent`、引用和结果格式化在回答阶段有什么作用？ | 为什么“命中 child、返回 parent”是常见但容易被误解的策略？ | 精确命中 vs 完整上下文 |

## D5 数据摄取与索引

| ID | 主问题 A | 主问题 B | 追问方向 |
|----|----------|----------|----------|
| D5.1 | loader 和 source_type 推断是怎么配合的？ | 为什么课程学习场景不能把所有文档都当成同一种普通文本？ | 资料类型差异 |
| D5.2 | Parent-Child 和 QuestionParser 分别解决什么问题？ | 为什么题库解析是课程场景里的加分项，而不是可有可无？ | 课程导向设计 |
| D5.3 | transform 阶段里有哪些增强器？ | 为什么这些增强不应该简化成只有 metadata enrichment？ | 语义质量与多模态 |
| D5.4 | dense/sparse 编码和写库是怎么协同的？ | 为什么不能只保留向量库、把 BM25 和图片索引都删掉？ | 多索引协同 |
| D5.5 | DocumentManager 的 list/detail/delete 三个方向分别做什么？ | 为什么 delete_document 要跨多个存储协调，而不是只删 Chroma？ | 一致性问题 |

## D6 Memory、Skills 与 Hooks

| ID | 主问题 A | 主问题 B | 追问方向 |
|----|----------|----------|----------|
| D6.1 | ContextEngineeringFilter 的 4 个 level 是怎么工作的？ | 为什么长上下文治理在学习 Agent 里不是“优化项”而是“必需项”？ | 上下文成本与稳定性 |
| D6.2 | StudentProfile、ErrorMemory、KnowledgeMap 的数据职责怎么分？ | 为什么这个项目不把所有长期记忆都揉进一个 store？ | 记忆分层 |
| D6.3 | Memory enhancer 和 review schedule 是如何影响当前对话的？ | 为什么主动复习推荐适合做成 hook，而不是只做成被动查询？ | 介入时机 |
| D6.4 | Skill registry、workflow、guardrails 是怎么在一次请求里同时起作用的？ | 为什么 skill 流程和安全防护不能混为一个机制？ | 流程控制与安全 |
| D6.5 | 课程内置 skill definitions 起什么作用？ | 为什么课程学习场景适合把常见学习动作抽成 skill？ | 学习引导模板化 |

## D7 配置、Provider 与存储

| ID | 主问题 A | 主问题 B | 追问方向 |
|----|----------|----------|----------|
| D7.1 | 为什么这个仓库同时有 typed settings 和原始 YAML dict？ | 这种双轨配置现实会给阅读和维护带来什么影响？ | 配置层历史与现实 |
| D7.2 | LLM、Embedding、Reranker、VectorStore 的工厂模式是怎么工作的？ | 为什么“换 provider 不改上层业务代码”在这个项目里很重要？ | 可插拔架构 |
| D7.3 | Chroma、BM25、SQLite memory、image storage 各自存什么？ | 如果强行把它们合成单库，最先出现的问题会是什么？ | 数据模型差异 |
| D7.4 | 默认课程数据和 collection 是怎么落在本地目录里的？ | 单课程到多课程扩展，哪一层会先出现组织复杂度？ | 多 collection 设计 |
| D7.5 | routing、semantic_cache、memory、guardrails 这些能力分别在哪些配置项上开启？ | 为什么有些开关适合默认打开，有些更适合按成本按需启用？ | 配置粒度与默认值 |

## D8 Web 与 MCP 接口层

| ID | 主问题 A | 主问题 B | 追问方向 |
|----|----------|----------|----------|
| D8.1 | `routes.py`、`models.py`、`chat_handler.py` 在 Web 层怎么分工？ | 为什么对话流处理不直接写在路由函数里？ | Web 分层 |
| D8.2 | 前端是怎么消费 SSE 事件流的？ | 为什么这个前端结构虽然简单，但对当前学习产品已经够用？ | 前后端契合度 |
| D8.3 | MCP server 和 protocol handler 是怎么把工具注册起来的？ | 为什么 stdio 模式下 stdout/stderr 的边界很关键？ | 协议实现 |
| D8.4 | `agent-server` 和 `mcp-server` 这两个脚本为什么不能混着理解？ | 为什么说打包入口和真实实现之间仍有历史偏差？ | CLI 与实现现实 |
| D8.5 | 上传文件、静态资源和首页是怎么挂到 FastAPI 上的？ | 如果以后要做更完整的前端产品层，这一层会先怎么演进？ | 服务器与静态资源 |

## D9 质量保障与可观测性

| ID | 主问题 A | 主问题 B | 追问方向 |
|----|----------|----------|----------|
| D9.1 | Query/Ingestion trace 分别记录什么？ | 为什么 trace 对 RAG/Agent 系统尤其重要？ | 白盒观测价值 |
| D9.2 | Dashboard 的页面和 services 是怎么组织的？ | 为什么 dashboard 适合独立成 observability 层？ | 可视化分层 |
| D9.3 | EvalRunner 和 evaluator 体系是怎么配合的？ | 为什么评估必须从“单次感觉”走向“批量回归”？ | 质量评估 |
| D9.4 | 当前测试结构大致怎么分层？ | 为什么当前环境缺 `mcp` 依赖会影响 smoke imports，但不等于 Web 主链路坏了？ | 环境 caveat |
| D9.5 | golden set、trace、tests、dashboard 这几块如何形成质量闭环？ | 如果少掉其中一个环节，调优过程最容易变成什么样？ | 工程闭环 |
