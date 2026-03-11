# 项目学习地图（按当前代码组织）

> 用于 `project-learner` 选择学习域和阅读路径。  
> 当前版本扩展为 9 个知识域、45 个知识点。

## 推荐阅读顺序

1. D1 项目定位与入口
2. D2 Agent 运行时与流式链路
3. D3 学习工具与课程闭环
4. D4 检索查询链路
5. D5 数据摄取与索引
6. D6 Memory、Skills 与 Hooks
7. D7 配置、Provider 与存储
8. D8 Web 与 MCP 接口层
9. D9 质量保障与可观测性

## D1 项目定位与入口

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D1.1 当前产品定位与演化 | 理解为什么它现在更像课程学习 Agent | `DEV_SPEC.md`, `src/server/app.py` |
| D1.2 Web 主入口与 app 装配 | 看服务启动时怎么把 Agent、Tools、Memory、RAG 装起来 | `run_server.py`, `src/server/app.py` |
| D1.3 MCP 实现与历史占位入口 | 区分真实 MCP 实现和 `main.py` 历史入口 | `src/mcp_server/server.py`, `src/mcp_server/protocol_handler.py`, `main.py`, `pyproject.toml` |
| D1.4 默认配置与启动自动 ingest | 看默认课程、默认 collection、自动入库目录 | `config/settings.yaml`, `src/server/app.py` |
| D1.5 当前数据资产与目录布局 | 看 `docs/`、`data/`、本地 collection 与课程资料布局 | `docs/computer_internet/`, `data/`, `config/settings.yaml` |

## D2 Agent 运行时与流式链路

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D2.1 Conversation 与 prompt 组装 | 会话如何持久化，system prompt 如何构造 | `src/agent/conversation.py`, `src/agent/prompt_builder.py` |
| D2.2 ReAct tool loop 主流程 | Agent 何时调工具、何时结束、如何处理工具结果 | `src/agent/agent.py` |
| D2.3 StreamEvent 与 SSE 输出 | 后端怎样把模型增量输出和工具事件发给前端 | `src/agent/types.py`, `src/server/chat_handler.py` |
| D2.4 Hooks 与 middleware 扩展点 | 生命周期 hook、retry、reflection、rate limit 如何插进去 | `src/agent/hooks/*.py` |
| D2.5 后台保存、flush 与错误处理 | 会话保存、background tasks、flush 与异常路径怎么兜底 | `src/agent/agent.py` |

## D3 学习工具与课程闭环

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D3.1 knowledge_query 问答主工具 | 用户问题如何进入检索与回答链路 | `src/agent/tools/knowledge_query.py` |
| D3.2 document_ingest 资料导入工具 | 用户上传资料后如何触发入库 | `src/agent/tools/document_ingest.py` |
| D3.3 review_summary 考点总结工具 | 如何围绕课程资料做复习总结 | `src/agent/tools/review_summary.py` |
| D3.4 quiz_generator 与 quiz_evaluator | 如何出题、判题并回写学习状态 | `src/agent/tools/quiz_generator.py`, `src/agent/tools/quiz_evaluator.py` |
| D3.5 ToolRegistry 与工具装配 | 工具是如何注册、暴露 schema 并被 Agent 调用的 | `src/agent/tools/base.py`, `src/server/app.py` |

## D4 检索查询链路

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D4.1 QueryRouter 与 Semantic Cache | 如何决定走哪种策略，何时直接命中缓存 | `src/core/query_engine/query_router.py`, `src/core/cache/semantic_cache.py` |
| D4.2 QueryEnhancer 与查询改写 | rewrite、HyDE、多 query 在哪里接入 | `src/core/query_engine/query_enhancer.py`, `src/agent/tools/knowledge_query.py` |
| D4.3 HybridSearch 混合检索 | dense/sparse 并行、RRF 融合、排序裁剪 | `src/core/query_engine/hybrid_search.py`, `dense_retriever.py`, `sparse_retriever.py`, `fusion.py` |
| D4.4 Rerank、MMR、过滤与冲突处理 | rerank、MMR、低分过滤、冲突检测的收尾逻辑 | `src/core/query_engine/reranker.py`, `src/core/query_engine/mmr.py`, `src/core/conflict/*.py` |
| D4.5 Parent chunk、结果格式化与引用 | parent 解析、返回内容和引用展示如何收尾 | `src/agent/tools/knowledge_query.py`, `src/core/response/*.py` |

## D5 数据摄取与索引

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D5.1 Loader 与 source type 推断 | PDF/PPTX/DOCX 如何解析，题库如何识别 | `src/libs/loader/*.py`, `src/ingestion/pipeline.py` |
| D5.2 Chunking、Parent-Child 与 QuestionParser | 分块、父子块和题目解析如何配合 | `src/ingestion/chunking/document_chunker.py`, `src/ingestion/transform/question_parser.py` |
| D5.3 Transform 链路与多模态增强 | refiner、metadata、caption、dedup 的执行顺序 | `src/ingestion/transform/*.py` |
| D5.4 Dense/Sparse 编码与写库 | encoder、vector upsert、BM25、图片索引怎么协同 | `src/ingestion/embedding/*.py`, `src/ingestion/storage/*.py` |
| D5.5 DocumentManager 与跨存储一致性 | 文档列表、详情、删除如何跨多个存储协调 | `src/ingestion/document_manager.py` |

## D6 Memory、Skills 与 Hooks

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D6.1 Context filter 与 session memory | 长上下文如何压缩，会话记忆如何保存 | `src/agent/memory/context_filter.py`, `src/agent/memory/session_memory.py` |
| D6.2 StudentProfile、ErrorMemory、KnowledgeMap | 长期记忆的数据模型与更新逻辑 | `src/agent/memory/student_profile.py`, `src/agent/memory/error_memory.py`, `src/agent/memory/knowledge_map.py` |
| D6.3 Memory enhancer 与 review schedule | 记忆如何注入 prompt，复习推荐怎么触发 | `src/agent/memory/enhancer.py`, `src/agent/hooks/review_schedule.py` |
| D6.4 Skill registry、workflow 与 guardrails | 技能匹配、workflow 注入、安全防护如何协同 | `src/agent/skills/registry.py`, `src/agent/skills/workflow.py`, `src/agent/hooks/guardrails.py` |
| D6.5 课程内置 skills 与学习引导策略 | 内置学习 skills 如何被组织和用于课程场景 | `src/agent/skills/definitions/*/SKILL.md`, `src/agent/skills/registry.py` |

## D7 配置、Provider 与存储

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D7.1 settings.yaml 与 typed settings | 为什么有 YAML dict 和 typed settings 两套入口 | `config/settings.yaml`, `src/core/settings.py` |
| D7.2 Provider 工厂 | LLM、Embedding、Reranker、VectorStore 如何可插拔 | `src/agent/llm/factory.py`, `src/libs/*/*factory*.py` |
| D7.3 Chroma、BM25、SQLite、image storage | 每种存储各存什么，为什么不能混成一个 | `src/libs/vector_store/*.py`, `src/ingestion/storage/*.py`, `data/` |
| D7.4 默认课程数据与 collection 组织 | 默认课程、默认 collection 和本地数据布局 | `config/settings.yaml`, `docs/computer_internet/` |
| D7.5 Routing、Cache、Memory、Guardrails 配置开关 | 检索、缓存、记忆、安全这些能力在哪些配置项上打开 | `config/settings.yaml` |

## D8 Web 与 MCP 接口层

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D8.1 FastAPI routes、models、chat handler | HTTP 层如何接消息、上传文件、返回 SSE | `src/server/routes.py`, `src/server/models.py`, `src/server/chat_handler.py` |
| D8.2 前端 SSE 交互与页面结构 | Web 前端如何消费事件流 | `src/web/index.html`, `src/web/app.js`, `src/web/style.css` |
| D8.3 MCP server、protocol handler、tools | MCP 工具是如何注册和调度的 | `src/mcp_server/server.py`, `src/mcp_server/protocol_handler.py`, `src/mcp_server/tools/*.py` |
| D8.4 包脚本与真实运行路径的偏差 | 为什么需要分清 `agent-server` 和 `mcp-server` | `pyproject.toml`, `run_server.py`, `main.py` |
| D8.5 上传文件、静态资源与启动方式 | Web 启动后如何挂静态资源、处理上传和页面入口 | `src/server/app.py`, `src/server/routes.py`, `src/web/*` |

## D9 质量保障与可观测性

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D9.1 Trace 系统与链路观测 | ingestion/query trace 如何记录和查看 | `src/core/trace/*.py` |
| D9.2 Dashboard 页面与服务层 | 观测面板如何组织数据和页面 | `src/observability/dashboard/pages/*.py`, `src/observability/dashboard/services/*.py` |
| D9.3 Evaluation runner 与指标体系 | 评估流程、检索指标和 ragas 如何挂接 | `src/observability/evaluation/*.py`, `src/libs/evaluator/*.py` |
| D9.4 测试结构与当前环境 caveat | unit/integration 测试布局，以及 MCP 依赖缺失这类环境问题 | `tests/`, `pyproject.toml` |
| D9.5 Golden set、回归思路与质量闭环 | 为什么评估、trace、测试要形成闭环 | `tests/fixtures/`, `src/observability/evaluation/eval_runner.py` |
