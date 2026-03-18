# 项目学习地图（按当前代码组织）

> 用于 `project-learner` 选择学习域和阅读路径。  
> 当前版本按 9 个知识域、45 个知识点组织。

## 推荐阅读顺序

1. D1 项目定位与入口
2. D2 App factory、配置与生产存储
3. D3 Agent 运行时与流式链路
4. D4 Planner、ToolRegistry 与学习工具
5. D5 检索查询链路
6. D6 数据摄取与文档生命周期
7. D7 Conversation、Memory 与上下文工程
8. D8 Web、前端与 MCP 接口
9. D9 可观测性、评估与测试

## D1 项目定位与入口

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D1.1 当前产品定位 | 理解为什么它现在更像课程学习 Agent，而不是单纯 MCP/RAG demo | `DEV_SPEC.md`, `src/server/app.py` |
| D1.2 Web 主入口 | 看 `run_server.py` 如何启动 `create_app()` | `run_server.py`, `src/server/app.py` |
| D1.3 MCP 实现与历史入口 | 区分真实 MCP 实现和 `main.py` 历史脚本 | `src/mcp_server/server.py`, `src/mcp_server/protocol_handler.py`, `main.py`, `pyproject.toml` |
| D1.4 主链路脑图 | 用一条链把 Web、Agent、Tools、Retrieval、Memory、Persistence 串起来 | `src/server/routes.py`, `src/agent/agent.py`, `src/storage/runtime.py` |
| D1.5 当前目录现实 | 看 `config/`、`data/`、`docs/`、`src/` 分别承载什么 | `config/`, `data/`, `docs/`, `src/` |

## D2 App factory、配置与生产存储

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D2.1 create_app 装配 | 启动时如何把 LLM、检索、memory、tools、skills 接起来 | `src/server/app.py` |
| D2.2 typed settings 与原始 YAML | 为什么这个项目同时存在 typed settings 和 raw dict | `src/core/settings.py`, `config/settings.yaml` |
| D2.3 runtime factories | conversation、memory、feedback、cache、breaker 如何按环境切换 | `src/storage/runtime.py` |
| D2.4 Postgres 持久化 | 生产环境下 conversation / memory / feedback / registry / task store 怎么落库 | `src/storage/postgres_backends.py` |
| D2.5 Redis 共享状态 | Redis 在 rate limit、semantic cache、distributed circuit breaker 里的角色 | `src/agent/hooks/redis_rate_limit.py`, `src/agent/hooks/redis_circuit_breaker.py`, `src/core/cache/redis_semantic_cache.py` |

## D3 Agent 运行时与流式链路

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D3.1 会话加载与消息追加 | conversation 如何读取、新消息何时进入会话对象 | `src/agent/agent.py`, `src/agent/conversation.py` |
| D3.2 prompt 预处理 | memory/review/skill 为什么要在 tool loop 前预取 | `src/agent/agent.py`, `src/agent/prompt_builder.py` |
| D3.3 ReAct tool loop | Agent 如何多轮调工具、继续生成、结束回答 | `src/agent/agent.py` |
| D3.4 流式事件链路 | LLM 流、Agent 事件流、SSE 推流怎么接起来 | `src/agent/types.py`, `src/server/chat_handler.py`, `src/server/routes.py` |
| D3.5 后台保存与 flush | 为什么保存 conversation / after hooks 放到后台做 | `src/agent/agent.py` |

## D4 Planner、ToolRegistry 与学习工具

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D4.1 TaskPlanner | task intent、control mode、composite subtask 是怎么判定的 | `src/agent/planner/task_planner.py` |
| D4.2 ToolRegistry | 工具 schema、参数校验、超时与错误包装如何统一 | `src/agent/tools/base.py` |
| D4.3 knowledge_query / review_summary | 问答与复习摘要工具分别解决什么问题 | `src/agent/tools/knowledge_query.py`, `src/agent/tools/review_summary.py` |
| D4.4 quiz_generator / quiz_evaluator | 出题、判题、记忆回写怎么形成学习闭环 | `src/agent/tools/quiz_generator.py`, `src/agent/tools/quiz_evaluator.py` |
| D4.5 document_ingest | 用户上传文件后如何进入异步/同步入库链路 | `src/agent/tools/document_ingest.py`, `src/server/routes.py` |

## D5 检索查询链路

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D5.1 QueryRouter | 为什么要先判断是否需要 RAG、偏好什么来源 | `src/core/query_engine/query_router.py` |
| D5.2 Semantic Cache 与查询增强 | 查询缓存、rewrite、HyDE、多 query 的组合位置 | `src/core/cache/redis_semantic_cache.py`, `src/core/query_engine/query_enhancer.py`, `src/agent/tools/knowledge_query.py` |
| D5.3 HybridSearch | dense/sparse 并行、RRF、rerank、MMR 的主链路 | `src/core/query_engine/hybrid_search.py`, `src/core/query_engine/` |
| D5.4 冲突检测 | 知识冲突为什么作为检索后处理存在 | `src/core/conflict/`, `src/agent/tools/knowledge_query.py` |
| D5.5 引用与结果格式化 | parent chunk、citation、result formatting 怎么收尾 | `src/core/response/`, `src/agent/tools/knowledge_query.py` |

## D6 数据摄取与文档生命周期

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D6.1 Loader 与 source type | PDF/PPTX/DOCX 如何解析，题库如何识别 | `src/libs/loader/`, `src/ingestion/pipeline.py` |
| D6.2 Chunking 与 QuestionParser | parent-child、题库抽题和 chunk 结构怎么配合 | `src/ingestion/chunking/document_chunker.py`, `src/ingestion/transform/question_parser.py` |
| D6.3 Transform 链路 | refiner、metadata、caption、dedup、contextual enrich 的顺序和作用 | `src/ingestion/transform/` |
| D6.4 编码与写库 | dense/sparse 编码、向量写库、BM25 和图片索引怎么协同 | `src/ingestion/embedding/`, `src/ingestion/storage/` |
| D6.5 DocumentManager | 已入库文档的 list/detail/delete 为什么是跨存储问题 | `src/ingestion/document_manager.py` |

## D7 Conversation、Memory 与上下文工程

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D7.1 ConversationStore | 聊天记录本地和生产环境分别怎么存 | `src/agent/conversation.py`, `src/storage/postgres_backends.py` |
| D7.2 长期记忆结构 | profile、error、knowledge_map、skill、session 各存什么 | `src/agent/memory/`, `src/storage/postgres_backends.py` |
| D7.3 Memory enhancer / record hook | 记忆什么时候读，什么时候写 | `src/agent/memory/enhancer.py` |
| D7.4 ContextEngineeringFilter | 长上下文压缩、工具结果卸载、token budget 为什么重要 | `src/agent/memory/context_filter.py` |
| D7.5 三个概念的边界 | 聊天记录、长期记忆、上下文窗口为什么不能混为一谈 | `src/agent/agent.py`, `src/agent/memory/enhancer.py` |

## D8 Web、前端与 MCP 接口

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D8.1 FastAPI 分层 | routes、models、chat_handler 的职责边界 | `src/server/` |
| D8.2 前端 SSE 页面 | 前端如何消费消息流、显示会话和上传结果 | `src/web/index.html`, `src/web/app.js`, `src/web/style.css` |
| D8.3 Conversation API | 历史会话列表、详情、删除接口如何暴露 | `src/server/routes.py` |
| D8.4 MCP server | server、protocol handler、默认工具如何接起来 | `src/mcp_server/` |
| D8.5 Web 与 MCP 定位差异 | 为什么 Web 更像当前产品接口，MCP 更像集成接口 | `src/server/app.py`, `src/mcp_server/server.py` |

## D9 可观测性、评估与测试

| 子主题 | 学什么 | 关键文件 |
|--------|--------|----------|
| D9.1 Trace 系统 | query / ingestion / agent / memory trace 如何记录 | `src/core/trace/` |
| D9.2 Dashboard | observability dashboard 的页面和 services 如何组织 | `src/observability/dashboard/*` |
| D9.3 Evaluation runner | golden set、eval runner、ragas evaluator 如何配合 | `src/observability/evaluation/`, `tests/fixtures/` |
| D9.4 测试分层 | unit / integration / e2e 各在验证什么 | `tests/` |
| D9.5 稳定性机制 | retry、rate limit、distributed circuit breaker 的工程价值 | `src/agent/hooks/`, `src/storage/runtime.py` |
