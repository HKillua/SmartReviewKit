# 项目复习题库（按当前代码组织）

> 当前版本基于课程学习 Agent 的真实代码主链路。  
> 共 9 章 36 题，每章 4 题。  
> 难度说明：`⭐` 基础识别，`⭐⭐` 链路理解，`⭐⭐⭐` 设计取舍与工程现实。

## 章节总览

| 章节 | 主题 | 题数 |
|------|------|------|
| 第 1 章 | 项目定位与入口 | 4 |
| 第 2 章 | 应用装配、配置与生产存储 | 4 |
| 第 3 章 | Agent 运行时与流式交互 | 4 |
| 第 4 章 | Planner、ToolRegistry 与学习工具 | 4 |
| 第 5 章 | Knowledge Query 与 HybridSearch | 4 |
| 第 6 章 | Ingestion Pipeline 与文档生命周期 | 4 |
| 第 7 章 | Conversation、Memory 与上下文工程 | 4 |
| 第 8 章 | Web、前端与 MCP 接口 | 4 |
| 第 9 章 | 观测、评估、测试与工程化 | 4 |

---

## 第 1 章：项目定位与入口

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R1-01 | 现在最准确地描述这个项目，应该怎么说？ | ⭐ | 项目定位 | 当前主产品是课程学习 Agent，RAG 是底座，MCP 是并存接口但不是唯一主叙事 | `DEV_SPEC.md`, `src/server/app.py` |
| R1-02 | 为什么说 `run_server.py` 才是当前 Web 主入口？ | ⭐⭐ | 真实入口 | 它读取配置并启动 `src.server.app:create_app`；这是当前用户侧最完整的运行路径 | `run_server.py`, `src/server/app.py` |
| R1-03 | `main.py` 在当前项目里处于什么位置？ | ⭐⭐ | 历史残留识别 | `main.py` 仍是历史脚本入口；真实 MCP 逻辑在 `src/mcp_server/` | `main.py`, `src/mcp_server/server.py`, `pyproject.toml` |
| R1-04 | 如果让你用“一条主链路”概括全项目，你会怎么画？ | ⭐⭐⭐ | 全局视角 | Web/Route -> Agent -> Planner/Tools -> Retrieval/Ingestion/Memory -> Streaming + Persistence | `src/server/routes.py`, `src/agent/agent.py`, `src/storage/runtime.py` |

---

## 第 2 章：应用装配、配置与生产存储

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R2-01 | `create_app()` 启动时主要装了哪些组件？ | ⭐⭐ | 应用工厂 | LLM、memory stores、HybridSearch、router/cache、tools、skills、Agent、routes、lifespan | `src/server/app.py` |
| R2-02 | `src/storage/runtime.py` 在整个项目里为什么重要？ | ⭐⭐ | 运行时切换 | 它统一决定 conversation、memory、feedback、rate limit、circuit breaker、semantic cache 等后端选型 | `src/storage/runtime.py` |
| R2-03 | 生产环境下聊天记录、长期记忆和共享状态分别存哪？ | ⭐⭐⭐ | 存储边界 | conversation / long-term memory / feedback / registry 走 Postgres；限流 / 分布式熔断 / 语义缓存可走 Redis；对象存储可走 MinIO | `src/storage/postgres_backends.py`, `src/storage/runtime.py`, `config/settings.storage_stack.yaml` |
| R2-04 | 为什么长期记忆不直接放 Redis 或向量数据库？ | ⭐⭐⭐ | 设计取舍 | 记忆主要是结构化状态和精确更新，适合 Postgres；Redis 更适合共享运行时状态，向量库更适合知识检索 | `src/agent/memory/`, `src/storage/postgres_backends.py` |

---

## 第 3 章：Agent 运行时与流式交互

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R3-01 | `Agent.chat()` 的整体流程是什么？ | ⭐⭐ | 主链路 | 读取/创建会话 -> before hooks -> 预取 memory/review/skill -> prompt 组装 -> tool loop -> done -> 后台保存与 after hooks | `src/agent/agent.py` |
| R3-02 | 为什么 memory、review、skill 要在进入 tool loop 之前并行获取？ | ⭐⭐ | 预处理设计 | 降低等待时间，并在首轮 LLM 调用前一次性补齐系统上下文 | `src/agent/agent.py`, `src/agent/memory/enhancer.py` |
| R3-03 | 当前流式输出链路是怎么接起来的？ | ⭐⭐ | 流式交互 | LLM 流 -> Agent StreamEvent -> ChatHandler chunk -> SSE 推给前端 | `src/agent/agent.py`, `src/server/chat_handler.py`, `src/server/routes.py` |
| R3-04 | `_post_message_tasks()` 和 `flush()` 分别解决什么问题？ | ⭐⭐⭐ | 后台保存 | 一个负责后台保存 conversation 和 after hooks，一个负责优雅退出时等待后台任务完成 | `src/agent/agent.py` |

---

## 第 4 章：Planner、ToolRegistry 与学习工具

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R4-01 | TaskPlanner 在这个项目里负责什么？ | ⭐⭐ | planner 定位 | 识别 task intent、决定 control mode、必要时拆复合任务，不是简单 keyword router | `src/agent/planner/task_planner.py` |
| R4-02 | ToolRegistry 为什么要统一管理工具 schema 和执行？ | ⭐⭐ | tool calling 基础设施 | 统一注册、参数校验、schema 暴露、超时与错误包装，给 Agent function-calling 使用 | `src/agent/tools/base.py` |
| R4-03 | 当前学习工具链包括哪些核心工具？ | ⭐ | 工具全景 | `knowledge_query`、`document_ingest`、`review_summary`、`quiz_generator`、`quiz_evaluator` | `src/server/app.py`, `src/agent/tools/` |
| R4-04 | 为什么说 quiz/review/knowledge_query 共同构成了学习闭环？ | ⭐⭐⭐ | 产品闭环 | 问答获取知识、复习总结考点、出题练习、判题回写 memory，形成持续学习路径 | `src/agent/tools/`, `src/agent/memory/enhancer.py` |

---

## 第 5 章：Knowledge Query 与 HybridSearch

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R5-01 | 为什么 `knowledge_query` 不能简化成“直接查向量库”？ | ⭐⭐ | 工具编排层 | 它前后还有 QueryRouter、Semantic Cache、query rewrite、HyDE、多 query、parent chunk 和冲突检测 | `src/agent/tools/knowledge_query.py` |
| R5-02 | QueryRouter 和 Semantic Cache 分别在解决什么问题？ | ⭐⭐ | 路由与缓存 | Router 决定是否需要 RAG 和偏好来源；cache 避免高相似查询重复检索 | `src/core/query_engine/query_router.py`, `src/core/cache/redis_semantic_cache.py` |
| R5-03 | `HybridSearch` 的主链路有哪些步骤？ | ⭐⭐ | 混合检索 | QueryProcessor -> dense/sparse -> RRF -> rerank -> MMR -> min_score / post-dedup | `src/core/query_engine/hybrid_search.py` |
| R5-04 | rerank、MMR、min_score、冲突检测为什么不能混成一个步骤？ | ⭐⭐⭐ | 后处理边界 | 它们分别处理排序精度、多样性、低质结果过滤和知识冲突，不是同一类问题 | `src/core/query_engine/`, `src/core/conflict/` |

---

## 第 6 章：Ingestion Pipeline 与文档生命周期

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R6-01 | `IngestionPipeline` 的阶段顺序是什么？ | ⭐ | 入库主流程 | 完整性检查 -> loader -> chunking -> transform -> encoding -> storage | `src/ingestion/pipeline.py` |
| R6-02 | 为什么课程场景要特别区分教材、课件和题库？ | ⭐⭐ | source type 设计 | 三类资料在解析方式、检索用途和下游生成场景上不同，题库还会走 QuestionParser | `src/ingestion/pipeline.py`, `src/ingestion/transform/question_parser.py` |
| R6-03 | transform 阶段为什么不只是 metadata enrichment？ | ⭐⭐⭐ | 多阶段增强 | 还包含 refiner、contextual enrich、image caption、dedup，分别提升可读性、上下文、图文可检索性和去重 | `src/ingestion/transform/` |
| R6-04 | `DocumentManager.delete_document()` 为什么是一个跨存储问题？ | ⭐⭐⭐ | 生命周期治理 | 删除文档要协调向量、BM25、图片索引、对象存储引用和完整性记录，不是只删一处 | `src/ingestion/document_manager.py`, `src/storage/runtime.py` |

---

## 第 7 章：Conversation、Memory 与上下文工程

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R7-01 | 聊天记录、长期记忆、上下文窗口分别是什么？ | ⭐⭐ | 概念边界 | conversation 是原始会话，long-term memory 是结构化沉淀，上下文窗口是本次送进模型的临时材料 | `src/agent/conversation.py`, `src/agent/memory/enhancer.py`, `src/agent/agent.py` |
| R7-02 | 多次 request 之间，系统是怎么延续上下文的？ | ⭐⭐ | 跨请求连续性 | 上一轮先持久化 conversation / memory，下一轮根据 `conversation_id` 和 `user_id` 重新读回来 | `src/agent/agent.py`, `src/storage/postgres_backends.py` |
| R7-03 | `MemoryContextEnhancer` 和 `MemoryRecordHook` 为什么要分开？ | ⭐⭐⭐ | 读写分离 | 一个负责读取已有记忆并注入 prompt，一个负责从对话中抽取并写回长期记忆 | `src/agent/memory/enhancer.py` |
| R7-04 | `ContextEngineeringFilter` 的 4 级压缩各自解决什么问题？ | ⭐⭐⭐ | 长上下文治理 | 最近消息滑窗、超长工具结果卸载、旧历史摘要压缩、token budget 控制 | `src/agent/memory/context_filter.py` |

---

## 第 8 章：Web、前端与 MCP 接口

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R8-01 | `routes.py`、`models.py`、`chat_handler.py` 在 Web 层是怎么分工的？ | ⭐⭐ | Web 分层 | routes 暴露 HTTP/SSE 接口，models 约束输入输出，chat_handler 把 Agent 事件流转成前端 chunk | `src/server/` |
| R8-02 | 为什么当前用 SSE 就够了？ | ⭐⭐ | 协议选择 | 当前主要是服务端单向推流，SSE 已能满足文本增量和工具事件展示 | `src/server/routes.py`, `src/web/app.js` |
| R8-03 | MCP 这一层的真实实现在哪里？ | ⭐⭐ | MCP 事实边界 | `src/mcp_server/server.py` 和 `protocol_handler.py`；默认注册查询知识库、列集合、文档摘要工具 | `src/mcp_server/` |
| R8-04 | 为什么要区分“真实 MCP 实现”和“历史脚本入口”？ | ⭐⭐⭐ | 历史债识别 | 真实协议逻辑和当前包脚本不是同一层；不区分会把入口和实现混淆 | `main.py`, `src/mcp_server/server.py`, `pyproject.toml` |

---

## 第 9 章：观测、评估、测试与工程化

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R9-01 | trace、dashboard、evaluation runner 在这个项目里分别做什么？ | ⭐⭐ | 可观测性全景 | trace 记录链路，dashboard 做可视化查看，evaluation runner 跑数据集评估与回归 | `src/core/trace/`, `src/observability/` |
| R9-02 | retry、rate limit、circuit breaker 分别解决什么问题？ | ⭐⭐⭐ | 稳定性机制 | retry 处理偶发错误，rate limit 控制用户流量，circuit breaker 避免持续轰炸上游 | `src/agent/hooks/` |
| R9-03 | 为什么分布式熔断状态要放进 Redis，而不是只放进单机内存？ | ⭐⭐⭐ | 多实例生产化 | 多实例下需要共享 breaker 状态，否则某个 Pod 熔断后其他 Pod 仍会继续打上游 | `src/agent/hooks/redis_circuit_breaker.py`, `src/storage/runtime.py` |
| R9-04 | 当前测试结构能说明什么？还要特别提醒什么 caveat？ | ⭐⭐ | 工程现实 | 有 unit/integration/e2e 分层；环境依赖缺失会影响部分链路，但不能把环境问题等同于主架构问题 | `tests/`, `pyproject.toml` |

---

## 汇总统计

| 章节 | 题数 |
|------|------|
| 第 1 章 | 4 |
| 第 2 章 | 4 |
| 第 3 章 | 4 |
| 第 4 章 | 4 |
| 第 5 章 | 4 |
| 第 6 章 | 4 |
| 第 7 章 | 4 |
| 第 8 章 | 4 |
| 第 9 章 | 4 |
| 合计 | 36 |
