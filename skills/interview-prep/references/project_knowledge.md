# 当前项目知识底稿（面试官使用）

> 以当前代码和 `DEV_SPEC.md` 为准。  
> 面试时优先考查候选人是否理解“现在这个仓库是什么”，而不是背旧版名词。

## 1. 项目当前定位

- 当前主产品形态是课程学习 Agent
- 核心用户路径是 Web 对话学习
- RAG 是底座，但上层已经叠加了 Agent、Memory、Quiz、Review、Planner、Skills

## 2. 真实入口与装配关系

- Web 主入口：`run_server.py` -> `src/server/app.py`
- `create_app()` 会装配：
  - LLM service
  - Memory stores
  - HybridSearch / QueryRouter / QueryEnhancer
  - Semantic cache
  - ToolRegistry
  - Skill registry / workflow
  - Agent
  - FastAPI routes / static files / lifespan hooks
- `main.py` 不是当前 Web 主链路入口
- MCP 真实实现位于 `src/mcp_server/`

## 3. App factory 与 runtime factory

- `src/server/app.py` 负责“总装”
- `src/storage/runtime.py` 负责“按环境切换后端”
- 这两个文件一起定义了系统是本地模式还是更接近生产模式

## 4. 生产存储现实

- conversation store：本地文件 / 生产 Postgres
- long-term memory：本地 SQLite 风格 store / 生产 Postgres
- feedback、document registry、ingestion task：生产可走 Postgres
- rate limit、semantic cache、distributed circuit breaker：生产可走 Redis
- object store：本地目录 / MinIO

## 5. Web 对话主链路

用户发起消息后的典型路径：

1. `src/server/routes.py`
2. `src/server/chat_handler.py`
3. `src/agent/agent.py`
4. conversation 读取、memory/review/skill 预处理
5. planner 判定、tool loop 编排
6. 流式输出 `text_delta / tool_start / tool_result / done`
7. 后台保存 conversation 并触发 after hooks

## 6. Planner 与 tool orchestration

- 当前 Agent 不只是“识别到关键词就调单个工具”
- `TaskPlanner` 会产出：
  - task intent
  - control mode
  - selected tool
  - 必要时的 composite subtasks
- `ToolRegistry` 统一做 schema 暴露、参数校验、超时和错误包装

## 7. Knowledge Query / RAG 主链路

`knowledge_query` 不是直接把 query 扔给向量库，它前后还有：

1. QueryRouter
2. Semantic Cache
3. query rewrite / conversation-aware rewrite
4. 可选 HyDE / multi-query
5. `HybridSearch`
6. 可选 parent chunk 与冲突检测
7. 引用与结果格式化

如果候选人只会说“用了 BM25 + 向量检索”，说明理解还停留在检索层。

## 8. HybridSearch 要点

`HybridSearch` 更像检索编排器，不是完整 RAG 的全部。

核心链路：

1. QueryProcessor
2. DenseRetriever + SparseRetriever
3. RRF
4. rerank
5. MMR
6. min_score / post-dedup

## 9. IngestionPipeline 要点

入库链路不是“切块后丢进向量库”这么简单。

关键能力包括：

- 文件完整性检查
- PDF/PPTX/DOCX loader
- source type 推断：教材、课件、题库
- QuestionParser 解析题库
- chunking / parent-child
- transform：refiner、metadata、caption、contextual enrich、dedup
- dense / sparse 双编码
- 向量写库、BM25、图片索引和文档注册

## 10. Conversation、长期记忆、上下文窗口

这三者必须区分：

- conversation：原始会话记录
- long-term memory：结构化沉淀，如 profile / error / knowledge_map / session
- context window：本次调用模型时临时拼出的输入材料

强回答应能说明：

- conversation 跨 request 如何续上
- long-term memory 为什么更适合 Postgres
- context window 为什么通常不长期持久化

## 11. Learning loop

当前项目区别于普通 RAG demo 的核心之一，是学习闭环：

- `review_summary`：按主题做复习摘要
- `quiz_generator`：题库优先，其次 RAG，上不去再 LLM fallback
- `quiz_evaluator`：评判答案并更新长期记忆
- `MemoryContextEnhancer`：把历史记忆带回当前对话
- `ReviewScheduleHook`：补复习节奏

## 12. Web 与 MCP 双接口

- Web 层更像当前主产品接口
- MCP 层是真实存在的集成接口
- MCP 通过 `ProtocolHandler` 注册和执行工具
- 默认 MCP 工具包括：
  - `query_knowledge_hub`
  - `list_collections`
  - `get_document_summary`

## 13. 稳定性机制

- retry：处理偶发错误
- rate limit：限制用户流量
- circuit breaker：处理持续故障
- 新版分布式熔断状态在 Redis 中共享，避免多实例持续轰炸上游

## 14. 可观测性与质量闭环

- trace：白盒记录链路
- dashboard：可视化查看运行状态
- evaluation runner：批量评估与回归比较
- tests：unit / integration / e2e 分层保障

## 15. 高频露馅点

- 把 `main.py` 说成当前主入口
- 把项目说成“单纯的模块化 MCP Server”
- 把聊天记录、长期记忆、上下文窗口讲混
- 说做了生产化，但说不清 Postgres / Redis / object store 的边界
- 说做了 HybridSearch，却答不出 `knowledge_query` 前后编排
