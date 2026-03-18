# 项目亮点库（以当前代码为准）

> 用于挑选简历 bullet。每个亮点都尽量绑定真实模块、真实链路和真实边界。

## 1. 课程学习 Agent 主产品

- 安全主张：
  当前仓库的主产品形态是课程学习 Agent，RAG 是底座而不是全部。
- 关键文件：
  `DEV_SPEC.md`, `run_server.py`, `src/server/app.py`
- 适合岗位：
  LLM 应用、Agent、教育 AI、后端平台

## 2. App Factory 与组件装配

- 安全主张：
  `create_app()` 在启动时统一装配 LLM、检索、记忆、工具、技能、存储、CORS、路由和静态页面。
- 关键文件：
  `src/server/app.py`, `src/storage/runtime.py`
- 适合岗位：
  后端、平台、架构

## 3. Agent Runtime 与流式交互

- 安全主张：
  `Agent.chat()` 负责会话读取、prompt 预处理、memory/review/skill 注入、tool loop、SSE 事件流和后台保存。
- 关键文件：
  `src/agent/agent.py`, `src/server/chat_handler.py`, `src/server/routes.py`
- 适合岗位：
  Agent、后端、平台工程

## 4. Task Planner 与复合任务拆解

- 安全主张：
  Agent 不是只靠规则触发单工具，还带 task planner、control mode 和 composite subtask 拆解。
- 关键文件：
  `src/agent/planner/task_planner.py`, `src/agent/agent.py`
- 适合岗位：
  Agent、智能编排、工作流

## 5. Knowledge Query 工具编排

- 安全主张：
  `knowledge_query` 把 QueryRouter、Semantic Cache、查询增强、HybridSearch、parent chunk 和冲突检测串成统一工具。
- 关键文件：
  `src/agent/tools/knowledge_query.py`, `src/core/query_engine/query_router.py`, `src/core/conflict/`
- 适合岗位：
  RAG、检索、搜索工程

## 6. 混合检索主引擎

- 安全主张：
  检索链路包含 QueryProcessor、dense/sparse 召回、RRF、rerank、MMR、低分过滤与 post-dedup。
- 关键文件：
  `src/core/query_engine/hybrid_search.py`, `src/core/query_engine/`
- 适合岗位：
  RAG、搜索、检索平台

## 7. 多阶段资料入库流水线

- 安全主张：
  `IngestionPipeline` 串联完整性检查、loader、chunking、transform、编码、写库，支持 PDF/PPTX/DOCX 和题库场景。
- 关键文件：
  `src/ingestion/pipeline.py`, `src/libs/loader/`, `src/ingestion/transform/`
- 适合岗位：
  RAG 基建、数据平台、知识工程

## 8. 文档生命周期与跨存储一致性

- 安全主张：
  `DocumentManager` 负责 list/detail/delete 等文档生命周期操作，并协调向量、BM25、图片索引和完整性记录。
- 关键文件：
  `src/ingestion/document_manager.py`, `src/storage/runtime.py`
- 适合岗位：
  后端、平台、数据治理

## 9. 结构化长期记忆与学习闭环

- 安全主张：
  长期记忆不是 Markdown 文件，也不是单纯向量记忆；它按 profile / error / knowledge_map / skill / session 结构化存储。
- 关键文件：
  `src/agent/memory/enhancer.py`, `src/agent/memory/`, `src/storage/postgres_backends.py`
- 适合岗位：
  Agent、教育 AI、LLM 产品

## 10. 生产共享存储与运行时稳定性

- 安全主张：
  生产模式支持 Postgres 持久化、Redis 限流/语义缓存/分布式熔断、对象存储和任务队列化入库。
- 关键文件：
  `config/settings.storage_stack.yaml`, `src/storage/runtime.py`, `src/agent/hooks/redis_rate_limit.py`, `src/agent/hooks/redis_circuit_breaker.py`, `src/storage/postgres_backends.py`
- 适合岗位：
  后端、平台、生产工程

## 11. Web Agent 与 MCP 双接口

- 安全主张：
  仓库同时提供 FastAPI Web Agent 和 MCP Server；但当前 Web 产品链路更完整，MCP 更像标准工具接口。
- 关键文件：
  `src/server/`, `src/mcp_server/`, `main.py`
- 适合岗位：
  平台、Agent 集成、工具链

## 12. 可观测性、评估与测试闭环

- 安全主张：
  仓库包含 trace、dashboard、evaluation runner、golden test set 和较完整的 unit/integration/e2e 测试布局。
- 关键文件：
  `src/core/trace/`, `src/observability/dashboard/`, `src/observability/evaluation/`, `tests/`
- 适合岗位：
  平台工程、RAG、后端质量体系

## 选 bullet 时的证据规则

- 想写生产存储，先检查 `config/settings.storage_stack.yaml` 和 `src/storage/runtime.py`
- 想写记忆，先区分 conversation、long-term memory、context window
- 想写 resilience，先区分 retry、rate limit、circuit breaker 各自边界
- 想写 MCP，必须说明真实实现与历史入口不是同一层
