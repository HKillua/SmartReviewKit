# 项目复习题库（按当前代码组织）

> 当前版本基于课程学习 Agent 的真实代码主链路。  
> 共 8 章 32 题，每章 4 题。  
> 难度说明：`⭐` 基础识别，`⭐⭐` 链路理解，`⭐⭐⭐` 设计取舍与源码细节。

## 章节总览

| 章节 | 主题 | 题数 |
|------|------|------|
| 第 1 章 | 项目定位与入口 | 4 |
| 第 2 章 | Agent 主链路与流式输出 | 4 |
| 第 3 章 | 学习工具与课程闭环 | 4 |
| 第 4 章 | Knowledge Query 与 HybridSearch | 4 |
| 第 5 章 | Ingestion Pipeline 与索引 | 4 |
| 第 6 章 | Memory、Skills 与 Hooks | 4 |
| 第 7 章 | Web 与 MCP 接口层 | 4 |
| 第 8 章 | 配置、存储、观测与测试 | 4 |

---

## 第 1 章：项目定位与入口

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R1-01 | 现在最准确地描述这个项目，应该怎么说？ | ⭐ | 项目定位 | 当前主体是课程学习 Agent 平台，RAG 是底座，MCP 仍存在但不再是唯一主叙事 | `DEV_SPEC.md`, `src/server/app.py` |
| R1-02 | 为什么说 `run_server.py` 才是当前 Web 主入口？ | ⭐⭐ | 真实入口 | `run_server.py` 读取配置后启动 `src.server.app:create_app`；这是当前可用的 Web Agent 启动路径 | `run_server.py`, `src/server/app.py` |
| R1-03 | `main.py` 和 `pyproject.toml` 里的 `mcp-server` 脚本为什么不能代表当前主链路？ | ⭐⭐ | 历史残留识别 | `main.py` 仍是早期占位入口，而 `pyproject.toml` 里的脚本仍指向它；真实 MCP 在 `src/mcp_server/` | `main.py`, `pyproject.toml`, `src/mcp_server/server.py` |
| R1-04 | 默认课程、默认 collection 和启动自动 ingest 分别在哪里配置或触发？ | ⭐⭐⭐ | 启动装配 | 默认 collection / auto_ingest_dir 在 `config/settings.yaml`；启动时由 `src/server/app.py` 装配并触发自动 ingest | `config/settings.yaml`, `src/server/app.py` |

---

## 第 2 章：Agent 主链路与流式输出

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R2-01 | `Agent.chat()` 的整体流程是什么？ | ⭐⭐ | 主链路 | 读取/创建会话 -> before hooks -> 拼 system prompt -> 注入 memory/skill -> tool loop -> 输出 done -> 后台保存会话与 after hooks | `src/agent/agent.py` |
| R2-02 | system prompt 在这个项目里是怎么组装的？memory 和 skill 是在哪一步注入的？ | ⭐⭐ | prompt 组装 | `prompt_builder.build()` 接收 tool schemas、memory context、active skill；memory 和 skill 在进入 tool loop 前通过 gather 预取并注入 | `src/agent/agent.py`, `src/agent/prompt_builder.py` |
| R2-03 | 为什么这个项目适合用 SSE 做流式输出？都流哪些事件？ | ⭐⭐ | 流式事件 | HTTP SSE 足够支撑单向推流；会输出 `text_delta`、`tool_start`、`tool_result`、`error`、`done` 等事件 | `src/agent/types.py`, `src/server/chat_handler.py` |
| R2-04 | hooks 和 middleware 在这里分别解决什么问题？ | ⭐⭐⭐ | 扩展点边界 | hooks 更偏消息生命周期；middleware 更偏 LLM 请求前后拦截，例如重试、反思、保护 | `src/agent/hooks/*.py` |

---

## 第 3 章：学习工具与课程闭环

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R3-01 | `knowledge_query` 在课程学习场景里承担什么角色？ | ⭐ | 工具定位 | 负责课程知识问答，是最核心的检索型工具入口 | `src/agent/tools/knowledge_query.py` |
| R3-02 | `review_summary` 和普通问答有什么不同？ | ⭐⭐ | 工具差异 | 它不是逐句问答，而是按主题/章节把检索内容组织为结构化复习摘要，并可标记薄弱点 | `src/agent/tools/review_summary.py` |
| R3-03 | `quiz_generator` 为什么分成“题库优先 -> RAG 生成 -> LLM fallback”三层？ | ⭐⭐⭐ | 生成策略 | 先用已有题库保证真实性，再退到知识库上下文生成，最后才是纯 LLM；兼顾质量、覆盖和兜底 | `src/agent/tools/quiz_generator.py` |
| R3-04 | `quiz_evaluator` 评完一道题后，可能更新哪些学习记忆？ | ⭐⭐⭐ | 闭环更新 | 错题会写入 `ErrorMemory`；涉及知识点会更新 `KnowledgeMapMemory` 的掌握度 | `src/agent/tools/quiz_evaluator.py` |

---

## 第 4 章：Knowledge Query 与 HybridSearch

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R4-01 | 在这个项目里，为什么不能把 `knowledge_query` 简化成“直接查向量库”？ | ⭐⭐ | 工具编排 | 它前后还有 QueryRouter、Semantic Cache、query rewrite、HyDE/multi-query、parent chunk 与冲突检测等编排 | `src/agent/tools/knowledge_query.py` |
| R4-02 | QueryRouter 和 Semantic Cache 分别解决什么问题？ | ⭐⭐ | 路由与缓存 | 路由判断是否需要 RAG、偏好来源；缓存避免高相似问题重复检索 | `src/core/query_engine/query_router.py`, `src/core/cache/semantic_cache.py` |
| R4-03 | `HybridSearch` 的核心步骤有哪些？ | ⭐⭐ | 混合检索链路 | QueryProcessor -> Dense/Sparse 并行 -> RRF 融合 -> rerank -> MMR -> min_score / post-dedup | `src/core/query_engine/hybrid_search.py` |
| R4-04 | rerank、MMR、min_score、post-dedup 各自解决什么问题？ | ⭐⭐⭐ | 后处理取舍 | rerank 提高排序精度；MMR 控制多样性；min_score 去掉低质量结果；post-dedup 避免语义重复 | `src/core/query_engine/hybrid_search.py`, `src/core/query_engine/mmr.py` |

---

## 第 5 章：Ingestion Pipeline 与索引

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R5-01 | `IngestionPipeline` 的阶段顺序是什么？ | ⭐ | 入库主流程 | 完整性检查 -> load -> split -> transform -> embed -> upsert / persist | `src/ingestion/pipeline.py` |
| R5-02 | 为什么这个项目要特别区分课件、教材和题库这几类 source type？ | ⭐⭐ | 课程导向 | 学习场景需要不同的解析与下游使用方式，题库还会走 `QuestionParser` | `src/ingestion/pipeline.py`, `src/ingestion/transform/question_parser.py` |
| R5-03 | transform 阶段里为什么会同时存在 refiner、metadata、contextual enrich、caption、dedup？ | ⭐⭐⭐ | 多阶段增强 | 它们分别补齐 chunk 质量、语义元数据、上下文信息、图片可检索性和重复控制 | `src/ingestion/transform/*.py` |
| R5-04 | 为什么这个项目要同时写 Chroma、BM25 和图片相关存储？ | ⭐⭐⭐ | 多存储协同 | Dense 检索、Sparse 检索和图片返回不是一类数据；需要协同存储支持完整检索体验 | `src/ingestion/storage/*.py`, `src/libs/vector_store/*.py` |

---

## 第 6 章：Memory、Skills 与 Hooks

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R6-01 | 这个项目的 memory 大致分几类？各自存什么？ | ⭐⭐ | memory 分类 | student profile、error memory、knowledge map、skill memory、session memory 各自负责偏好、错题、掌握度、技能命中、会话记忆 | `src/agent/memory/*.py` |
| R6-02 | `MemoryContextEnhancer` 和 `MemoryRecordHook` 的职责为什么不同？ | ⭐⭐⭐ | 读写分离 | enhancer 负责把已有记忆注入上下文；record hook 负责从对话中抽取并写回记忆 | `src/agent/memory/enhancer.py` |
| R6-03 | `review_schedule` 和 `context_filter` 在学习产品里分别有什么价值？ | ⭐⭐⭐ | 学习体验 | review schedule 负责主动复习提示；context filter 负责长上下文压缩，避免对话越聊越失控 | `src/agent/hooks/review_schedule.py`, `src/agent/memory/context_filter.py` |
| R6-04 | skill workflow 和 guardrails 在这个项目里分别扮演什么角色？ | ⭐⭐ | 技能与安全 | workflow 负责识别/注入技能型流程；guardrails 负责高风险输入输出拦截与脱敏 | `src/agent/skills/workflow.py`, `src/agent/hooks/guardrails.py` |

---

## 第 7 章：Web 与 MCP 接口层

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R7-01 | `routes.py`、`models.py`、`chat_handler.py` 在 Web 层是怎么分工的？ | ⭐⭐ | Web 分层 | routes 暴露 API，models 定义请求响应模型，chat_handler 串接 Agent 并产出 SSE | `src/server/routes.py`, `src/server/models.py`, `src/server/chat_handler.py` |
| R7-02 | 前端为什么适合通过 SSE 接收回答？ | ⭐⭐ | 前端交互 | 当前主要是服务端单向事件流，不需要先引入更复杂的双向协议；实现简单且与 Agent streaming 匹配 | `src/web/app.js`, `src/web/index.html` |
| R7-03 | MCP 这一层的真实实现在哪里？默认注册了哪些工具？ | ⭐⭐ | MCP 实现 | `src/mcp_server/server.py` 和 `protocol_handler.py` 里创建 server 并注册 `query_knowledge_hub`、`list_collections`、`get_document_summary` | `src/mcp_server/*.py` |
| R7-04 | 为什么说“Web 主路径更完整，但 MCP 仍然是重要接口”？ | ⭐⭐⭐ | 双接口定位 | Web 更贴近当前学习产品体验；MCP 适合对外部 Agent 暴露标准工具接口，两者定位不同 | `src/server/app.py`, `src/mcp_server/server.py` |

---

## 第 8 章：配置、存储、观测与测试

| # | 题目 | 难度 | 考察要点 | 参考答案要点 | 关键文件 |
|---|------|------|---------|--------------|----------|
| R8-01 | 为什么这个仓库会同时出现 typed settings 和原始 YAML dict 两种配置读取方式？ | ⭐⭐⭐ | 配置双轨 | 基础设施大量走 `src/core/settings.py` typed settings；Agent / Web 装配不少地方直接读 YAML dict，这也是当前需要理解的现实 | `config/settings.yaml`, `src/core/settings.py`, `src/server/app.py` |
| R8-02 | LLM、Embedding、VectorStore、Reranker 是怎么做到可插拔的？ | ⭐⭐ | 工厂模式 | 通过 Base 抽象 + Factory + 配置切换 provider | `src/agent/llm/factory.py`, `src/libs/*/*factory*.py` |
| R8-03 | trace、dashboard、evaluation 在这个项目里分别解决什么问题？ | ⭐⭐ | 可观测性 | trace 记录链路，dashboard 提供可视化查看，evaluation 用于质量评估和回归比较 | `src/core/trace/*.py`, `src/observability/dashboard/*`, `src/observability/evaluation/*` |
| R8-04 | 测试和当前环境里最值得注意的一个 caveat 是什么？ | ⭐⭐⭐ | 工程现实 | 测试体系覆盖 unit/integration；当前环境若缺 `mcp` 依赖，会让 MCP 相关 smoke import 失败，这属于环境问题不是主链路设计问题 | `tests/`, `pyproject.toml` |

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
| 合计 | 32 |
