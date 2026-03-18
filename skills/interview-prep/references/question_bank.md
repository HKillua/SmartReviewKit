# 面试题库（扩展版）

> 目标：既覆盖当前项目的广度，也能扛住深挖。  
> 建议顺序：先用 A 组校准定位，再用 B 组做简历辩护，最后从 C 组挑 3-4 个主题深挖。

## A. 开场题池：项目定位与主链路

1. 现在最准确地描述这个仓库，应该怎么说？
2. 为什么它已经不是单纯的 MCP/RAG demo？
3. `run_server.py` 和 `src/server/app.py` 在当前项目里分别负责什么？
4. 从网页里发一条消息，到最终拿到流式回答，中间会经过哪些模块？
5. 你怎么向面试官解释 Agent 层和 RAG 层的边界？
6. 为什么 `main.py` 不能代表当前主运行链路？
7. 当前 Web 接口和 MCP 接口分别服务什么场景？
8. 你觉得这个仓库最像“产品”的地方是哪一段链路？

## B. 简历辩护题池

### B1. 项目包装与事实边界

1. 你为什么把这个项目写成“课程学习 Agent”，而不是“模块化 MCP Server”？
2. 如果你把这个项目写成“企业知识库平台”，哪些地方算合理包装，哪些地方会过界？
3. 你怎么解释“当前 Web 主路径更完整，而 MCP 是并存接口”？
4. 你觉得这个项目最容易被误包装的点是什么？

### B2. 强动词与 ownership

1. 你说自己“设计了 Agent 主链路”，那 conversation、planner、tool loop、streaming 分别落在哪？
2. 你说自己“实现了知识问答工具”，那 `knowledge_query` 在 `HybridSearch` 前后做了什么？
3. 你说自己“做了学习闭环”，那 quiz、review、memory 三者怎么互相更新？
4. 你说自己“做了生产化存储”，那 Postgres、Redis、object store 分别存什么？
5. 你说自己“支持了 MCP”，那真实 MCP 实现和 `main.py` 是什么关系？

### B3. 指标与结果

1. 如果你写“提升了问答效果”，你会用什么指标解释？
2. 如果没有真实线上数据，你怎么避免简历变空话？
3. 这个项目的价值除了准确率，还能从哪些角度解释？
4. 你会如何区分“当前仓库可证明的结果”和“建议补充的业务数字”？

## C. 技术深挖题池

### C1. App factory、配置与生产存储

1. `create_app()` 启动时装了哪些关键组件？
2. `runtime.py` 为什么是理解生产切换的关键文件？
3. 为什么项目里同时存在 typed settings 和原始 YAML dict？
4. conversation、long-term memory、feedback、registry、task store 各自怎么持久化？
5. Redis 在限流、语义缓存、分布式熔断里分别扮演什么角色？

### C2. Agent runtime、planner 与 streaming

1. `Agent.chat()` 为什么先读 conversation，再做 before-message hooks，再 append 用户消息？
2. memory/review/skill 为什么要并行获取？
3. planner 的 `control_mode` 和 `selected_tool` 分别控制什么？
4. 如果一个请求是 composite task，会发生什么？
5. 流式回答里模型流、Agent 事件流、SSE 推流是怎么接起来的？

### C3. Tool orchestration

1. ToolRegistry 在这个项目里具体承担什么职责？
2. 为什么工具要暴露 schema，而不是只保留 Python 函数签名？
3. `knowledge_query`、`review_summary`、`quiz_generator`、`quiz_evaluator` 的目标分别是什么？
4. 为什么 `document_ingest` 也适合做成工具？
5. `ToolContext` 为什么存在？

### C4. Knowledge Query 与 HybridSearch

1. 为什么 `knowledge_query` 不能简化成“直接查 HybridSearch”？
2. QueryRouter 返回“不需要 RAG”时，系统应该怎么退化？
3. Semantic Cache 为什么按相似度命中，而不是简单字符串完全匹配？
4. RRF 为什么比直接对 dense/sparse 分数做线性加权更稳？
5. rerank、MMR、min_score、post-dedup 各自解决什么问题？

### C5. Ingestion Pipeline 与文档生命周期

1. `IngestionPipeline` 的阶段顺序是什么，为什么这样排？
2. source type 为什么对课程资料特别重要？
3. QuestionParser 在课程场景里带来了什么价值？
4. 图片 caption 为什么能增强图文混合资料的可检索性？
5. `DocumentManager.delete_document()` 为什么是跨存储一致性问题？

### C6. Conversation、Memory 与上下文工程

1. 聊天记录、长期记忆、上下文窗口三者有什么区别？
2. 跨多次 request，系统靠什么保持连续性？
3. 长期记忆为什么更适合结构化存储在 Postgres？
4. `MemoryContextEnhancer` 和 `MemoryRecordHook` 为什么要分离？
5. `ContextEngineeringFilter` 的 4 级压缩各自解决什么问题？

### C7. Web、前端与 MCP

1. `routes.py`、`models.py`、`chat_handler.py` 的边界为什么这样分？
2. 为什么当前用 SSE 就够了？
3. 历史会话 API、上传文件 API、健康检查 API 各在解决什么问题？
4. MCP 默认注册了哪些工具？为什么日志不能打到 stdout？
5. 为什么说 Web 更像当前产品路径，MCP 更像集成路径？

### C8. 稳定性、观测与测试

1. retry 和 circuit breaker 的区别是什么？
2. 为什么分布式熔断状态要放进 Redis，而不是只放单机内存？
3. trace、dashboard、evaluation、tests 为什么不是重复能力？
4. 当前测试是怎么分层的？这些分层各在防什么问题？
5. 如果继续演进，你会先补哪一条工程化链路？

## 最容易露馅的 killer questions

1. 你为什么把它写成 MCP Server，而不是课程学习 Agent？
2. 你说自己做了记忆，那聊天记录、长期记忆、上下文窗口三者分别是什么？
3. 你说自己做了生产化，那 Postgres、Redis、object store 分别负责什么？
4. 你说自己做了 HybridSearch，那 `knowledge_query` 和 `HybridSearch` 的边界到底怎么分？
5. 你说自己做了流式响应，那模型输出是怎么一步步流到前端的？
