# 面试追问库（简历防守版）

> 用于 `resume-writer` 在输出项目经历后，补一组基于当前代码的深挖问题。

## 使用规则

- 如果简历主线偏 Agent，优先选 A、C、F
- 如果简历主线偏 RAG，优先选 D、E
- 如果简历主线偏后端 / 平台，优先选 B、G、H
- 如果简历里写了 MCP，一定补 G 组
- 如果简历里写了生产化，一定补 B 组和 H 组

## A. 项目定位与入口

1. 你会怎么一句话解释这个项目“现在是什么”？
2. 为什么 `run_server.py` 比 `main.py` 更能代表当前主入口？
3. Web 路径和 MCP 路径分别服务什么场景？
4. 这个仓库为什么不能只包装成“模块化 MCP Server”？

## B. App 装配、配置与生产存储

1. `src/server/app.py` 启动时到底装了哪些核心组件？
2. `src/storage/runtime.py` 为什么重要？它解决了什么运行时切换问题？
3. 生产环境下聊天记录、长期记忆、反馈和入库任务分别落在哪些后端？
4. 为什么长期记忆主存储是 Postgres，而不是 Redis 或向量数据库？
5. Redis 在这套系统里主要承担哪些职责？

## C. Agent Runtime、Planner 与 Streaming

1. `Agent.chat()` 一次请求大概经历哪些阶段？
2. planner / control mode / composite task 在这个 Agent 里起什么作用？
3. memory、review、skill 是在哪一步被注入的？
4. 除了文本，前端还能收到哪些流式事件？
5. `_post_message_tasks()` 和 `flush()` 分别解决了什么问题？

## D. Knowledge Query 与 HybridSearch

1. `knowledge_query` 为什么不能简化成“直接调 HybridSearch”？
2. QueryRouter、Semantic Cache、query rewrite、HyDE 分别在什么阶段介入？
3. dense + sparse 为什么要一起保留？
4. RRF、rerank、MMR、post-dedup 各自解决什么问题？
5. 为什么会有 parent chunk 和冲突检测？

## E. Ingestion Pipeline 与文档生命周期

1. `IngestionPipeline` 的阶段顺序是什么？
2. 为什么课程场景要区分教材、课件和题库这几类 source type？
3. `QuestionParser`、image caption、contextual enrich 的价值各是什么？
4. `DocumentManager.delete_document()` 为什么要跨多个存储一起删？
5. 入库完成后为什么要联动处理缓存和文档注册信息？

## F. Conversation、Memory 与上下文工程

1. 聊天记录、长期记忆、上下文窗口分别是什么？
2. 多次 request 之间，系统如何延续上下文？
3. `MemoryContextEnhancer` 和 `MemoryRecordHook` 为什么要分开？
4. `ContextEngineeringFilter` 的 4 级压缩各自解决什么问题？
5. 为什么“上下文窗口”本身通常不做长期持久化？

## G. Web、前端与 MCP

1. `routes.py`、`chat_handler.py`、`models.py` 为什么要分层？
2. 为什么当前用 SSE 就够了？
3. MCP 默认注册了哪些工具？它们和 Agent 侧工具是什么关系？
4. stdio 协议下为什么日志不能打到 stdout？
5. 上传文件、会话列表和历史对话分别是怎么暴露给前端的？

## H. 生产化、稳定性与质量闭环

1. retry 和 circuit breaker 的区别是什么？
2. 为什么要把分布式熔断状态放进 Redis，而不是只放进单进程内存？
3. 你会怎么解释这个项目里的“共享状态”和“持久化状态”的区别？
4. trace、dashboard、evaluation、tests 各自解决什么问题？
5. 当前最像“历史债”的地方是什么？如果继续演进你会先改哪一层？

## 常见简历措辞 -> 必问追问

- “设计了 Agent 架构”
  追问：`Agent.chat()`、planner、tool loop、streaming、background save 怎么连起来？
- “实现了混合检索”
  追问：`knowledge_query` 在 `HybridSearch` 前后还做了什么？
- “做了个性化记忆”
  追问：长期记忆、聊天记录、上下文窗口怎么区分？生产下存哪？
- “做了生产级稳定性”
  追问：Redis 在限流、缓存、分布式熔断里分别扮演什么角色？
- “支持 MCP 集成”
  追问：真实 MCP 在哪？为什么当前 Web 路径更完整？

## 最容易露馅的 killer questions

1. 你为什么把它写成 MCP Server，而不是课程学习 Agent？
2. 你说自己做了 memory，那聊天记录、长期记忆、上下文窗口三者分别是什么？
3. 你说自己做了生产化，那 Postgres、Redis、对象存储分别存什么？
4. 你说自己做了混合检索，那 `knowledge_query` 和 `HybridSearch` 的边界到底怎么分？
5. 你说自己做了流式响应，那模型流、Agent 事件流、SSE 推流是怎么接起来的？
