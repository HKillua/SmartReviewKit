# 面试题库（扩展版）

> 目标：既能覆盖当前项目的广度，也能扛住深挖。  
> 使用顺序建议：先从 A 组选开场题，再根据简历从 B 组选辩护题，最后从 C 组挑 3-4 个技术主题深挖。  
> 用户没有简历时，可直接从 A + C 组构成完整面试。

## A. 开场题池：项目定位与主链路

1. 现在最准确地描述这个仓库，应该怎么说？
2. 为什么你认为它已经不是单纯的 MCP/RAG demo？
3. `run_server.py` 和 `src/server/app.py` 在当前项目里分别负责什么？
4. 从网页里发一条消息，到最终拿到流式回答，中间会经过哪些模块？
5. 你怎么向面试官解释 Agent 层和 RAG 层的边界？
6. 为什么 `main.py` 不能代表当前主运行链路？
7. 默认课程、默认 collection 和自动 ingest 在哪里配置或触发？
8. 当前项目里 Web 接口和 MCP 接口分别服务什么场景？
9. 你觉得这个仓库最像“产品”的地方是哪一段链路？
10. 如果你只能用 3 个模块来概括全项目，你会选哪 3 个，为什么？
11. 你怎么看这个项目从“基础设施”到“学习产品”的演进？
12. 如果我要快速判断一个候选人是不是看过这个仓库，你觉得第一题应该问什么？

## B. 简历辩护题池

### B1. 项目定位与包装辨别

1. 你为什么把这个项目写成“课程学习 Agent”，而不是“模块化 MCP Server”？
2. 如果面试官只看到你的简历条目，他会不会误以为这只是个 RAG demo？你怎么纠偏？
3. 你简历里强调的是产品闭环、检索引擎，还是 Agent 编排？为什么这样选？
4. 如果你把这个项目写成“企业知识库平台”，哪些部分是合理包装，哪些部分会过界？
5. 你会怎么解释“当前主路径更完整的是 Web，而不是打包脚本里的 MCP CLI”？
6. 你觉得这个项目最容易被误包装的点是什么？

### B2. 强动词与 ownership

1. 你说自己“设计了 Agent 主链路”，那 system prompt、tool loop、streaming 分别在哪些文件里落地？
2. 你说自己“实现了知识问答工具”，那 `knowledge_query` 在 `HybridSearch` 前后分别做了什么？
3. 你说自己“搭了学习闭环”，那 quiz、review、memory 三者是如何互相更新的？
4. 你说自己“做了资料入库流水线”，那 source type、question parser、多模态增强分别落在哪层？
5. 你说自己“支持了 MCP”，那真实实现在哪？默认注册哪些工具？
6. 你说自己“搭了可观测性”，那 trace、dashboard、evaluation 分别解决什么问题？
7. 你说自己“做了配置驱动的可插拔架构”，那换 provider 到底需要改几类文件？
8. 你说自己“做了工程质量保障”，那测试、评估、trace 三者怎么形成闭环？

### B3. 指标、结果与影响

1. 你简历里如果写“提升了问答效果”，这个效果你准备用什么指标解释？
2. 如果你写“提升了检索质量”，会选 Hit Rate、MRR 还是别的？为什么？
3. 如果面试官追问“这些结果是在线上的还只是本地验证”，你怎么回答最稳？
4. 如果简历里没有真实数字，你会怎么避免被面试官质疑成空话？
5. 你怎么解释“这个项目的价值不只在准确率，还在学习闭环和状态化体验”？
6. 如果面试官问“你做的这套系统实际服务谁”，你会用什么场景来回答？
7. 你会如何区分“仓库里当前可证明的结果”和“建议补充的业务指标”？
8. 如果被要求解释测试或评估数据的可信度，你会从哪里开始说？

### B4. 模块落地细节核验

1. 你要是写“做了 HybridSearch”，那 QueryProcessor、dense、sparse、RRF、rerank、MMR 的顺序能说清吗？
2. 你要是写“做了 memory”，那 student profile、error memory、knowledge map、session memory 的边界是什么？
3. 你要是写“做了 quiz”，那出题为什么要分题库优先、RAG 生成、LLM fallback 三层？
4. 你要是写“做了 review summary”，那它和普通问答相比，多了哪些结构化要求？
5. 你要是写“做了 ingestion”，那 DocumentManager 和 IngestionPipeline 的职责有什么不同？
6. 你要是写“做了前端流式交互”，那前端到底消费了哪些事件？
7. 你要是写“支持图片检索”，那 image caption 是如何进入文本检索链路的？
8. 你要是写“做了 MCP 集成”，那为什么 stdout 不能打日志？
9. 你要是写“做了配置驱动架构”，那 typed settings 和 YAML dict 双轨会带来什么现实影响？
10. 你要是写“做了 observability”，那 dashboard 的页面和 trace 的关系是什么？

### B5. 取舍、反思与重设计

1. 如果现在回头看，你觉得这个项目里最值得重构的一个历史债是什么？
2. 如果面试官问“为什么不把所有入口统一掉”，你怎么回答？
3. 你觉得这个项目最应该保留的设计取舍是什么？
4. 如果要求你把这套系统做成多课程或多租户，第一批改动会落在哪？
5. 你觉得当前代码里最容易让新人误解的地方是什么？
6. 如果给你一周时间继续演进，你优先补哪条链路？

## C. 技术深挖题池

### C1. Agent 运行时与流式链路

1. `Agent.chat()` 为什么要先做 before-message hooks，再 append user message？
2. memory context、review context、skill workflow 为什么要并行获取？
3. `_tool_loop()` 是怎么把一次对话拆成多轮 LLM 和工具交互的？
4. `_accumulate_tool_calls()` 为什么存在？它解决了流式 tool call 的什么问题？
5. 如果 tool schemas 为空，这个 Agent 会以什么模式运行？
6. 后台保存 conversation 为什么不直接阻塞在主响应里？
7. `flush()` 在什么场景下尤其重要？
8. 你会怎么解释“这是一个会话编排器，不只是一个聊天接口”？

### C2. Tool orchestration 与 ToolRegistry

1. ToolRegistry 在这个项目里具体承担什么职责？
2. 为什么工具要暴露 schema，而不是只保留 Python 函数签名？
3. `knowledge_query`、`document_ingest`、`review_summary`、`quiz_generator`、`quiz_evaluator` 为什么适合被统一成工具？
4. 如果你要新增一个学习工具，最少需要接到哪几层？
5. Agent 为什么需要 `ToolContext`，而不是把所有上下文直接塞进 args？
6. 如果工具执行结果很长，为什么 ContextEngineeringFilter 会考虑把工具结果卸载到文件？

### C3. Knowledge Query、Router 与 Cache

1. `knowledge_query` 为什么把路由、缓存、查询增强和检索放在一个工具里编排？
2. QueryRouter 返回 `need_rag = False` 时，Agent 应该怎么退化？
3. `preferred_sources` 和 metadata filter 的关系是什么？
4. Semantic Cache 为什么按查询相似度命中，而不是简单字符串完全匹配？
5. `document_ingest` 为什么要在导入后主动 invalidates semantic cache？
6. conversation-aware rewrite 和普通 rewrite 的触发条件应该怎么区分？
7. HyDE 在这个项目里为什么默认更适合按需启用？
8. multi-query decomposition 对检索质量和成本各有什么影响？

### C4. HybridSearch、RRF、Rerank、MMR

1. DenseRetriever 和 SparseRetriever 在课程场景里分别更擅长什么类型的问题？
2. RRF 为什么比直接对分数加权更稳？
3. QueryProcessor 在混合检索前做的事情为什么不能忽略？
4. 如果 dense 和 sparse 的 top_k 设得不一样，RRF 还能正常工作吗？
5. rerank 和 MMR 为什么是两类不同问题？
6. min_score 阈值设太高或太低，各会出现什么问题？
7. post-dedup 为什么不应该过早做？
8. 如果用户问的是术语型、公式型、概念型问题，这套混合检索各自可能怎么表现？

### C5. IngestionPipeline、Loader 与 Chunking

1. `IngestionPipeline` 的阶段顺序是什么，为什么这样排？
2. source type 推断为什么对课程资料特别重要？
3. `QuestionParser` 为什么是课程场景的关键差异化点？
4. Parent-child chunking 在这里解决的是召回问题还是生成问题？
5. metadata、contextual enrich、caption、dedup 各自更偏“提升召回”还是“提升可读性”？
6. 图片 caption 为什么能复用现有文本检索链路？
7. 同一份资料重复导入时，系统是怎么尽量避免重复处理的？
8. `DocumentManager.delete_document()` 为什么要跨 Chroma、BM25、ImageStorage、Integrity 一起删？

### C6. Memory、Quiz、Review

1. student profile、error memory、knowledge map、session memory 为什么不合并成一个大表？
2. `quiz_generator` 为什么先查 question bank，再查全库，再退到纯 LLM？
3. `quiz_evaluator` 评完后，哪些信息会写回 memory？
4. `review_summary` 如何把薄弱点带进复习摘要？
5. review schedule hook 为什么适合在会话开始时介入？
6. `KnowledgeMapMemory.update_mastery()` 这种机制为什么比单次对错更重要？
7. ContextEngineeringFilter 的四级压缩，各自解决什么问题？
8. 如果 memory 注入过重或过旧，会对 Agent 造成什么负面影响？

### C7. Web、SSE 与前端交互

1. `routes.py`、`models.py`、`chat_handler.py` 的边界为什么这样分？
2. 为什么前端用 EventSource/SSE 就能满足当前交互需求？
3. 文件上传为什么不能只是“把文件丢到磁盘”这么简单？
4. 静态资源和首页是怎么挂到 FastAPI 上的？
5. 如果一次对话里工具调用很多，前端怎么知道当前阶段发生了什么？
6. 如果以后换成更复杂的前端框架，你觉得当前后端最可能保留的接口形态是什么？

### C8. MCP 协议与工具暴露

1. MCP 真实入口和 Web 主入口分别在哪？
2. `ProtocolHandler` 为什么要显式注册工具并持有 schema？
3. `query_knowledge_hub` 和 Agent 侧的 `knowledge_query` 在职责上有什么差异？
4. stdio transport 下 stdout/stderr 的边界为什么是协议正确性的关键？
5. 当前默认 MCP 工具有哪些？各自适合什么调用场景？
6. 为什么说 MCP 在当前仓库里更像“集成接口”，而不是“最完整的终端产品入口”？

### C9. 配置、Provider 与存储

1. typed settings 和原始 YAML dict 双轨并存，会给开发者带来什么阅读成本？
2. LLM、Embedding、VectorStore、Reranker 的 provider 工厂在这个项目里怎么工作？
3. 为什么 Chroma、BM25、SQLite memory、图片索引不适合强行统一为一个后端？
4. 你会怎么解释当前 `settings.yaml` 里 retrieval、memory、routing、guardrails 的几个关键开关？
5. 如果要把单课程扩成多课程，collection、数据目录和默认配置需要怎么演进？
6. 如果面试官问你“provider 换掉会不会波及上层业务代码”，你怎么证明影响被隔离了？

### C10. 可观测性、评估与测试

1. ingestion trace 和 query trace 各自记录什么？
2. Dashboard 为什么要按 pages + services 分层？
3. EvalRunner 跑一次 golden set 的主流程是什么？
4. Hit Rate、MRR、faithfulness 在这个项目里分别适合回答什么问题？
5. 为什么 trace、evaluation、tests 三者应该形成闭环，而不是各自为政？
6. 当前环境里缺 `mcp` 依赖导致 smoke import 失败，这类问题为什么要和“主链路设计是否正确”区分开？

### C11. 系统设计与扩展题

1. 如果要支持多课程/多租户，最先需要改的是哪个边界？
2. 如果要把 Web 和 MCP 的入口重新统一，你会从哪层着手？
3. 如果要进一步提升学习闭环，而不是只提升检索效果，你会优先补哪条链路？
4. 如果要让系统支持更复杂的教师工作流或课程编排，你会把逻辑放进 skill、tool 还是 memory？
5. 你认为当前仓库里最值得继续演进的一条技术线是什么？
6. 哪一段代码最能代表你对“工程化 AI 系统”的理解？
