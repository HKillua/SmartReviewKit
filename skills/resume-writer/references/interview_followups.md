# 面试追问库（简历防守版）

> 用于 `resume-writer` 在输出项目经历后，补一组足够扛深挖的追问。  
> 题目全部围绕当前代码主链路，而不是旧版项目叙事。  
> 默认从与用户简历主线最相关的 2-3 个模块中各选 2-4 题。

## 使用规则

- 如果简历主线偏“学习产品/Agent”，优先选 A、B、E、F 组
- 如果简历主线偏“RAG/检索”，优先选 C、D、G 组
- 如果简历主线偏“平台/后端”，优先选 A、B、F、G 组
- 如果简历里写了 “MCP”，必须补 G 组问题，但不能把 MCP 说成当前唯一主路径

---

## A. 项目定位与入口

| 方向 | 追问 | 深挖点 |
|------|------|--------|
| A1 | 你会怎么一句话解释这个项目“现在是什么”？ | 是否知道当前主产品是课程学习 Agent |
| A2 | 为什么 `run_server.py` 比 `main.py` 更能代表当前主入口？ | 是否分得清真实入口和历史残留 |
| A3 | `src/server/app.py` 在启动时具体装了哪些核心组件？ | 是否知道 app 装配不是单点初始化 |
| A4 | 默认课程、默认 collection、自动 ingest 是怎么接起来的？ | 是否能落到配置和启动逻辑 |
| A5 | 这个仓库为什么不能只包装成“一个模块化 MCP Server”？ | 是否理解当前产品叙事已经变化 |
| A6 | 如果面试官问你“这个项目最大的演进是什么”，你会怎么回答？ | 是否能讲清从早期 RAG/MCP 到学习 Agent 的演化 |

---

## B. Agent 运行时与流式交互

| 方向 | 追问 | 深挖点 |
|------|------|--------|
| B1 | `Agent.chat()` 一次请求大概经历哪些阶段？ | 是否真的读过主链路 |
| B2 | memory、review、skill 是在哪一步被注入到 system prompt 里的？ | 是否知道 gather 预取与 prompt build |
| B3 | tool loop 是怎么判断继续调用工具还是直接结束的？ | 是否理解 ReAct loop |
| B4 | 为什么这个项目用 SSE 就够了？什么时候才需要更复杂的协议？ | 是否理解单向事件流的适配性 |
| B5 | 除了文本，前端还能收到哪些事件？ | 是否知道 tool_start/tool_result/done 等事件 |
| B6 | hooks 和 middleware 的区别是什么？为什么不混成一个？ | 是否理解扩展点边界 |
| B7 | 如果工具执行失败或超时，用户侧会看到什么退化？ | 是否考虑异常处理 |
| B8 | `_post_message_tasks` 和 `flush()` 解决了什么问题？ | 是否理解后台保存与优雅退出 |

---

## C. Knowledge Query 与检索工具编排

| 方向 | 追问 | 深挖点 |
|------|------|--------|
| C1 | 为什么 `knowledge_query` 不能简化成“直接查 HybridSearch”？ | 是否知道工具前后还有路由、缓存、改写 |
| C2 | QueryRouter 在这个项目里扮演什么角色？ | 是否理解 need_rag / preferred_sources |
| C3 | Semantic Cache 为什么放在 `knowledge_query` 层而不是底层检索器里？ | 是否理解缓存粒度 |
| C4 | query rewrite、conversation-aware rewrite、HyDE、multi-query 分别适合什么场景？ | 是否区分查询增强手段 |
| C5 | `use_parent` 为真时，为什么要把 child chunk 换成 parent chunk？ | 是否理解检索精度和上下文完整性的权衡 |
| C6 | 冲突检测放在这里有什么价值？ | 是否理解不是所有问题都该直接把 top-k 扔给 LLM |
| C7 | 如果缓存命中但知识库刚更新了，系统会怎么避免旧结果长期污染？ | 是否知道 ingest 后 cache invalidation |
| C8 | 你会怎么解释“工具编排层”和“检索层”的边界？ | 是否能清晰分层 |

---

## D. HybridSearch 与排序链路

| 方向 | 追问 | 深挖点 |
|------|------|--------|
| D1 | 为什么这个项目要同时保留 dense 和 sparse 两路检索？ | 是否理解语义召回与关键词精确性的互补 |
| D2 | RRF 为什么适合融合这两路结果？ | 是否理解不同分值体系不可直接加权 |
| D3 | `HybridSearch` 里 QueryProcessor 的作用是什么？ | 是否只会背 BM25/RRF 而不知道前处理 |
| D4 | rerank、MMR、min_score、post-dedup 各自解决什么问题？ | 是否理解后处理链路 |
| D5 | 如果两路召回结果重叠很少，你觉得是好事还是坏事？ | 是否有分析思维 |
| D6 | 为什么 MMR 不等于 rerank？ | 是否分清相关性与多样性 |
| D7 | 相关度阈值设得过高或过低会怎样？ | 是否理解召回/精排平衡 |
| D8 | 如果面试官问“你怎么验证混合检索比单路更好”，你会怎么答？ | 是否能连到 evaluation/metrics |
| D9 | embedding cache 在这个检索体系里主要优化了哪一步？ | 是否知道 CachedEmbedding 的作用 |
| D10 | 如果 reranker 不可用，系统为什么还能继续工作？ | 是否理解 graceful degradation |

---

## E. Ingestion Pipeline、题库与多模态

| 方向 | 追问 | 深挖点 |
|------|------|--------|
| E1 | `IngestionPipeline` 的阶段顺序是什么？ | 是否能讲出完整入库流程 |
| E2 | 为什么这个项目特别重视 `source_type`？ | 是否理解课件/教材/题库差异 |
| E3 | `QuestionParser` 在课程学习场景里带来了什么价值？ | 是否理解题库优先能力 |
| E4 | Parent-child chunking 在这里解决什么问题？ | 是否理解检索精度与上下文平衡 |
| E5 | transform 阶段为什么不是只做 metadata enrichment？ | 是否理解 refiner/context/caption/dedup 的组合价值 |
| E6 | Image caption 为什么能让系统“搜文出图”？ | 是否知道 image-to-text 的实现策略 |
| E7 | 为什么要同时写 Chroma、BM25 和图片索引？ | 是否理解多存储协同 |
| E8 | `DocumentManager.delete_document()` 为什么要协调删除四类存储？ | 是否能落到一致性问题 |
| E9 | 同一份资料重复导入时，系统如何尽量避免重复计算？ | 是否理解 integrity / idempotence |
| E10 | 如果文件删了但 integrity 记录没清，后果是什么？ | 是否理解重导入风险 |

---

## F. Memory、Quiz、Review 与学习闭环

| 方向 | 追问 | 深挖点 |
|------|------|--------|
| F1 | 这个项目和普通问答系统最大的差异，为什么在学习闭环上？ | 是否能跳出“只是 RAG” |
| F2 | student profile、error memory、knowledge map、session memory 分别存什么？ | 是否理解多类记忆 |
| F3 | `quiz_generator` 为什么要先查题库，再退到 RAG，再退到 LLM fallback？ | 是否理解三层出题策略 |
| F4 | `quiz_evaluator` 具体会更新什么？ | 是否知道 error memory + mastery update |
| F5 | `review_summary` 怎么利用用户薄弱点做个性化总结？ | 是否理解 memory 注入 |
| F6 | review schedule hook 在用户体验上有什么价值？ | 是否理解主动复习 |
| F7 | ContextEngineeringFilter 的 4 个 level 是怎么工作的？ | 是否理解长上下文治理 |
| F8 | 为什么 memory enhancer 和 memory record hook 要分开设计？ | 是否理解读写分离 |
| F9 | 如果知识图谱的掌握度更新错了，会影响什么？ | 是否理解长期个性化后果 |
| F10 | 为什么说这个项目的核心不是单轮问答，而是持续学习？ | 是否能概括产品价值 |

---

## G. Web、MCP、配置与工程质量

| 方向 | 追问 | 深挖点 |
|------|------|--------|
| G1 | `routes.py`、`chat_handler.py`、`models.py` 为什么要分层？ | 是否理解 Web 边界 |
| G2 | 前端为什么直接用 SSE，而不是先上 WebSocket？ | 是否有工程取舍 |
| G3 | MCP 真实实现在哪？默认注册了哪些工具？ | 是否分清真实 MCP 与历史脚本 |
| G4 | 如果 stdio 协议把日志打到了 stdout，会发生什么？ | 是否理解协议约束 |
| G5 | 为什么这个仓库会同时出现 typed settings 和原始 YAML dict？ | 是否理解配置双轨现实 |
| G6 | 多 Provider 可插拔是怎么做出来的？ | 是否理解 Base + Factory + Config |
| G7 | trace、dashboard、evaluation 各解决什么问题？ | 是否理解可观测性闭环 |
| G8 | 当前测试里最值得提前说明的环境 caveat 是什么？ | 是否知道 `mcp` 依赖缺失会影响 smoke imports |
| G9 | 如果让你把它扩成多课程或多租户，先改哪几层？ | 是否有系统设计能力 |
| G10 | 你觉得当前代码里最像“历史债”的地方是什么？ | 是否能诚实讨论 `main.py` / package entry 偏差 |

---

## 常见简历措辞 -> 必问追问

| 简历措辞 | 建议必问 |
|----------|----------|
| “主导 Agent 架构设计” | `Agent.chat()`、prompt 组装、tool loop、hooks/middleware 是怎么连起来的？ |
| “实现混合检索” | `knowledge_query` 前后做了什么？`HybridSearch` 内部链路是什么？ |
| “构建课程知识库” | `IngestionPipeline` 如何处理课件/题库/图片？幂等怎么保证？ |
| “做了个性化学习闭环” | quiz、review、memory 三者如何互相更新？ |
| “支持 MCP 集成” | 真实 MCP 在哪？为什么现在 Web 主路径更完整？ |
| “负责系统工程化” | trace、evaluation、tests、配置双轨各解决什么问题？ |

## 最容易露馅的 killer questions

1. 你为什么把这个项目写成“MCP Server”，而不是“课程学习 Agent”？
2. 你说你做了 `knowledge_query`，那它和 `HybridSearch` 的边界到底怎么分？
3. 你说你实现了学习闭环，`quiz_evaluator` 改写了哪些 memory？
4. 你说 Web 和 MCP 都支持，那为什么 `main.py` 不是当前主入口？
5. 如果我删掉 `review_schedule` 或 `context_filter`，用户体验会退化在哪？
