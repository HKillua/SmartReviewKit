# 当前项目知识底稿（面试官使用）

> 以当前代码和 `DEV_SPEC.md` 为准。  
> 面试时要优先考查候选人是否理解“现在这个仓库是什么”，而不是背旧版名词。

## 1. 项目当前定位

- 当前主产品形态是课程学习 Agent
- 核心用户路径是 Web 对话学习，不是单独的 MCP CLI
- RAG 基础设施仍然是底座，但上层已经叠加了 Agent、Memory、Quiz、Review、Skills

## 2. 真实入口与装配关系

- Web 主入口：`run_server.py` -> `src/server/app.py`
- `src/server/app.py` 会装配：
  - LLM service
  - Memory stores
  - HybridSearch
  - QueryEnhancer
  - QueryRouter
  - SemanticCache
  - Agent tools
  - Skill registry/workflow
  - Agent 实例
- `main.py` 仍是早期占位入口，不能当作当前主运行链路
- `pyproject.toml` 里 `mcp-server = "main:main"` 仍带历史残留

## 3. Web 对话主链路

用户发起消息后，典型路径是：

1. `src/server/routes.py` / `src/server/chat_handler.py`
2. `src/agent/agent.py`
3. prompt 组装、conversation 读取、memory/skill 注入
4. ReAct tool loop 决定是否调用工具
5. 工具返回结果后继续生成
6. 通过 SSE 把文本增量、工具事件和结束状态返回前端

强回答至少要能说清 Agent 不只是“调用一下 LLM”，而是一个带会话、工具、记忆和 hook 的编排层。

## 4. Knowledge Query / RAG 主链路

`knowledge_query` 不是直接把 query 扔给向量库，它前后还有编排：

1. QueryRouter 判断意图或路由策略
2. Semantic Cache 尝试命中
3. Query rewrite / conversation-aware rewrite
4. 可选 HyDE / multi-query
5. 调用 `HybridSearch`
6. 可选 parent chunk、冲突检测、回答拼装

如果候选人只会说“用了 BM25 + 向量检索”，说明理解还停留在检索层，不理解工具编排层。

## 5. HybridSearch 要点

`HybridSearch` 是检索编排器，不是完整 RAG 的全部。

核心链路：

1. QueryProcessor
2. DenseRetriever + SparseRetriever 并行召回
3. RRF 融合
4. 可选 rerank
5. 可选 MMR
6. min_score 过滤与 post-dedup

强回答应能解释：

- 为什么要 dense + sparse
- 为什么 RRF 适合融合不同分值体系
- rerank/MMR 分别解决什么问题

## 6. IngestionPipeline 要点

入库链路不是“切块后丢进向量库”这么简单。

当前关键能力包括：

- 文件完整性检查
- PDF/PPTX/DOCX loader
- source type 推断：课件、教材、题库
- QuestionParser 解析题库
- Chunking / parent-child
- Transform：refiner、metadata、contextual enrich、caption、dedup
- dense / sparse 双编码
- Chroma / BM25 / 图片索引写入

强回答应能说出为什么这个项目特别重视题库和课件，而不是泛文档场景。

## 7. Memory 与学习闭环

当前项目区别于普通 RAG demo 的核心之一，是学习闭环：

- `review_summary`：总结考点
- `quiz_generator`：优先题库，其次 RAG 生成，最后 LLM fallback
- `quiz_evaluator`：评判答案并更新 memory
- `memory`：学生画像、错题、知识掌握度、session memory、上下文压缩
- `review_schedule`：给出复习建议

如果候选人说不清 `quiz_evaluator` 会更新什么，通常说明没有真正看过代码。

## 8. Web 与 MCP 双接口

- Web 层更像当前主产品接口
- MCP 层是真实存在的集成接口，在 `src/mcp_server/`
- MCP 通过 `ProtocolHandler` 注册和执行工具
- `query_knowledge_hub`、`list_collections`、`get_document_summary` 是当前默认工具

强回答应能说明：

- 为什么 MCP 适合外部 Agent 集成
- 为什么当前仓库里 Web 体验更完整
- 为什么要区分真实 MCP 实现和历史打包入口

## 9. 配置、存储与质量保障

候选人还应至少知道：

- 配置层有 typed settings 和原始 YAML dict 两套读取方式
- 存储不是单一数据库，而是 Chroma、BM25、SQLite memory、图片索引协同
- 项目有 trace、dashboard、evaluation runner 和 tests

## 10. DocumentManager 与跨存储一致性

- `DocumentManager` 不是入库流水线本身，而是面向“已入库文档”的生命周期管理层
- 它提供 list/detail/delete 这类跨存储操作
- `delete_document()` 会协调：
  - Chroma 向量
  - BM25 索引
  - 图片文件与图片索引
  - 文件完整性记录

强回答应能说明：

- 为什么只删 Chroma 不够
- 为什么文件完整性记录不删会影响重新导入

## 11. ContextEngineeringFilter 与长上下文治理

- `ContextEngineeringFilter` 不是简单的截断器，而是 4 级压缩策略：
  - sliding window
  - 大型 tool 结果卸载到文件
  - 老对话 LLM 摘要压缩
  - token budget 控制
- 它的价值在于：让 Agent 不会因为工具结果太长或历史太长而失控

强回答应能说明：

- 为什么工具结果卸载比直接保留全文更稳
- 为什么压缩要保留文件路径、函数名、薄弱点和下一步计划

## 12. Dashboard、EvalRunner 与质量闭环

- Dashboard 当前是多页面 Streamlit 应用，关注的是“系统可观测性”
- EvalRunner 负责批量跑 golden test set
- evaluator 体系负责把检索或问答质量结构化打分

强回答应能说明：

- 为什么 trace、dashboard、evaluation、tests 不是重复能力
- 为什么没有这些工具，RAG/Agent 调优很容易退化成拍脑袋

## 13. 配置双轨与 Provider 现实

- 基础设施层大量走 typed settings
- Agent/Web 装配层有不少地方直接读 YAML dict
- 这会让新人第一次读代码时觉得“配置系统不够统一”，但这是当前仓库的现实

强回答应能说明：

- 这不是架构理念，而是当前演进阶段的现实折中
- 面试时应该如实承认，而不是硬说“配置层已经完全统一”

## 14. 当前仓库现实与环境 caveat

- Web 主路径当前更稳、更完整
- MCP 有真实实现，但包脚本仍有历史偏差
- 当前环境如果缺 `mcp` 依赖，MCP 相关 smoke import 可能失败

强回答应能说明：

- 环境问题和架构问题要分开讲
- 不能因为某次本地依赖缺失，就把整个项目描述成“不可用”

## 15. 高频露馅点

- 把 `main.py` 说成当前主入口
- 说项目只是“模块化 MCP Server”，忽略学习 Agent 主体
- 说自己做了 quiz/memory，但解释不出 memory stores 和更新点
- 说做了 HybridSearch，却答不出 `knowledge_query` 在检索前后的编排
- 把历史数字、旧测试数、旧任务数当成现状
