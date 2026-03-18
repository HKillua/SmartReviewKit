# 简历编写原则（基于当前仓库）

## 1. 先定主叙事，再写 bullet

这个项目当前最稳的 4 条叙事：

1. 课程学习 Agent
2. Agent runtime / tool orchestration
3. RAG 基础设施
4. 生产化运行时与共享存储

一份简历里只选 1 条主线，最多补 1 条副线。

## 2. 每条 bullet 都要能落到文件级

最少能回答下面 3 个问题：

1. 在哪条链路里做的
2. 落在哪些文件
3. 为什么这样设计

答不出来，就弱化，不要硬写。

## 3. 推荐 bullet 公式

每条 bullet 尽量同时包含：

- 动作：设计 / 实现 / 重构 / 编排 / 治理 / 接入
- 机制：Agent、HybridSearch、IngestionPipeline、Memory、Postgres/Redis、SSE、MCP
- 价值：提升体验、增强稳定性、补齐生产共享状态、支持学习闭环

示例结构：

- 设计基于 ReAct 的 Agent 运行时，将会话管理、技能注入、工具调用与 SSE 流式输出统一到单次对话编排链路中
- 实现 dense + sparse 混合检索，并通过 RRF、rerank 和 MMR 组合提升课程知识问答的召回与排序稳定性

## 4. 数字使用规则

优先级如下：

1. 用户提供的真实业务数据
2. 你从当前仓库重新核实过的数据
3. 明确标注为“建议补充”的数字

没有证据时，不写：

- 准确率提升百分比
- QPS / TPS
- 并发规模
- 用户规模
- 团队规模

## 5. 岗位映射建议

### LLM / Agent 岗

优先强调：

- `src/server/app.py`
- `src/agent/agent.py`
- `src/agent/planner/task_planner.py`
- `src/agent/tools/`
- `src/agent/memory/`

### RAG / 检索岗

优先强调：

- `src/agent/tools/knowledge_query.py`
- `src/core/query_engine/`
- `src/ingestion/pipeline.py`
- `src/ingestion/transform/`
- `src/ingestion/document_manager.py`

### 后端 / 平台岗

优先强调：

- `src/server/app.py`
- `src/storage/runtime.py`
- `src/storage/postgres_backends.py`
- `src/agent/hooks/`
- `src/core/settings.py`

## 6. 当前仓库最稳的安全事实

- 当前主产品是课程学习 Agent
- Web 主入口是 `run_server.py` -> `src/server/app.py`
- `Agent.chat()` 串起会话、记忆、planner、tool loop 和 streaming
- 长期记忆是结构化持久化记忆，生产可走 Postgres
- 共享运行时状态可走 Redis，例如限流、语义缓存、分布式熔断
- 入库链路支持 PDF/PPTX/DOCX、题库解析、多模态增强
- 仓库同时提供 Web Agent 与 MCP 接口

## 7. 高风险误区

- 把 `main.py` 当当前主入口
- 把 MCP 说成当前唯一产品形态
- 把 Redis 说成长期记忆主存储
- 把上下文窗口说成“长期保存的记忆”
- 把历史规划、旧测试数字、旧任务分期写成当前事实

## 8. 默认收尾

生成简历内容后，默认再补：

1. 技术栈关键词
2. 高频追问
3. 仍待用户确认的数字
