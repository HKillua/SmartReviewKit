# 简历编写原则（针对当前仓库）

## 1. 先讲产品，再讲技术

这个项目最容易写偏的地方，是把它写成“一个传统 RAG demo”。

更准确的表述通常是三选一：

- 面向课程学习场景的 Agent 平台
- 带混合检索与资料入库能力的 RAG 基础设施
- 同时支持 Web 对话与 MCP 集成的知识系统

写之前先选主叙事，不要三个方向平均用力。

## 2. 只写自己能在面试里落到文件级的内容

每条 bullet 最好都能回答下面三个问题：

1. 具体在哪个模块做的
2. 这条链路前后依赖什么
3. 为什么这么设计，而不是别的方案

如果答不出，宁可弱化，不要硬写。

## 3. 推荐的 bullet 结构

每条 bullet 尽量包含三段：

- 动作：设计 / 实现 / 重构 / 编排 / 优化
- 手段：用了什么模块、机制或架构
- 结果：解决了什么问题，或补齐了哪种能力

示例：

- 设计基于 ReAct 的 Agent 编排链路，将工具调用、会话持久化与 SSE 流式输出统一到单次对话执行模型中
- 实现 dense + sparse 混合检索，并通过 RRF、rerank 和 MMR 组合提升知识问答的召回与排序质量

## 4. 量化规则

优先级如下：

1. 用户提供的真实业务指标
2. 当前仓库重新验证过的本地事实
3. 无法核实时，不写具体数字

不建议直接写：

- “准确率提升 30%”
- “QPS 提升 X 倍”
- “支持万级并发”

除非这些数字被明确证明。

## 5. 关键词建议

可按岗位选择关键词，不要机械堆满：

- RAG/检索：Hybrid Search、BM25、Dense Retrieval、RRF、Rerank、MMR、Semantic Cache
- Agent：ReAct、Tool Calling、Memory、Skill Workflow、Guardrails、SSE Streaming
- 后端/平台：FastAPI、Factory Pattern、Config-Driven、SQLite、Chroma、Observability
- 集成：MCP、JSON-RPC、Protocol Handler、Tool Registry

## 6. 当前仓库的高风险误区

- 把 `main.py` 当成当前主入口
- 把早期规划写成“现已完成”
- 使用旧文档里的测试数、任务数、skill 数、开发周期等数字
- 把“MCP”写成唯一产品形态，忽略现在的 Web Agent 主链路

## 7. 收尾时要主动补的内容

最终输出后，默认再给用户补三样东西：

1. 技术栈关键词
2. 3-5 个面试追问
3. 哪些数字还需要用户自己确认
