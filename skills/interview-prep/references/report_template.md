# 面试报告模板

> 面试结束后按这个结构输出。  
> 重点不是“礼貌总结”，而是明确指出候选人是否真的理解当前仓库。

## 报告模板

```markdown
# 项目模拟面试报告

**项目**: Modular RAG MCP Server
**时间**: {datetime}
**风格**: {style}
**总评**: {overall_rating}
**总分**: {score}/10

## 1. 结论

- 是否真正理解当前项目定位: {yes/no + 说明}
- 是否能讲清主链路: {yes/no + 说明}
- 是否能区分 conversation / long-term memory / context window: {yes/no + 说明}
- 是否能说清生产存储与共享状态边界: {yes/no + 说明}
- 是否存在明显“简历包装站不住”的点: {yes/no + 说明}

## 2. 问答记录

| 轮次 | 问题 | 回答判断 | 关键依据 | 缺失点 | 相关文件 |
|------|------|----------|----------|--------|----------|
| 1 | {question} | ✅/⚠️/❌ | {why} | {missing} | {files} |
| 2 | ... | ... | ... | ... | ... |

## 3. 亮点

- {答得好的点，必须具体到模块或设计判断}
- {另一个强项}

## 4. 风险点 / 露馅点

- {错误表述或讲不清的点}
- {是否把历史入口、旧叙事、旧数字当成现状}

## 5. 建议补看的文件

- {file}: {why}
- {file}: {why}

## 6. 分项评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 项目定位准确度 | x/10 | {comment} |
| 主链路掌握 | x/10 | {comment} |
| 代码级细节 | x/10 | {comment} |
| 架构取舍理解 | x/10 | {comment} |
| 生产化边界理解 | x/10 | {comment} |
| 简历真实性 | x/10 | {comment} |
| 表达清晰度 | x/10 | {comment} |

## 7. 下一轮建议

- {下一步该怎么补}
- {建议继续练哪类题}
```

## 分数解释

- `9-10`: 项目定位、主链路、生产边界和关键模块都能落到代码
- `7-8`: 主体理解正确，但个别链路或取舍不够扎实
- `5-6`: 知道大方向，但经常把概念答案当源码答案
- `0-4`: 对当前项目理解明显失真，存在较强包装风险

## 强回答检查点

### 1. 项目定位与入口

强回答应包含：

- 当前是课程学习 Agent
- Web 主入口是 `run_server.py` -> `src/server/app.py`
- `main.py` 不是当前 Web 主链路入口
- MCP 有真实实现，但不是当前最完整的用户路径

### 2. Agent runtime

强回答应包含：

- conversation 读取与 prompt 预处理
- planner / tool loop
- SSE 事件流
- 后台保存与 hooks / middleware

### 3. Retrieval 与 Ingestion

强回答应包含：

- `knowledge_query` 在 `HybridSearch` 前后都有编排
- dense + sparse + RRF + rerank + MMR
- ingestion 不是“切块后丢向量库”
- 题库解析、图文增强、文档生命周期是课程场景重点

### 4. Conversation、Memory 与上下文工程

强回答应包含：

- conversation、long-term memory、context window 三者边界
- 长期记忆是结构化持久化，不是 Markdown 记事本
- context window 本身通常不做长期持久化
- `ContextEngineeringFilter` 解决长上下文治理

### 5. 生产化边界

强回答应包含：

- conversation / memory / feedback / registry 等持久化边界
- Redis 在 rate limit、cache、distributed breaker 中的作用
- retry / circuit breaker 的区别
- “共享状态”和“持久化状态”不是一个概念
