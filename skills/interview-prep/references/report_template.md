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
- {是否把旧入口、旧叙事、旧数字当成现状}

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
| 简历真实性 | x/10 | {comment} |
| 表达清晰度 | x/10 | {comment} |

## 7. 下一轮建议

- {下一步该怎么补}
- {建议继续练哪类题}
```

## 分数解释

- `9-10`: 能准确描述当前产品定位，主链路和关键模块都能落到代码
- `7-8`: 主体理解正确，但个别链路或设计取舍不够扎实
- `5-6`: 知道大方向，但常把概念答案当源码答案
- `0-4`: 对当前项目理解明显失真，存在较强包装风险

## 强回答检查点

### 1. 项目定位与入口

强回答应包含：

- 当前是课程学习 Agent
- Web 主入口是 `run_server.py` -> `src/server/app.py`
- `main.py` 只是历史占位入口
- MCP 有真实实现，但不是当前最完整的用户路径

### 2. Agent 运行时

强回答应包含：

- Conversation / prompt builder / memory injection
- ReAct tool loop
- SSE 事件流
- hooks / middleware 的扩展点

### 3. Knowledge Query 与 HybridSearch

强回答应包含：

- `knowledge_query` 在 `HybridSearch` 前后都有编排
- dense + sparse + RRF
- rerank / MMR / min_score / 去重
- 不把“检索层”和“完整问答层”混为一谈

### 4. IngestionPipeline

强回答应包含：

- 文件检查、loader、chunking、transform、编码、写库
- PDF/PPTX/DOCX 与题库导向
- parent-child 或 question parsing 等课程相关特性

### 5. Memory 与学习闭环

强回答应包含：

- student profile、error memory、knowledge map、session memory
- quiz generator / evaluator / review summary 的关系
- 这是“持续学习系统”，不是单轮问答壳子

### 6. Web / MCP / 配置 / 质量

强回答应包含：

- Web 与 MCP 两套接口的定位差异
- 包脚本与真实运行路径的不一致
- trace / dashboard / evaluation / tests 的作用
