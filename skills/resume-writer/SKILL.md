---
name: resume-writer
description: "基于当前仓库源码，为本项目生成简历项目经历、技术亮点、中英文 bullet 和面试防守话术。Use when the user asks to write resume bullets, package this repository into a project story, prepare interview-safe project experience, or defend a resume entry with current code facts."
---

# Resume Writer

目标：把当前仓库整理成“面试能自圆其说”的项目经历，而不是把历史入口、旧规划或未经核实的数据写进简历。

默认输出中文。只有在事实校准后再给英文版。

## 事实边界

- 先信源码，再信 `DEV_SPEC.md`，最后才信 README 和旧说明文档。
- 当前主产品形态是课程学习 Agent；RAG 是底座，不是全部。
- Web 主入口是 `run_server.py` -> `src/server/app.py`。
- `src/mcp_server/` 有真实 MCP 实现；`main.py` 和 `pyproject.toml` 中的 `mcp-server` 仍带历史残留。
- 聊天记录、长期记忆、反馈、文档注册、任务记录在生产环境可切到 Postgres。
- 限流、分布式熔断、语义缓存在生产环境可切到 Redis。
- 对象存储可切到 MinIO；默认本地开发仍可走本地目录。

## 先读什么

按这个顺序建立上下文：

1. `skills/resume-writer/references/resume_principles.md`
2. `skills/resume-writer/references/project_highlights.md`
3. `skills/resume-writer/references/interview_followups.md`
4. `DEV_SPEC.md`
5. `src/server/app.py`
6. `src/agent/agent.py`
7. `src/storage/runtime.py`
8. 需要补证据时，再读对应模块源码

如果材料冲突：

- 先信源码
- 再信 `DEV_SPEC.md`
- 旧引用材料只作辅助，不可反向覆盖当前实现

## 开始前最多确认 4 件事

一次性确认，不要来回盘问：

1. 目标岗位
   例如：LLM 应用、RAG/检索、Agent、后端平台、教育 AI
2. 主叙事角度
   例如：课程学习 Agent、RAG 基建、生产化 runtime、Web/MCP 双接口
3. 输出形式
   例如：4-6 条简历 bullet、项目经历段、英文版、面试防守版
4. 数字边界
   用户是否提供了真实业务数据；如果没有，就不要编数字

如果用户没给信息，默认按：

- 岗位：LLM 应用 / 后端平台
- 主叙事：课程学习 Agent + Agent runtime + RAG 基础设施
- 输出：中文简历版 + 技术栈关键词 + 面试追问

## 工作流

### 1. 先选主叙事，不先堆技术名词

优先从这 4 条主线里选 1 条作为主线，最多补 1 条副线：

- 课程学习 Agent
- Agent runtime / tool orchestration
- RAG / ingestion / hybrid retrieval
- 生产化 runtime / storage stack

不要把 4 条主线平均展开，简历会失焦。

### 2. 只挑能落到代码文件的亮点

优先从 `references/project_highlights.md` 里选 3-5 个亮点。

每个亮点至少能回答：

- 做了什么
- 在哪条链路里
- 落在哪些文件
- 为什么要这样设计

### 3. 把亮点写成“动作 + 机制 + 价值”

推荐句式：

- 动作：设计 / 实现 / 重构 / 编排 / 治理 / 接入
- 机制：Agent、HybridSearch、IngestionPipeline、Memory、Postgres/Redis、SSE、MCP
- 价值：提升学习闭环、增强检索稳定性、补齐生产共享状态、支持多接口访问

### 4. 量化必须保守

只允许 3 类数字：

1. 用户亲自提供的真实数据
2. 你现场从当前仓库核实出的数据
3. 明确标注为“建议补充”的占位数字

没有证据时，不写准确率、QPS、线上规模、团队规模、用户数。

### 5. 默认补“面试防守”

只要用户不是明确拒绝，输出简历内容后默认补：

1. 技术栈关键词
2. 5-8 个高频深挖问题
3. 哪些数字仍需用户确认

如果用户说“面试安全版”或“防守版”，再额外补：

- 3-5 个 killer questions
- 哪些说法不能写

## 当前项目可安全主张的方向

优先围绕这些事实展开：

- 课程学习 Agent，而不是单纯 RAG demo
- `create_app()` 统一装配 LLM、检索、记忆、工具、技能、存储和 Web 路由
- `Agent.chat()` 串起会话加载、memory/review/skill 注入、planner、tool loop、streaming 和后台保存
- `knowledge_query` 不是裸查向量库，而是工具编排层
- 检索链路是 dense + sparse + RRF + rerank + MMR + 过滤/去重
- 入库链路支持 PDF/PPTX/DOCX、题库解析、多模态增强、双路索引写入
- 长期记忆是结构化记忆，不是 Markdown 记事本，也不是单纯向量记忆
- 生产运行时支持 Postgres + Redis + object store 的共享状态切换
- Web 与 MCP 双接口并存，但 Web 产品链路更完整

## 必须避免的说法

- 不要把 `main.py` 写成当前主入口
- 不要把“未来规划”写成“已完成上线能力”
- 不要把历史文档里的旧数字直接写进简历
- 不要把聊天记录、长期记忆、上下文窗口混成一个概念
- 不要把 Redis 说成长期记忆主存储
- 不要把 MCP 说成当前唯一产品形态
- 不要把“生产可切换”写成“已经大规模线上验证”

## 输出要求

- 默认给 1 个主版本，不要一次给 4 种互相竞争的版本
- 如果给英文版，先给中文事实版，再给英文版
- 语言风格偏简洁、硬核、可落地
- 所有亮点都优先指向真实文件和真实链路
