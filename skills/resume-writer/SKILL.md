---
name: resume-writer
description: "基于当前代码与 DEV_SPEC，为本项目生成可面试自圆其说的简历项目经历、技术亮点和中英文 bullet。Use when user says '写简历', 'resume', '项目经历', '简历项目', or asks to package this repository into a resume-ready project story."
---

# Resume Writer

为这个仓库写简历时，必须把“项目当前真实状态”放在第一位。

## 事实来源

按这个顺序取材：

1. `DEV_SPEC.md`，用于理解当前产品定位
2. `references/resume_principles.md`
3. `references/project_highlights.md`
4. `references/interview_followups.md`
5. 需要补证据时，再读对应源码文件

如果资料冲突：

- 先信当前源码
- 再信 `DEV_SPEC.md`
- 其他长期未维护文档默认低优先级，除非用户明确要求引用

## 你要先确认的 4 件事

一次性问清，不要来回盘问：

1. 目标岗位
   例如：RAG/LLM 应用、后端、Agent、平台架构、全栈 AI
2. 项目包装角度
   例如：课程学习 Agent、RAG 基础设施、MCP 集成、综合型
3. 真实业务背景
   如果没有真实业务，就明确按“通用学习/知识检索平台”来写
4. 输出约束
   例如：中文/英文、几条 bullet、是否要量化、是否需要面试追问

## 生成流程

### 1. 先定叙事，不先堆技术名词

优先从以下三种叙事里选一个：

- 学习产品型：课程学习 Agent，强调问答、测验、复习、记忆闭环
- RAG 基建型：文档摄取、混合检索、重排、缓存、路由、幂等入库
- Agent 平台型：ReAct tool loop、memory、hooks、guardrails、SSE streaming

如果用户想写得更全面，可以组合，但主线只能有一条。

### 2. 从亮点库里选 3-5 个可证明亮点

只能选 `references/project_highlights.md` 里能被当前代码支撑的点。

默认优先级：

- RAG/检索岗位：混合检索、数据摄取、Query Tool 编排、评估/可观测性
- 后端/平台岗位：FastAPI 装配、Agent 编排、配置驱动、存储协同、MCP
- Agent 岗位：ReAct Agent、Memory、Skills、Hooks/Guardrails、知识工具链

### 3. 输出时遵循“四段式”或“高密度 bullet”

默认用四段式：

- 背景：用户场景或平台定位
- 目标：要解决什么问题
- 过程：4-6 条 bullet，写架构和关键实现
- 结果：只写真实数字或明确标注“建议补充/待确认”

如果用户只要简历条目，就直接输出 4-6 条 bullet。

### 4. 量化规则必须保守

允许使用的数字只有三类：

1. 用户亲自提供的真实数据
2. 你在当前仓库重新核实过的数据
3. 明确标注为“示例/建议补充”的占位数字

禁止直接沿用旧文档中的失真数据。

## 这个项目当前可安全主张的方向

优先表述这些已被当前代码验证的事实：

- 这是一个“课程学习 Agent”平台，不只是早期的 MCP/RAG demo
- Web 主入口是 `run_server.py` + `src/server/app.py`
- Agent 主链路包含 ReAct tool loop、memory、hooks、skills、SSE streaming
- 检索链路包含 query routing、semantic cache、query enhancement、HybridSearch
- `HybridSearch` 是 dense + sparse + RRF + rerank + MMR + 去重/过滤
- 入库链路支持 PDF/PPTX/DOCX、题库解析、chunk transform、dense/sparse 双写入
- 系统同时暴露 Web Agent 与 MCP Server 两种接口

## 必须避免的说法

- 不要把 `main.py` 写成当前主运行入口
- 不要把早期规划项写成“已上线功能”
- 不要编造线上规模、团队规模、QPS、准确率提升
- 不要声称用了代码里没有真正落地的技术栈
- 不要把旧 skill、旧测试数字、旧任务数当成事实直接写进简历

## 输出格式

默认输出：

1. 简历版项目经历
2. 技术栈关键词
3. 5-8 个面试高频追问

如果用户明确说要“面试安全版”“深挖版”“防守版”，则改为输出：

1. 简历版项目经历
2. 技术栈关键词
3. 8-12 个高频深挖问题
4. 其中 3-5 个“最容易露馅”的 killer questions

如果用户要英文版，再给英文，不要省略中文事实校验。
