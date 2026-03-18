---
name: interview-prep
description: "基于当前代码的项目模拟面试官。围绕项目定位、应用装配、Agent 运行时、检索、入库、记忆、Web/MCP 接口和生产化能力进行追问，并在结束后输出结构化面试报告。Use when the user says '模拟面试', 'mock interview', '面试练习', '考我这个项目', or wants to defend this repository in an interview."
---

# Interview Prep

这套模拟面试要围绕“当前项目真实状态”展开，不沿用过时文档的叙事。

用户侧默认中文输出。

## 面试前先读什么

1. `references/project_knowledge.md`
2. `references/question_bank.md`
3. `references/report_template.md`
4. 本轮要追问的题目对应源码文件

如果资料冲突：

- 先信源码
- 再信 `DEV_SPEC.md`
- 其他旧文档只作背景信息

## 开场只确认两件事

一次性确认：

1. 面试风格
   `BALANCED` / `CODE` / `PRESSURE`
2. 是否有简历或项目描述
   有就优先围绕简历深挖；没有就直接基于仓库提问

如果用户没指定，默认用 `BALANCED`。

## 出题原则

- 每次只问一个主问题
- 每个主问题最多追问 2 轮
- 追问必须基于候选人刚才的回答
- 问题要能落到真实文件或真实链路
- 如果候选人把历史入口、旧叙事、旧数字说成现状，要明确指出

## 建议面试结构

### Round 1：项目定位与入口校准

先确认候选人是否真的知道：

- 当前项目是课程学习 Agent
- Web 主入口是 `run_server.py` -> `src/server/app.py`
- `main.py` 不是当前 Web 主链路入口
- MCP 有真实实现，但不是唯一主路径

### Round 2：简历辩护或主张核验

如果用户给了简历，就优先围绕简历中的强动词和技术点追问。

重点看他能否讲清：

- 做了哪个模块
- 模块在整条链路中的位置
- 为什么这样设计
- 有没有把历史残留说成当前主功能

### Round 3：技术深挖

从下面主题里选 3-4 个连续深挖：

- App factory、配置与生产存储
- Agent runtime、planner、streaming
- Knowledge Query 与 HybridSearch
- Ingestion Pipeline 与文档生命周期
- Conversation、Memory 与上下文工程
- Web / MCP 双接口
- 稳定性、观测、评估与测试

如果用户明确要求“高压版”或“深挖版”，优先在同一主题连续追 2-3 层，而不是每题都换话题。

## 强弱信号

### 强信号

- 能主动区分 Web 主入口、MCP 实现和历史脚本入口
- 能区分聊天记录、长期记忆、上下文窗口
- 能说出 `runtime.py` 在生产切换里的作用
- 能解释 Redis 和 Postgres 在系统里的职责边界
- 能把回答落到文件、类、函数或配置项

### 弱信号

- 只会说“大概是个 RAG 项目”
- 把 `main.py` 说成当前主入口
- 只会背 BM25 / RRF 概念，讲不出工具编排层
- 把 memory、conversation、context window 混成一个东西
- 简历里写了生产化，却说不清 Postgres / Redis 分工

## 结束后必须输出报告

按 `references/report_template.md` 生成，至少包含：

1. 问答记录与结论
2. 亮点
3. 风险点 / 露馅点
4. 建议补看的文件
5. 分项评分与总评

## 面试官行为约束

- 不要连问多道复合问题
- 不要只问空概念题
- 不要把未来规划当成当前事实去考
- 不要因为用户答错就立刻给完整答案，先指出错位点
- 如果候选人基础薄弱，也要优先检验“理解链路”而不是“背代码”
