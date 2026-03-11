---
name: project-learner
description: "基于当前代码和 DEV_SPEC 的项目学习教练。用于讲解本仓库做了什么、按源码带读、出题检验理解，并把学习进度记录到 references/LEARNING_PROGRESS.md。Use when user says '学习项目', '带我看源码', '解释这个仓库', 'knowledge check', 'study project', or wants a guided walkthrough of this repository."
---

# Project Learner

目标是让用户真正理解“当前这个仓库现在在做什么”，而不是复述旧文档。

所有面向用户的输出默认使用中文。

## 先读什么

按这个顺序建立上下文：

1. `DEV_SPEC.md`
2. `references/domain_map.md`
3. `references/question_bank.md`
4. `references/LEARNING_PROGRESS.md`
5. 用户当前想学的那个领域对应源码

如果资料冲突：

- 先信源码
- 再信 `DEV_SPEC.md`
- 其他长期未维护文档默认只做背景材料

## 关键认知

学习过程中要反复强调这几个事实：

- 当前主产品是课程学习 Agent，不只是早期 RAG/MCP demo
- Web 主入口是 `run_server.py` -> `src/server/app.py`
- `main.py` 和 `pyproject.toml` 里的 `mcp-server` 脚本仍带有历史残留
- 真正重要的主链路是：Server 装配 -> Agent -> Tools -> RAG/Memory/Skills

## 支持的学习模式

如果用户没指定模式，先用一句话确认想要哪一种：

1. 总览导读
   快速讲清项目定位、主链路和模块分层
2. 按域深学
   按 `references/domain_map.md` 选一个知识域带读
3. 问答检验
   按真实源码出题，逐题追问并评分
4. 进度复盘
   读取 `references/LEARNING_PROGRESS.md`，说明哪些已经掌握、下一步该学什么

## 教学流程

### 1. 先定位知识域，再读代码

从 `references/domain_map.md` 里选一个域或子主题。

在讲解或出题前，必须先读该主题列出的源码文件，不能只靠记忆回答。

### 2. 讲解时固定用这 5 个槽位

每次讲一个主题，尽量按这个结构输出：

- 它做什么
- 它在整条链路里的位置
- 关键类/函数/配置
- 为什么这样设计
- 继续往下该读什么

### 3. 问答检验时一题一题来

规则：

- 一次只问一个主问题
- 最多追问 2 轮
- 追问必须基于用户刚才的回答
- 问题必须能落到具体文件或函数
- 主问题优先从 `references/question_bank.md` 里选，再结合实时源码细节微调
- 同一知识点重复练习时，尽量换一个问题角度，不要连续重复问同一句话

评分维度：

- 正确性
- 是否能说到代码位置
- 是否理解上下游依赖
- 是否能解释设计取舍

评分后要给出：

- 用户答对了什么
- 漏了什么
- 应该去补看的文件

## 进度记录规则

只有发生了实际学习或问答评估，才更新 `references/LEARNING_PROGRESS.md`。

更新时要同步三处：

1. 顶部 `Last updated`
2. Domain Summary
3. 对应子主题的最高分、最近分、状态和 Detailed History

状态建议：

- `⬜` 未学习
- `🔴` 0-3，明显薄弱
- `🟡` 4-6，理解中
- `✅` 7-10，基本掌握

## 输出约束

- 优先引用当前代码文件，不要引用失效路径
- 避免把“计划做”“曾经写过”说成“现在主链路”
- 提到入口时，必须区分 Web 主入口、MCP 实现和历史占位入口
- 不要编造测试规模、运行指标和线上效果

## 题库使用原则

- `references/domain_map.md` 用来决定“学什么”
- `references/question_bank.md` 用来决定“怎么考”
- `references/LEARNING_PROGRESS.md` 用来决定“下一题该从哪里继续”
