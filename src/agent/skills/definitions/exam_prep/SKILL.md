---
name: exam_prep
description: "期末考试复习：按章节梳理考点、标注薄弱环节、生成复习计划"
trigger_patterns:
  - "考点复习"
  - "期末复习"
  - "考试复习"
  - "复习计划"
  - "考前准备"
tools_required:
  - knowledge_query
  - review_summary
  - quiz_generator
memory_required:
  - error_memory
  - knowledge_map
allowed_tools:
  - concept_graph_query
  - knowledge_query
  - review_summary
  - quiz_generator
  - network_calc
required_memory:
  - knowledge_map
  - error_memory
  - student_profile
allow_autonomous: true
max_steps: 5
entry_conditions:
  min_topics: 1
output_contract:
  - review_summary
  - weak_points_highlight
  - optional_quiz
post_actions:
  - schedule_export
estimated_tokens: 800
difficulty: medium
---

# 考试复习技能 (Exam Prep)

## 操作步骤

1. 使用 `knowledge_query` 检索用户指定章节或主题的核心内容
2. 使用 `review_summary` 生成该主题的结构化考点摘要
3. 查询 ErrorMemory 获取该主题下的历史错题，标注 ⚠️ 薄弱知识点
4. 查询 KnowledgeMap 获取各概念掌握度，生成个性化复习优先级
5. 可选：使用 `quiz_generator` 生成 3-5 道针对薄弱点的练习题

## 输出格式

```markdown
# 📖 [主题] 考点复习

## 核心概念
- 概念 1: 简要解释（如：TCP 三次握手的目的与流程）
- ⚠️ 概念 2: 简要解释（薄弱）

## 重要定理 / 规则
- 如：Shannon 定理、Nyquist 定理、路由收敛条件

## 易错点
- 如：TCP 与 UDP 的区别、子网掩码计算

## 与其他章节的关联
- 如：运输层与网络层的关系

## 个性化建议
- 建议重点复习: ...
- 掌握度最低的知识点: ...

## 自测题（可选）
- ...
```

## 质量自检
- 考点是否覆盖该章节核心内容？
- 薄弱知识点是否正确标注？
- 输出格式是否结构化且易读？
