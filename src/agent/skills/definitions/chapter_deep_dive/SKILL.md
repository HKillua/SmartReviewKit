---
name: chapter_deep_dive
description: "章节深入学习：对某一章节进行详细讲解、概念辨析和例题演示"
trigger_patterns:
  - "章节详解"
  - "深入讲解"
  - "详细解释"
  - "帮我理解"
tools_required:
  - knowledge_query
  - review_summary
memory_required:
  - knowledge_map
allowed_tools:
  - concept_graph_query
  - knowledge_query
  - protocol_state_simulator
  - network_calc
required_memory:
  - knowledge_map
allow_autonomous: true
max_steps: 4
entry_conditions:
  min_topics: 1
output_contract:
  - structured_explanation
  - optional_simulation
  - mind_map
post_actions:
  - notes_export
estimated_tokens: 700
difficulty: hard
---

# 章节深入学习技能 (Chapter Deep Dive)

## 操作步骤

1. 使用 `knowledge_query` 全面检索该章节的知识库内容（top_k=10）
2. 使用 `review_summary` 生成该章节的完整考点摘要
3. 对核心概念进行详细解释，配合例子说明
4. 进行易混淆概念辨析（如 TCP vs UDP、距离向量 vs 链路状态、电路交换 vs 分组交换）
5. 给出典型例题和解题思路

## 输出格式

```markdown
# 📚 [章节名] 详细讲解

## 章节概述
- 本章主要内容: ...
- 与前后章节的关系: ...

## 核心概念详解

### 概念 1: [名称]
- **定义**: ...
- **举例**: ...
- **常见误区**: ...

## 概念辨析
| 概念 A | 概念 B |
|--------|--------|
| TCP（面向连接、可靠传输） | UDP（无连接、尽力交付） |
| 距离向量路由 | 链路状态路由 |

## 典型例题
- **题目**: ...
- **解题思路**: ...
- **答案**: ...
```

## 质量自检
- 概念解释是否通俗易懂？
- 是否包含具体例子？
- 概念辨析是否覆盖常见混淆点？
