---
name: quiz_drill
description: "习题训练：出题、答题、评判的交互式练习流程"
trigger_patterns:
  - "出题"
  - "做题"
  - "练习"
  - "习题"
  - "刷题"
  - "测验"
tools_required:
  - quiz_generator
  - quiz_evaluator
  - knowledge_query
memory_required:
  - error_memory
  - knowledge_map
estimated_tokens: 500
difficulty: medium
---

# 习题训练技能 (Quiz Drill)

## 操作步骤

1. 确认用户想练习的主题、题型（选择/填空/简答/分析）、数量和难度
2. 使用 `quiz_generator` 生成题目
3. 逐题展示给用户，等待用户作答
4. 用户提交答案后，使用 `quiz_evaluator` 评判并给出解析
5. 自动更新 ErrorMemory（错题）和 KnowledgeMap（掌握度）
6. 所有题目完成后，给出练习总结

## 输出格式

出题时：
```markdown
## 📋 第 X 题（选择题，难度 ⭐⭐⭐）

[题目内容]

A. ...
B. ...
C. ...
D. ...
```

评判时：
```markdown
✅ 正确！/ ❌ 错误

**正确答案**: B
**解析**: ...
**涉及知识点**: ...
```

练习总结：
```markdown
## 📊 练习总结
- 总题数: X, 正确: Y, 正确率: Z%
- 薄弱知识点: ...
- 建议: ...
```

## 质量自检
- 题目难度是否与用户要求匹配？
- 评判是否准确且有详细解析？
- 是否记录了错题到 ErrorMemory？
