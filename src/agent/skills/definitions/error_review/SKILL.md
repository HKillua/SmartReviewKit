---
name: error_review
description: "错题回顾：整理历史错题、分析错因、进行针对性练习"
trigger_patterns:
  - "错题回顾"
  - "错题本"
  - "错题复习"
  - "我的错题"
  - "错误分析"
tools_required:
  - knowledge_query
  - quiz_generator
  - quiz_evaluator
memory_required:
  - error_memory
  - knowledge_map
estimated_tokens: 600
difficulty: medium
---

# 错题回顾技能 (Error Review)

## 操作步骤

1. 从 ErrorMemory 获取用户的未掌握错题列表
2. 按知识点分类统计错误频率
3. 对高频错误知识点，使用 `knowledge_query` 检索相关解释
4. 展示错题详情：题目、用户答案、正确答案、解析
5. 可选：使用 `quiz_generator` 为错误知识点生成类似题目

## 输出格式

```markdown
# 📝 错题回顾报告

## 错题统计
- 未掌握错题总数: X 道
- 高频错误知识点: A, B, C

## 错题详情

### 错题 1
- **题目**: ...
- **你的答案**: ...
- **正确答案**: ...
- **解析**: ...
- **相关知识点**: ...

## 针对性练习
- 以下题目针对你的薄弱点...
```

## 质量自检
- 是否展示了所有未掌握的错题？
- 错因分析是否准确？
- 针对性练习是否覆盖薄弱知识点？
