---
name: knowledge_check
description: "知识掌握度检查：展示各概念掌握度、学习进度和复习建议"
trigger_patterns:
  - "掌握度"
  - "知识图谱"
  - "学习进度"
  - "复习建议"
  - "学习报告"
tools_required:
  - knowledge_query
memory_required:
  - student_profile
  - knowledge_map
  - error_memory
estimated_tokens: 400
difficulty: easy
---

# 知识掌握度检查技能 (Knowledge Check)

## 操作步骤

1. 从 KnowledgeMap 获取用户所有知识点的掌握度
2. 从 ErrorMemory 统计错误分布
3. 从 StudentProfile 获取总体学习数据
4. 按章节/主题分组展示掌握度
5. 标识需要复习的知识点（低掌握度 + 超过复习间隔）
6. 生成个性化学习建议

## 输出格式

```markdown
# 📊 学习进度报告

## 总体概况
- 总学习会话: X 次
- 总测验数: Y 次
- 整体正确率: Z%
- 最近活跃: ...

## 知识点掌握度

### 第 X 章: [章节名]
| 知识点 | 掌握度 | 练习次数 | 状态 |
|--------|--------|----------|------|
| TCP 三次握手 | 85% ✅ | 10 | 已掌握 |
| 子网划分 | 30% ⚠️ | 3 | 需加强 |

## 需复习的知识点
- ⚠️ 子网划分: 掌握度 30%, 上次复习 3 天前
- ⚠️ OSPF 路由: 掌握度 45%, 已超过复习间隔

## 学习建议
1. 优先复习: ...
2. 建议练习: ...
3. 推荐章节: ...
```

## 质量自检
- 掌握度数据是否准确反映实际情况？
- 复习建议是否具体可执行？
- 是否覆盖所有已学习的知识点？
