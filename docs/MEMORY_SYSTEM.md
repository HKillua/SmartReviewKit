# 记忆系统深度升级 — 个性化复习助手

> 阶段 K 实现文档 | 借鉴阿里 CoPaw ReMe 框架理念

---

## 1. 设计理念

将 Agent 从"无状态问答机器人"升级为"有长期记忆的个性化复习助手"。

### 对标 CoPaw ReMe

| CoPaw ReMe 特性 | 我们的实现 | 说明 |
|---|---|---|
| MEMORY.md 长期记忆 | StudentProfile + KnowledgeMap | SQLite 结构化存储，支持快速查询 |
| memory/YYYY-MM-DD.md 每日日志 | **SessionMemory** | 每次会话存储结构化摘要 |
| Compaction 上下文压缩 | **ContextEngineeringFilter Level 3** | LLM 摘要压缩旧消息 |
| 混合检索 (Vector + BM25) | HybridSearch (已有) | 0.7/0.3 权重融合 |
| 效用驱动淘汰 | KnowledgeMap Ebbinghaus 衰减 | 遗忘曲线自动降低掌握度 |

### 核心升级

1. **从被动到主动** — Agent 主动推荐复习，不再等待用户提问
2. **从单次到连续** — 跨会话记忆，记住用户学过什么、错过什么
3. **从通用到个性化** — 记住偏好（简洁/详细/考点版），调整回答风格

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    会话开始 (before_message)                       │
│  ReviewScheduleHook                                              │
│    → Ebbinghaus 衰减 → 到期检查 → 错题提醒 → 上次话题 → 注入推荐  │
└──────────────────────────────────┬──────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    会话进行 (Agent ReAct Loop)                     │
│  ContextEngineeringFilter                                        │
│    Level 1: 滑动窗口 (>40条消息)                                   │
│    Level 2: 工具结果卸载 (>2000字符)                                │
│    Level 3: LLM 压缩 (>30条消息, CoPaw Compaction)                 │
│                                                                   │
│  MemoryContextEnhancer → system prompt 注入                       │
│    - 学习偏好 / 上次学习 / 需复习知识点 / 错题提醒                    │
└──────────────────────────────────┬──────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    会话结束 (after_message)                        │
│  MemoryRecordHook                                                │
│    → LLM/Rule 双模式提取学习数据                                    │
│    → SessionMemory.save_session()                                │
│    → StudentProfile.update_profile() (含 weak/strong/accuracy)    │
│    → SkillMemory.save_usage()                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 新增/修改文件清单

### 新增文件

| 文件 | 说明 |
|---|---|
| `src/agent/memory/session_memory.py` | SessionMemory + SessionSummary — 会话摘要存储 |
| `src/agent/hooks/review_schedule.py` | ReviewScheduleHook — 主动复习推荐 |
| `docs/MEMORY_SYSTEM.md` | 本文档 |

### 修改文件

| 文件 | 修改内容 |
|---|---|
| `src/agent/memory/enhancer.py` | K2: MemoryRecordHook 增强 (LLM/Rule提取), K6: get_memory_summary 结构化输出 |
| `src/agent/memory/student_profile.py` | K3: preferences 默认结构化 |
| `src/agent/memory/knowledge_map.py` | 新增 `_get_all_nodes()`, 供 profile 同步使用 |
| `src/agent/memory/context_filter.py` | K7: Level 3 LLM 压缩 (CoPaw Compaction) |
| `src/agent/agent.py` | 接入 context_filter + review_hook, `_build_llm_messages` 改为 async |
| `src/agent/config.py` | MemoryConfig 新增 session/extraction/compaction 字段 |
| `src/server/app.py` | 初始化新组件并注入 Agent |
| `config/settings.yaml` | 新增 memory 配置项 |
| `DEV_SPEC.md` | 添加阶段 K 详细规格 |

---

## 4. 关键组件详解

### 4.1 SessionMemory (K1)

**文件**: `src/agent/memory/session_memory.py`

```python
class SessionSummary(BaseModel):
    session_id: str          # 关联 Conversation ID
    user_id: str
    timestamp: datetime
    topics: list[str]        # ["TCP三次握手", "拥塞控制"]
    key_questions: list[str] # ["TCP为什么需要三次握手？"]
    mastery_observations: dict[str, str]  # {"TCP": "weak", "IP": "strong"}
    preference_snapshot: dict             # {"detail_level": "concise"}
    summary_text: str        # "主要复习了TCP运输层协议"

class SessionMemory:
    save_session(user_id, summary)
    get_recent_sessions(user_id, limit=5)
    get_topic_history(user_id, topic)      # 按话题搜索历史
    search_sessions(user_id, query)        # 关键词搜索
```

**面试要点**:
- 借鉴 CoPaw 每日日志理念，但用 SQLite 替代 Markdown 文件
- 支持结构化查询（按话题、时间），不仅仅是全文搜索

### 4.2 MemoryRecordHook 增强 (K2)

**双模式提取**:

```
会话结束
   │
   ├─ extraction_mode == "llm"
   │   └─ 发给 LLM → 返回结构化 JSON (话题/薄弱点/偏好/关键问题)
   │
   ├─ extraction_mode == "rule"
   │   └─ 正则匹配: TCP/UDP/IP 等协议名 + "简洁点"/"详细" 偏好检测
   │
   └─ extraction_mode == "both"  ← 推荐
       └─ 先 LLM，失败回退 Rule
```

**分发更新**:
- → SessionMemory.save_session() — 本次会话摘要
- → StudentProfile.update_profile() — 偏好、弱点、总会话数(累加)
- → KnowledgeMap — 同步 weak/strong topics

**修复**: `total_sessions` 从覆写 (`= 1`) 改为累加 (`= profile.total_sessions + 1`)

### 4.3 ReviewScheduleHook (K4)

**文件**: `src/agent/hooks/review_schedule.py`

每次新会话开始时自动执行:

```
1. Ebbinghaus 衰减 (apply_decay) — 带冷却时间，12h 内不重复
2. 到期复习检查 (get_due_for_review) — 间隔天数已过
3. 未掌握错题 (get_errors, mastered=False)
4. 上次话题延续 (get_recent_sessions)
```

**输出示例**:
```
### 主动复习建议
- 知识点「TCP三次握手」掌握度 30%，已5天未复习，建议今天复习
- 错题提醒：「TCP协议」— TCP为什么需要三次握手…
- 上次学习了「TCP三次握手、拥塞控制」
  其中「TCP三次握手」掌握较弱，建议继续巩固
```

### 4.4 偏好学习 (K3)

**StudentProfile.preferences 结构**:
```python
{
    "detail_level": "concise",     # concise / normal / detailed
    "style": "exam_focused",       # default / exam_focused / example_heavy
    "quiz_difficulty": "medium",   # easy / medium / hard
}
```

**检测规则**:
| 用户输入 | 检测偏好 |
|---|---|
| "简洁点"、"简短" | detail_level: concise |
| "详细讲"、"展开说" | detail_level: detailed |
| "考点"、"重点"、"应试" | style: exam_focused |
| "举个例子" | style: example_heavy |

### 4.5 上下文压缩 (K7)

借鉴 CoPaw Compaction 机制:

```
消息数 > compaction_threshold (30)
   │
   ├─ 旧消息 → LLM 压缩为一条 [对话历史摘要]
   ├─ 最近 10 条消息 → 保留原文
   └─ 压缩内容: 学习目标 + 已解决问题 + 关键发现 + 下一步
```

- LLM 失败时回退到 Level 1 滑动窗口
- 压缩结果带缓存，相同内容不重复调用 LLM

### 4.6 记忆注入优化 (K6)

**system prompt 中注入格式**:

```markdown
## 学生记忆上下文

### 学习偏好
- 回答风格: 简洁
- 内容偏好: 考点版

### 上次学习
- 话题: TCP三次握手, 拥塞控制
- 关键问题: TCP为什么需要三次握手？
- 薄弱: TCP三次握手
- 掌握: IP协议

### 需要复习的知识点
- TCP三次握手 (掌握度: 30%, 5天未复习)

### 错题提醒
- TCP协议: TCP为什么需要三次握手… (错1次)

### 主动复习建议
- 知识点「TCP三次握手」掌握度 30%，建议今天复习
```

---

## 5. 配置项

```yaml
memory:
  enabled: true
  db_dir: "data/memory"
  session_memory_enabled: true         # K1: 会话摘要
  extraction_mode: "both"              # K2: llm / rule / both
  review_schedule_enabled: true        # K4: 主动复习
  decay_on_session_start: true         # K4: 新会话触发衰减
  compaction_enabled: true             # K7: 上下文压缩
  compaction_threshold_messages: 30    # K7: 压缩阈值
```

---

## 6. 面试亮点叙事

### 问：你的 Agent 是怎么做记忆的？

> 我们参考了阿里最新开源的 CoPaw 项目的 ReMe (Remember Me, Refine Me) 记忆框架，
> 实现了一套完整的记忆生命周期管理：
>
> 1. **提取** — 每次会话结束后，通过 LLM 或规则从对话中提取学习数据（话题、薄弱点、偏好）
> 2. **存储** — 四类长期记忆（学生档案、错题本、知识图谱、会话摘要）+ SQLite 持久化
> 3. **衰减** — Ebbinghaus 遗忘曲线自动降低掌握度，模拟真实记忆衰退
> 4. **召回** — 新会话开始时主动检查到期复习、未掌握错题，生成复习建议
> 5. **注入** — 结构化记忆上下文注入 system prompt，引导 Agent 个性化回答
> 6. **压缩** — 借鉴 CoPaw Compaction，长对话自动 LLM 摘要压缩，保留要点

### 问：主动复习是怎么实现的？

> 用了一个 `before_message` LifecycleHook，在新会话第一条消息时触发：
> 先 `apply_decay()` 执行 Ebbinghaus 衰减，
> 再查 `get_due_for_review()` 找到期知识点，
> 加上 `get_errors(mastered=False)` 的错题，
> 和 `get_recent_sessions()` 的上次话题延续，
> 合成一段"主动复习建议"注入到 system prompt 中。
> Agent 看到后会自然地说"你上次 TCP 掌握不好，要不要复习一下"。

### 问：偏好记忆是怎么工作的？

> 两种模式可切换：
> - **Rule**: 正则匹配"简洁点""详细讲""考点"等关键词
> - **LLM**: 把对话发给 LLM 提取结构化偏好 JSON
> - **Both**: LLM 优先，失败回退 Rule
>
> 提取的偏好存入 `StudentProfile.preferences`，下次注入 system prompt 时
> 会包含"回答风格: 简洁考点版"，Agent 就会自动调整回答风格。

---

## 7. 数据流图

```
用户发消息
    │
    ▼
ReviewScheduleHook.before_message()
    ├── apply_decay() → 衰减知识点
    ├── get_due_for_review() → 到期复习
    ├── get_errors(mastered=False) → 错题
    └── get_recent_sessions() → 上次话题
         │
         ▼ review_context
MemoryContextEnhancer.get_memory_summary()
    ├── StudentProfile → 偏好/统计
    ├── SessionMemory → 上次学习
    ├── KnowledgeMap → 需复习/薄弱
    └── ErrorMemory → 错题提醒
         │
         ▼ memory_context + review_context
SystemPromptBuilder.build()
    → 完整 system prompt
         │
         ▼
Agent._build_llm_messages()
    → ContextEngineeringFilter
       Level 2: 卸载大型工具结果
       Level 3: LLM 压缩旧消息 (if > 30)
       Level 1: 滑动窗口 (if > 40)
         │
         ▼
LLM Call → Tool Loop → Response
         │
         ▼
MemoryRecordHook.after_message()
    ├── LLM/Rule 提取学习数据
    ├── SessionMemory.save_session()
    ├── StudentProfile.update_profile()
    │   ├── total_sessions += 1
    │   ├── preferences 更新
    │   ├── weak_topics 同步
    │   ├── strong_topics 同步
    │   └── overall_accuracy 计算
    └── SkillMemory.save_usage()
```
