# Phase 2：Skill Policy

## 背景问题

原来的 Skill 已经有学习场景意识，但本质上仍然更像“命中关键词后注入一段 SOP 文本”。  
这种方式能增强回答风格，但还不足以真正控制 Agent 的执行策略。

问题主要有两个：

- Skill 更像软提示，不是真正的运行时约束
- Planner 决定了任务类型，但没有一个稳定层来约束“这个任务允许怎么做”

## 为什么要这样改

如果要把项目从 Workflow 讲成 Agent，Skill 必须从“提示模板”升级成“策略层”。

也就是说：

- Planner 负责识别任务是什么
- SkillPolicy 负责约束这个任务允许用什么工具、最多走几步、需要什么记忆、最后要输出什么教学结果

这一步的本质不是“给 Skill 加字段”，而是把 Skill 真正变成学习策略。

## 核心思路

这次把 Skill 升级成了结构化 `SkillPolicy`，核心字段包括：

- `allowed_tools`
- `required_memory`
- `allow_autonomous`
- `max_steps`
- `entry_conditions`
- `output_contract`
- `post_actions`

同时保留了旧的 skill frontmatter 兼容映射，所以旧 Skill 不会因为升级而失效。

第一版重点改造了四个技能：

- `exam_prep`
- `chapter_deep_dive`
- `quiz_drill`
- `error_review`

## 运行时链路怎么变化了

升级后，SkillWorkflow 不再只返回一段文字说明，而是同时返回：

- 命中的 Skill
- 对应的 SkillPolicy

运行时链路变成：

用户问题 → 命中 Skill → 读取 SkillPolicy → Planner 识别任务类型 → SkillPolicy 约束工具白名单、步数和后动作

所以 Skill 的作用从“提示模型怎么说”变成了“决定这轮允许怎么做”。

## 关键 tradeoff

最大的 tradeoff 是：**SkillPolicy 的优先级要高于 Planner，但又不能完全替代 Planner**。

如果完全交给 Skill：

- 任务识别会僵硬
- 泛化能力下降

如果完全交给 Planner：

- Skill 就退化回软提示

所以最终做法是：

- Planner 识别任务类型
- SkillPolicy 约束运行边界

这是一种“识别和执行约束分层”的设计。

## 效果与验证

升级后的直接效果有三个：

- Trace 中可以看到 `skill_policy_applied`
- 工具白名单开始真正生效，不只是写在 prompt 里
- 自主模式的开关不再由 Planner 单独决定，而是由 SkillPolicy 明确授权

这让项目在面试里更容易讲成：

> Tool 是动作，Skill 是战术，Agent 是指挥层。

## 面试时怎么讲

可以直接这样讲：

> 我把 Skill 从 SOP 文本升级成了结构化 SkillPolicy。原来 Skill 更多是提示模型“怎么回答”，升级后它开始真正约束运行时行为，比如允许哪些工具、最多走几步、需要读哪些记忆、结束后要不要产生后动作。这样 Planner 负责识别任务，SkillPolicy 负责限制执行边界，整个系统就从“带模板的问答系统”升级成了“带策略层的学习 Agent”。
