# Phase 7 - Agenda ReAct Multi Goal

## 背景问题

前面的 Agent 已经能做受控 ReAct，也支持显式顺序复合任务和 skill 区间约束。  
但运行时还有一个明显问题：

当某个 tool 返回的是用户态结果时，runtime 很容易把它理解成“整轮任务已经结束”。  
这在单目标任务里问题不大，但在多目标任务里会出现过早收敛。

最典型的例子有两类：

1. `先复习，再出题`
   `review_summary` 本身已经是用户可读结果，但整轮请求其实还没完成，因为后面还要继续出题。

2. `quiz_generator -> 等用户回答 -> quiz_evaluator`
   出题结果当然可以直接给用户看，但它只表示“当前步骤完成”，不表示“整轮学习闭环已经结束”。

所以这里真正缺的是：

不是更复杂的 tool，也不是更复杂的 skill，  
而是一个更清晰的 **request 级执行状态模型**。

## 为什么要这样改

这次重构的核心思想是把“结束条件”拆成两层：

第一层是 **step-level final**  
表示当前 tool 的输出已经是用户态文本，可以直接展示。

第二层是 **request-level final**  
表示整轮请求的所有 required goals 都已经完成。

在经典单轮 ReAct 里，常见的结束条件是：

“模型这轮不再调用工具了，所以结束。”

这对单目标任务是合理的。  
但对我们这种学习型 Agent 不够，因为很多任务天然是多目标、跨步骤、甚至跨轮的。

所以这次改成：

外层用 `Agenda` 管 request 级 goals  
内层继续用 `ReAct` 管单个 goal 的收敛

也就是说：

当前 goal 不再调用工具，只表示当前 goal 完成  
只有所有 required goals 都完成，request 才算完成

## 核心思路

这次实现收敛成四个关键点。

### 1. planner 只负责产出 goal agenda

planner 继续沿用 Phase 6 的简化思路：

- 有显式顺序词时，按顺序切段
- 每段复用 `rule -> embedding -> default` 意图识别
- 保留 `skill_start_index / skill_end_index`

但 planner 的产物不再只是“给 composite runtime 用的一次性 subtasks”，  
而是正式进入 request 级 agenda。

每个 goal 至少包含：

- `goal_id`
- `intent`
- `selected_tool`
- `segment_text`
- `required`
- `depends_on_user_input`
- `status`

### 2. runtime 改成“外层 agenda，内层 per-goal ReAct”

运行时不再简单地把 composite 任务一口气跑完。  
现在是：

1. 外层顺序推进 goals
2. 每个 goal 内部仍然用现有 `_tool_loop()` 做受控 ReAct
3. `_tool_loop()` 的结束只代表当前 goal 完成、等待用户、需要澄清，或者被阻塞
4. 外层 agenda 再决定 request 是否继续

### 3. tool metadata 新增 `completion_hint`

之前只有 `tool_output_kind`，只能回答：

- 这是 `final_answer`
- 还是 `evidence_context`
- 还是 `analysis_context`

现在额外加了：

- `completion_hint = continue`
- `completion_hint = step_done`
- `completion_hint = wait_user`
- `completion_hint = clarify`

这样 runtime 能把“输出类型”和“当前 goal 状态”分开处理。

### 4. agenda_state 支持跨轮恢复

这次把 request 级状态正式放进 `Conversation.metadata`。

当 request 进入：

- `waiting_user`
- `clarification_required`

系统会把当前 `agenda_state` 持久化下来。  
下一轮用户消息进来时，先尝试恢复 agenda，而不是默认 fresh plan。

## 运行时链路怎么变化了

现在一条多目标请求的完整链路是：

### Step 1. skill workflow 先做场景级匹配

如果命中 skill，先拿到：

- `matched_skill`
- `skill_policy`

它们仍然只负责给 planner 和 runtime 提供场景级先验。

### Step 2. planner 产出 goals

如果用户说：

`先复习传输层，再出两道题`

planner 会得到两个显式 goals：

1. `review_summary`
2. `quiz_generator`

然后 agenda builder 会在尾部补一个等待型 goal：

3. `quiz_evaluator`

这个 goal 不会当前轮立刻执行，但会进入 agenda。

### Step 3. 外层 agenda 开始顺序推进

先执行 goal 1。

`review_summary` 返回的是用户态结果，所以：

- `tool_output_kind = final_answer`
- `completion_hint = step_done`

runtime 只把它理解成：

“第一个 goal 完成了”

而不是：

“整轮 request 结束了”

### Step 4. 继续执行 goal 2

`quiz_generator` 生成题目后：

- 输出本身仍然是 `final_answer`
- 但在等待型场景下会附带：
  - `completion_hint = wait_user`
  - `resume_payload.quiz_bundle`

于是 runtime 会把当前 request 状态改成：

- `request_status = waiting_user`

并把 agenda_state 保存进 conversation metadata。

### Step 5. 当前轮结束，但 request 未完成

这一轮用户会看到：

- 已完成的复习结果
- 新出的题目
- 下一步提示

但 request 不是 `completed`，而是 `waiting_user`。

### Step 6. 用户下一轮给答案时先尝试恢复 agenda

下一轮消息进来后，agent 先检查：

- 当前 conversation 里有没有 `agenda_state`
- 现在是不是在等某个 goal 的输入

如果当前 goal 是 `quiz_evaluator`，并且用户消息看起来像答案输入，就优先恢复 agenda。

恢复时：

- 不再重新从整句生成一个全新的多目标计划
- 而是从未完成的 evaluator goal 继续

### Step 7. evaluator 优先从 resume payload 取题目 bundle

这次把 Phase 5 的 batch quiz alignment 接到了 agenda 上。

恢复 evaluator goal 时：

1. 优先读 `agenda_resume_payload.quiz_bundle`
2. 如果没有，再回退到最近消息解析
3. 最后才考虑自由文本 LLM 拆题

这样跨轮判题比只靠最近 assistant 消息更稳。

### Step 8. 所有 required goals 完成后，request 才进入 completed

当 `quiz_evaluator` 判完之后：

- 当前 goal `step_done`
- 所有 required goals 都完成

这时 request 才真正进入：

- `request_status = completed`

并清理掉 `conversation.metadata["agenda_state"]`

## quiz / error_review 跨轮恢复怎么做

这次优先落地的是最典型、最值钱的一类：

`quiz_generator -> 等用户作答 -> quiz_evaluator`

这类模式既可以来自：

- `quiz_drill`
- `error_review`
- 也可以来自显式复合请求里的出题目标

核心做法有三点。

### 1. quiz_generator 写入 resume payload

在等待型场景下，quiz_generator 不只返回题目文本，还会把结构化的题目 bundle 写进：

- `resume_payload.quiz_bundle`

### 2. evaluator 恢复时优先用 quiz_bundle

下一轮如果 agenda 正在等 `quiz_evaluator`，  
direct path 构造 evaluator 参数时，会优先把 quiz bundle 带进批量对齐器。

这样不需要每次都重新从历史 assistant 文本里反解析题目。

### 3. 对齐失败时走 clarify，不再默默失败

如果用户答案无法稳定对齐，  
不会再把 request 当成“执行失败”直接吞掉，而是明确进入：

- `clarification_required`

这样后面仍然可以继续恢复，而不是让对话状态断掉。

## 这次改动的 tradeoff

这次不是把整个 Agent 变成一个通用 workflow engine，而是只补了最关键的 request 级状态。

主要 tradeoff 有三点。

### 1. 先只支持一个 active agenda

同一个 conversation 里当前只维护一个活跃 agenda。  
如果用户明显发起无关新任务，会放弃旧 agenda，不做多 agenda 并存。

这牺牲了并发复杂度，但换来了更可控的恢复逻辑。

### 2. 先只强支持 quiz 等待链

虽然 agenda 模型是通用的，  
但 v1 强支持的跨轮模式主要还是：

- `quiz_generator -> quiz_evaluator`

其他等待型 goal 先统一用 `clarify` / `blocked` 处理。

### 3. 仍然保留现有 direct path 和 tool loop

这次没有重写全部 runtime。  
Agenda-ReAct 是在现有 `_tool_loop()` 基础上改成 per-goal 语义，尽量减少回归风险。

## 效果与验证

这次重点验证了几类行为。

### 1. `final_answer + step_done` 不再提前结束 request

比如：

`review_summary` 结束后，如果 agenda 里后面还有 `quiz_generator`，runtime 会继续推进下一个 goal。

### 2. 多目标任务能进入 waiting_user

例如：

`先复习传输层，再出两道题`

当前轮会结束在：

- `request_status = waiting_user`

而不是误判成 request 已完成。

### 3. 下一轮答案能恢复到 evaluator goal

用户下一轮作答时：

- 优先恢复 agenda
- evaluator 可直接利用 `resume_payload.quiz_bundle`
- request 最终正常进入 `completed`

### 4. 无关新请求会放弃旧 agenda

如果当前还在等题目答案，  
但用户下一轮改成了一个明显无关的新请求，比如文档导入，  
旧 agenda 会被放弃，再对新消息 fresh plan。

## 面试时怎么讲

这次改动最适合从“结束条件设计”来讲。

可以直接这样表述：

我们前面的 Agent 已经能做受控 ReAct，但在复杂任务里还有一个问题：运行时会把某个 tool 的用户态输出误当成整轮请求的完成信号。比如复习总结和出题都能直接给用户看，但在“先复习、再出题、再等用户作答判题”这种任务里，它们其实只是步骤级完成，不是 request 级完成。所以这次我把运行时改成了外层 Agenda、内层 per-goal ReAct 的结构。planner 继续只负责产出 goals，tool 继续只声明输出是 `final_answer` 还是 `evidence_context`，但 request 是否结束改由 agenda state 决定。这样 `final_answer` 只表示当前步骤已经是用户态结果，不再等价于整轮结束。我们还把 `quiz_generator -> quiz_evaluator` 做成了强支持的跨轮恢复链，出题后会把结构化题目 bundle 写进 resume payload，用户下一轮作答时优先恢复 evaluator goal，而不是重新 fresh plan。这个改动的核心价值，是把复杂任务从“tool 驱动收敛”升级成“request agenda 驱动收敛”。  
