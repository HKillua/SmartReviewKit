# Phase 4：高级 Agent 特性 — 从"能用工具"到"像真人教练"

本阶段的目标不是加新工具，而是让 Agent 的**行为模式**发生质变。以下 4 个特性各自独立，可并行开发。

---

## 特性 A：主动式学习推荐（Proactive Push）

### 为什么做

当前 Agent 是纯被动的——用户不问就不动。但真正的学习教练会在学生上线时主动说"你有 3 天没复习子网划分了，掌握度在下降"。这把 Agent 从"问答机器"变成"主动教练"。

### 实现思路

在 `agent.py` 的 `chat()` 入口处新增一个**入口钩子 `_proactive_check()`**，在正式进入 Planner 之前执行：

1. **触发条件判断**：只在低信息量消息时触发（如"你好"、"在吗"、"hi"等），避免干扰用户的正常提问
2. **状态扫描**：异步并行查询三个数据源：
   - `knowledge_map.apply_decay()` → 找出掌握度衰减低于 0.5 的知识点
   - `error_memory.get_weak_concepts()` → 找出未消灭的高频错题
   - `student_profile` → 获取上次活跃时间，计算间隔天数
3. **信号聚合**：如果满足以下任一条件，生成推荐：
   - 有 ≥ 2 个知识点掌握度 < 0.4
   - 有未消灭的错题且距上次学习 > 3 天
   - 距上次活跃 > 7 天
4. **注入方式**：将推荐信号拼接进 System Prompt 的末尾，作为 Agent 的补充上下文：
   ```
   [Proactive Insight]
   该学生有 3 天未活跃。以下知识点掌握度显著衰减：
   - 子网划分: 0.82 → 0.43
   - 拥塞控制: 0.55 → 0.28
   建议在回复中主动提及这些薄弱点，询问学生是否需要快速复习。
   ```
5. Agent 拿到这段上下文后，自主决定如何组织开场白

### 涉及改动

| 文件 | 改动 |
|------|------|
| `agent.py` | 新增 `_proactive_check()` 方法，在 `chat()` 入口调用 |
| `knowledge_map.py` | 复用 `apply_decay()` + `get_due_nodes()`（新增） |
| 无新文件 | 纯逻辑改动 |

### 面试演示脚本

```
用户: "你好"

Agent: "你好！欢迎回来 👋 我注意到你已经 3 天没有复习了。
        你上次学的子网划分掌握度已经从 82% 衰减到了 43%，
        拥塞控制也降到了 28%。
        要不要现在花 5 分钟快速复习一下？我可以帮你出两道
        针对性练习题巩固一下。"
```

### 验收标准

- [ ] 低信息量消息触发主动推荐，正常提问不触发
- [ ] 推荐内容基于真实的知识衰减数据
- [ ] Trace 中可见 `proactive_check_triggered: true`

---

## 特性 B：Plan-and-Replan 自动回退

### 为什么做

当前 Agent 的 ReAct 是线性的：Thought → Action → Observe → 下一步。但如果第一步的工具结果很差（如知识库里查不到内容），Agent 不会主动推翻计划走替代路径。具备 Replan 能力是区分"玩具 Agent"和"工业级 Agent"的关键分水岭。

### 实现思路

不需要改架构，核心在 **AUTONOMOUS 模式的 System Prompt 增强**：

1. **在 System Prompt 中注入 Replan 策略指令**：
   ```
   ## [Autonomous Replan Policy]
   你在每次工具调用后必须评估结果质量。如果遇到以下情况，应主动调整计划：

   - 知识库检索结果 < 2 条或内容太短：切换到使用其他工具或你自身知识补充
   - concept_graph_query 返回"未找到知识点"：尝试用更宽泛的主题名重新查询
   - protocol_state_simulator 返回"不支持的协议"：降级为知识库检索 + 文字讲解
   - network_calc 参数错误：向用户确认参数后重试

   在调整计划时，输出你的推理过程，说明为什么原计划不可行以及新计划是什么。
   ```

2. **在 `_tool_loop` 中增加 Replan 事件追踪**：当 Agent 在一次 AUTONOMOUS 循环中的工具调用出现 `success=False`，在 trace metadata 里记录 `replan_triggered: true` 和 `replan_reason`

3. **安全约束**：Replan 不增加额外步数——Agent 的总步数上限仍然由 SkillPolicy 的 `max_steps` 控制。Replan 只是允许 Agent 在已有步数预算内换方向

### 涉及改动

| 文件 | 改动 |
|------|------|
| `agent.py` | `_tool_loop` 中 AUTONOMOUS 分支的 System Prompt 拼接增加 Replan Policy 文本；工具失败时在 trace 增加 `replan_triggered` 字段 |
| 无新文件 | 纯 Prompt 策略 + trace 字段 |

### 面试演示脚本

```
用户: "帮我讲讲 OSPF 的链路状态算法"

[Thought] 先检索知识库
[Action] knowledge_query("OSPF 链路状态算法")
[Observe] 只返回 1 条，内容只有一句话定义，明显不足

[Replan] 知识库证据不足，无法生成高质量讲解。
         新计划：用 protocol_state_simulator 模拟路由更新过程，
         结合我自身知识进行补充讲解
[Action] protocol_state_simulator("rip_update", ...)
[Observe] 返回完整的距离向量更新模拟过程
[Thought] 虽然模拟器只支持 RIP，但我可以用 RIP vs OSPF 的对比
         来讲解 OSPF 链路状态的优势
[Output] "知识库中 OSPF 的内容有限，我用 RIP 的模拟结果做对比来
         帮你理解：[RIP 模拟结果] → OSPF 与 RIP 的核心区别在于..."
```

### 验收标准

- [ ] 当工具返回 `success=False` 或结果极少时，Agent 能自主切换到替代工具
- [ ] Trace 中可见 `replan_triggered` 和 `replan_reason` 字段
- [ ] Replan 后的回答质量不低于直接使用备选工具的回答

---

## 特性 C：学习节奏自适应（Adaptive Pacing）

### 为什么做

当前 Agent 对所有学生一视同仁：基础好的和基础差的拿到的回答详细度一样。真正的教练会根据学生的实时表现动态调整节奏——学得快就加速，卡住了就放慢。

### 实现思路

同样不需要新工具，核心是**基于本轮对话的实时信号做 Prompt 动态调整**：

1. **信号采集**：在 `_tool_loop` 的每次迭代中，从对话历史里提取以下信号：
   - 本轮做题正确率（如果有 `quiz_evaluator` 的调用结果）
   - 用户提问的复杂度（问号数量、是否追问细节）
   - 用户回复长度（极短回复可能表示不耐烦或理解困难）

2. **节奏标签计算**：根据信号自动计算一个 pacing_level：
   - `accelerate`：连续 2 次答对 + 用户追问更深层问题
   - `normal`：默认
   - `decelerate`：连续答错 + 用户回复极短或表达困惑

3. **注入方式**：在每轮对话的 System Prompt 末尾追加节奏指令：
   ```
   ## [Adaptive Pacing]
   当前学习节奏判定: decelerate（放慢）
   依据: 学生在本轮连续 2 次答错，最近一条消息只有 3 个字
   建议: 放慢讲解速度，补充更多基础概念和例子。
         如有前置知识薄弱，主动调用 concept_graph_query 查询前置依赖。
   ```

4. **信号持久化**：将本轮的 pacing_level 写入 `student_profile.learning_pace` 字段（已有该字段，目前一直是默认 "medium"），使其在跨 Session 时也能被记忆

### 涉及改动

| 文件 | 改动 |
|------|------|
| `agent.py` | 新增 `_compute_pacing()` 方法，从当前对话历史提取信号；结果注入 System Prompt |
| `enhancer.py` | `MemoryRecordHook` 的 `after_message` 中将 pacing_level 同步到 `student_profile.learning_pace` |
| 无新文件 | 纯逻辑改动 |

### 面试演示脚本

```
--- Session 中学生连续做对两道题 ---

[Pacing Signal] 学生连续正确 2 次，切换到 accelerate 模式
[Agent 行为变化] 跳过基础概念重复，直接推进到高阶内容：
  "看来你对三次握手已经很熟了！我们直接跳到拥塞控制的
   快速恢复阶段——这是考试中最容易丢分的部分。"

--- 同一 Session 中学生第三题答错 ---

[Pacing Signal] 刚才从 accelerate 降回 normal
[Agent 行为变化] 恢复正常讲解详细度

--- 学生连续两题答错 ---

[Pacing Signal] 从 normal 降到 decelerate
[Agent 行为变化] 自动放慢：
  "没关系，这道题确实容易混淆。我们退一步，先看看滑动窗口
   的基本概念——它是理解拥塞控制的前置知识。"
[Action] concept_graph_query("拥塞控制", query_type="prerequisites")
```

### 验收标准

- [ ] 连续做对后 Agent 的回答明显更简洁、推进更快
- [ ] 连续做错后 Agent 自动放慢并补充前置知识
- [ ] `student_profile.learning_pace` 字段被持久化更新
- [ ] Trace 中可见 `pacing_level` 字段

---

## 特性 D：Agent 行为评估框架（Eval Framework）

### 为什么做

面试官一定会问"你怎么知道你的 Agent 改得更好了？"如果回答"我跑了几个 Case 看了看"，印象会直接减半。需要一套**自动化的、可量化的评测体系**。

### 实现思路

1. **测试用例集（Test Suite）**：在 `tests/eval/` 目录下定义 YAML 格式的测试用例：

   ```yaml
   # tests/eval/cases/review_tcp.yaml
   name: "个性化复习（老用户）"
   user_message: "帮我复习 TCP"
   user_id: "test_user_with_history"
   
   # 预设学生状态
   mock_knowledge_map:
     - { concept: "三次握手", mastery: 0.85 }
     - { concept: "拥塞控制", mastery: 0.2 }
     - { concept: "流量控制", mastery: 0.35 }
   
   # 期望行为断言
   expectations:
     tool_must_include: ["concept_graph_query"]
     tool_should_include: ["knowledge_query"]
     tool_must_not_include: ["document_ingest"]
     output_must_contain: ["拥塞控制"]
     output_should_not_contain: ["三次握手.*详细讲解"]  # 已掌握的不应重复讲
     min_tool_steps: 2
     max_tool_steps: 5
     control_mode: "autonomous"
   ```

2. **评估维度**（4 个指标）：

   | 指标 | 含义 | 计算方式 |
   |------|------|---------|
   | **工具召回率** | Agent 该用的工具是否都用了 | `tool_must_include` 命中率 |
   | **工具精确率** | Agent 有没有多调无关工具 | `tool_must_not_include` 违规率 |
   | **路径合理性** | 工具调用顺序是否合理 | 按 `tool_must_include` 的顺序判定 |
   | **输出覆盖度** | 回答是否覆盖了关键知识点 | `output_must_contain` 匹配率 |

3. **执行器（Eval Runner）**：`tests/eval/runner.py`
   - 加载所有 YAML 用例
   - 对每个用例：Mock 掉知识图谱和错题数据 → 调用 `agent.chat()` → 收集 trace 和最终输出
   - 逐条检查 `expectations` 中的断言
   - 输出报告：通过/失败数、各维度评分、失败用例详情

4. **核心测试用例清单（第一版 12 条）**：

   | 编号 | 场景 | 核心验证点 |
   |------|------|-----------|
   | 1 | 新用户复习 TCP | 没有历史数据时走通用路线 |
   | 2 | 老用户复习 TCP | 有掌握度数据时跳过已掌握的知识 |
   | 3 | 子网划分计算题 | 必须调用 `network_calc` |
   | 4 | CRC 校验计算题 | 必须调用 `network_calc`，不用大模型脑算 |
   | 5 | TCP 三次握手讲解 | 应调用 `protocol_state_simulator` |
   | 6 | 故障注入（SYN-ACK 丢包） | 模拟器能正确注入故障 |
   | 7 | 知识库证据不足时的 Replan | 检索结果少于 2 条时切换替代路线 |
   | 8 | 工具返回错误时的降级 | 不支持的协议类型时降级为知识检索 |
   | 9 | 纯闲聊不触发工具 | "你好"不应调用任何专业工具 |
   | 10 | 主动推荐触发 | 低信息量消息 + 知识衰减时触发 proactive push |
   | 11 | 文件上传走 FORCE_TOOL | 不应进入 AUTONOMOUS 模式 |
   | 12 | 复合任务拆解 | "先总结再出题"必须拆成两步 |

### 涉及改动

| 文件 | 说明 |
|------|------|
| `tests/eval/cases/*.yaml` | 12 个测试用例定义（新建） |
| `tests/eval/runner.py` | 评估执行器（新建） |
| `tests/eval/report.py` | 报告生成器（新建） |
| `tests/eval/conftest.py` | Mock 工厂（新建） |

### 面试演示

直接在终端跑：
```bash
python -m tests.eval.runner --cases tests/eval/cases/ --report

=== Agent Eval Report ===
总用例: 12 | 通过: 11 | 失败: 1

工具召回率:  91.7%  (11/12)
工具精确率:  100%   (0 违规)
路径合理性:  83.3%  (10/12)
输出覆盖度:  91.7%  (11/12)

❌ 失败用例:
  - Case 7 (Replan): Agent 未在知识库结果不足时切换工具
    期望: tool_path 包含 protocol_state_simulator
    实际: tool_path = ["knowledge_query"]
```

### 验收标准

- [ ] 12 条核心用例全部定义完毕
- [ ] Runner 可一键执行全部用例
- [ ] 报告包含四维评分 + 失败用例详情
- [ ] 通过率 ≥ 80%

---

## 开发顺序建议

```
Week 1: 特性 A（主动推荐）+ 特性 C（节奏自适应）
         → 两个都是 Prompt 策略 + 轻量代码，可并行
Week 2: 特性 B（Plan-and-Replan）
         → 依赖特性 A/C 的 Prompt 注入模式作为参考
Week 3: 特性 D（Eval 框架）
         → 最后做，因为需要前三个特性都就位后才能写出完整的测试用例
```
