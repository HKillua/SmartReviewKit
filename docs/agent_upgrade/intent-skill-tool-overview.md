# Intent / Skill / Tool 总结

## 一句话总览
这个项目里，`skill`、`intent`、`tool` 是三层不同的概念。

`skill` 负责定义学习场景，`intent` 负责定义当前这一步是什么业务动作，`tool` 负责真正执行动作。  
如果只记一句话，可以记成：

> Skill 是场景层，Intent 是任务层，Tool 是执行层。

---

## 一、三者分别是什么

### 1. Skill
Skill 表示“当前用户处在哪种学习场景里”。  
它解决的问题不是“这一句具体要不要出题”，而是“这轮对话更像考前复习、章节精讲、刷题训练还是错题复盘”。

Skill 的主要作用有三点：

1. 给系统一个高层学习场景标签  
2. 提供结构化策略约束，比如允许哪些工具、最多几步、是否允许自主  
3. 决定回答结束后是否触发后动作，比如导出笔记、闪卡、复习计划

当前项目里的 skill 一共有 5 个：

| Skill 名称 | 场景含义 | 核心作用 |
| --- | --- | --- |
| `exam_prep` | 考前复习 | 按章节梳理考点、标注薄弱点、可选出题 |
| `chapter_deep_dive` | 章节深入学习 | 做结构化讲解、概念辨析、过程演示 |
| `quiz_drill` | 习题训练 | 出题、答题、评判、练习总结 |
| `error_review` | 错题复盘 | 看历史错题、分析错因、做补救练习 |
| `knowledge_check` | 知识掌握度检查 | 看知识图谱、学习进度、复习建议 |

这些 skill 的定义来自：

- [/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/skills/definitions/exam_prep/SKILL.md](/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/skills/definitions/exam_prep/SKILL.md)
- [/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/skills/definitions/chapter_deep_dive/SKILL.md](/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/skills/definitions/chapter_deep_dive/SKILL.md)
- [/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/skills/definitions/quiz_drill/SKILL.md](/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/skills/definitions/quiz_drill/SKILL.md)
- [/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/skills/definitions/error_review/SKILL.md](/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/skills/definitions/error_review/SKILL.md)
- [/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/skills/definitions/knowledge_check/SKILL.md](/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/skills/definitions/knowledge_check/SKILL.md)

### 2. Intent
Intent 表示“当前这一步到底是什么业务动作”。  
它解决的是任务路由问题，也就是：

- 这是普通聊天吗？
- 这是知识问答吗？
- 这是复习总结吗？
- 这是出题吗？
- 这是判题吗？
- 这是文档入库吗？

当前项目里，planner 真正建模的 intent 只有 6 个：

| Intent 名称 | 含义 | planner 默认理解 |
| --- | --- | --- |
| `general_chat` | 通用聊天 | 不强推工具，直接正常回复 |
| `knowledge_query` | 知识问答 | 优先考虑知识检索，但不强制 |
| `review_summary` | 复习总结 | 先产出结构化复习摘要 |
| `quiz_generator` | 出题训练 | 先调用出题工具 |
| `quiz_evaluator` | 判题评估 | 先调用判题工具 |
| `document_ingest` | 文档入库 | 先调用文档导入工具 |

这些 intent 定义在：

- [/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/planner/task_planner.py](/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/agent/planner/task_planner.py)

### 3. Tool
Tool 是真正执行动作的能力模块。  
它不是“抽象的任务类型”，而是系统运行时会实际调用的工具。

当前运行时真实注册的 tool 一共有 8 个：

| Tool 名称 | 作用 | 更常出现在哪类场景 |
| --- | --- | --- |
| `knowledge_query` | 检索知识库并生成知识问答依据 | 普通知识问答、章节讲解、复习 |
| `document_ingest` | 把 PDF/PPTX/DOCX 等资料导入知识库 | 资料上传与入库 |
| `review_summary` | 生成结构化复习摘要 | 考前复习、章节总结 |
| `quiz_generator` | 生成题目 | 习题训练、考前练习 |
| `quiz_evaluator` | 判题、解析、记录错题 | 刷题训练、错题复盘 |
| `network_calc` | 做网络计算题 | 子网划分、CRC、吞吐/时延等 |
| `concept_graph_query` | 查知识图谱、依赖关系、掌握度 | 复习规划、章节深入 |
| `protocol_state_simulator` | 做协议过程模拟 | TCP 状态机、拥塞控制、故障注入 |

这些 tool 注册在：

- [/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/server/app.py](/Users/dongjinshuo/projects/work/rag_agent/MODULAR-RAG-MCP-SERVER/src/server/app.py)

---

## 二、为什么 skill、intent、tool 不能混为一谈

### 1. Skill 不是 Tool
Skill 不是真正执行动作的模块。  
Skill 更像一个“学习场景策略包”。

例如，`exam_prep` 并不是一个工具。  
它不会像 `knowledge_query` 那样被直接执行。  
它做的是告诉系统：

- 这是考前复习场景
- 允许用哪些工具
- 最多走几步
- 回答最好产出什么

### 2. Intent 不是 Skill
Intent 不是学习场景，而是当前这一步的具体动作。

例如：

- “先复习传输层，再出两道题”

这里整个请求可能命中 `exam_prep` 这个 skill，  
但里面会拆成两个 intent：

1. `review_summary`
2. `quiz_generator`

所以：

- `exam_prep` 是整轮的学习场景
- `review_summary` 和 `quiz_generator` 是其中两步的任务动作

### 3. Tool 也不是 Intent
Intent 是抽象任务类别，Tool 是执行入口。

当前很多情况下，两者名字刚好一样，比如：

- `knowledge_query -> knowledge_query`
- `quiz_generator -> quiz_generator`
- `review_summary -> review_summary`

但系统设计上它们依然要分开，因为：

1. planner 先识别业务动作  
2. runtime 再决定真正执行哪个工具  
3. skill policy 可能在后面修正某一步优先调用的 tool

所以 `selected_tool` 的存在，就是为了把“任务是什么”和“具体用什么执行”分开。

---

## 三、当前项目里，三者是怎么配合的

### 阶段 1：先看有没有命中 Skill
系统先做一次 skill workflow 匹配，检查当前这句话像不像某种学习场景。

如果命中，会得到：

- `matched_skill`
- `skill_policy`

例如：

- “帮我期末复习传输层”  
  很可能命中 `exam_prep`

### 阶段 2：再做 Intent 识别
然后 planner 再判断：

- 当前这句话具体属于哪种业务动作
- 如果是顺序复合任务，要不要拆成多个 subtasks
- 每个 subtask 各自是什么 intent

例如：

“先讲一下 TCP 和 UDP 的区别，再帮我考前复习传输层，再出两道题”

可能会拆成：

| 顺序 | subtask | intent |
| --- | --- | --- |
| 1 | 讲 TCP 和 UDP 的区别 | `knowledge_query` |
| 2 | 考前复习传输层 | `review_summary` |
| 3 | 出两道题 | `quiz_generator` |

### 阶段 3：最后执行 Tool
runtime 再根据每个 subtask 的 `selected_tool` 去调用真正的工具。

如果 skill 生效，还会用 `skill_policy` 约束：

- 允许哪些 tools
- 能不能 autonomous
- 最多几步
- 回答结束后要不要做 post actions

---

## 四、当前项目的真实任务集合与工具集合

### 1. 任务集合比工具集合更小
这是当前架构里一个很重要的特征。

planner 的任务集合只有 6 个，但运行时工具有 8 个。  
原因是：

- planner 负责高层动作路由
- tool 负责具体能力执行

并不是每个 tool 都值得上升成 planner 的一级 intent。

### 2. 哪些 tool 是 planner 的一等公民
直接和 intent 一一对应的，主要是下面 5 个：

| Intent | 常用 selected_tool |
| --- | --- |
| `knowledge_query` | `knowledge_query` |
| `review_summary` | `review_summary` |
| `quiz_generator` | `quiz_generator` |
| `quiz_evaluator` | `quiz_evaluator` |
| `document_ingest` | `document_ingest` |

### 3. 哪些 tool 更像专业辅助工具
下面 3 个工具更常通过 skill/autonomous/runtime 后续选择进入执行，而不是直接由 planner 把整句分类成这个 intent：

| Tool | 更像什么 |
| --- | --- |
| `network_calc` | 专业计算器 |
| `concept_graph_query` | 学习结构与掌握度分析器 |
| `protocol_state_simulator` | 协议过程模拟器 |

这也是为什么项目里“tool 数量 > intent 数量”是合理的。

---

## 五、当前每个 Skill 更偏向哪些 Intent 和 Tool

### `exam_prep`
这是“考前复习”场景。  
它更偏向的 intent 是：

- `review_summary`
- `quiz_generator`
- 某些情况下也会带 `knowledge_query`

它允许的 tool 主要是：

| Tool | 作用 |
| --- | --- |
| `concept_graph_query` | 看掌握度与复习顺序 |
| `knowledge_query` | 查课程证据 |
| `review_summary` | 生成复习摘要 |
| `quiz_generator` | 补自测题 |
| `network_calc` | 做计算类复习题 |

### `chapter_deep_dive`
这是“章节深入讲解”场景。  
它更偏向的 intent 是：

- `knowledge_query`
- 某些情况下也会走 `review_summary`

它允许的 tool 主要是：

| Tool | 作用 |
| --- | --- |
| `concept_graph_query` | 先查前置依赖与主题结构 |
| `knowledge_query` | 检索章节知识 |
| `protocol_state_simulator` | 做过程模拟 |
| `network_calc` | 对计算题或定量过程做精确解释 |

### `quiz_drill`
这是“刷题训练”场景。  
它更偏向的 intent 是：

- `quiz_generator`
- `quiz_evaluator`

它允许的 tool 主要是：

| Tool | 作用 |
| --- | --- |
| `quiz_generator` | 出题 |
| `quiz_evaluator` | 判题 |
| `knowledge_query` | 做题后补知识解释 |
| `network_calc` | 校验计算题结果 |

### `error_review`
这是“错题复盘”场景。  
它更偏向的 intent 是：

- `knowledge_query`
- `quiz_generator`

它允许的 tool 主要是：

| Tool | 作用 |
| --- | --- |
| `concept_graph_query` | 看错误相关知识点与依赖 |
| `knowledge_query` | 检索针对性解释 |
| `quiz_generator` | 生成补救练习 |
| `protocol_state_simulator` | 对过程型知识点做重讲 |

### `knowledge_check`
这是“掌握度检查”场景。  
它更偏向：

- `knowledge_query`

当前更偏向 advisory，不是高自由度 autonomous 场景。

---

## 六、当前 planner 为什么不直接把 8 个 Tool 都建成 8 个 Intent

原因很简单：planner 负责的是“高层动作路由”，不是“所有底层能力枚举”。

如果把每个 tool 都强行抬成一级 intent，会带来三个问题：

第一，planner 会变得更重。  
它本来只需要判断用户是在问知识、复习、出题、判题还是入库，如果再让它直接区分所有专业工具，复杂度会明显上升。

第二，很多专业工具不是直接由用户语义触发的。  
例如 `network_calc`、`concept_graph_query`、`protocol_state_simulator`，很多时候是系统在 skill 场景或 autonomous 过程中自主选择的，不是用户每次都会明确说“请调用概念图谱工具”。

第三，这样会削弱 skill 的价值。  
skill 本来就是用来约束“在某个学习场景下允许哪些工具”。如果 planner 直接把所有 tools 都当一等 intent，skill 层的场景控制意义就会被冲淡。

所以当前更合理的架构是：

- planner 只建模少量高价值任务
- runtime 再在 skill policy 约束下选择更细的专业工具

---

## 七、最适合面试时直接复述的版本

这个项目里我把 `skill`、`intent`、`tool` 明确拆成了三层。`skill` 表示学习场景，比如考前复习、章节精讲、刷题训练、错题复盘；`intent` 表示当前这一步是什么业务动作，比如知识问答、复习总结、出题、判题、文档入库；`tool` 则是真正执行动作的能力模块，比如 `knowledge_query`、`quiz_generator`、`quiz_evaluator`、`network_calc`、`concept_graph_query` 和 `protocol_state_simulator`。这样拆的好处是，skill 负责高层场景策略，intent 负责任务路由，tool 负责执行落地。planner 不需要直接理解所有专业工具，只需要先判断当前任务属于哪类动作，后面再由 skill policy 和 runtime 决定具体调用哪些工具。  
