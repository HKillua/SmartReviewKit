# Developer Specification (DEV_SPEC)

# Course Learning Agent — 课程智能学习助手

> 基于 RAG + Memory + Skill 的 ReAct Agent，参考 Vanna.ai v2.0 架构  
> 系统设计详见：[docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)  
> 当前默认课程：计算机网络（通用架构，支持多课程切换）

---

## 1. 项目概述

### 1.1 项目定位

本项目在现有 MODULAR-RAG-MCP-SERVER 的 RAG 基础设施之上，构建一个面向课程学习的智能 Agent。Agent 具备考点复习、习题生成与评判、智能问答、动态知识库管理四大核心能力，并通过 Memory 记忆系统和 Skill 技能系统实现个性化学习体验。RAG 层采用结构化解析、多策略分块、Parent-Child 层级索引、双向量数据库（ChromaDB + Milvus）等面试级深度设计。

### 1.2 设计理念

| 原则 | 说明 |
|------|------|
| **Pluggable** | 所有外部依赖（LLM、Embedding、VectorStore）通过抽象接口 + 工厂模式注入 |
| **Config-Driven** | 行为参数集中在 `config/settings.yaml`，运行时不修改代码 |
| **Reuse-First** | 最大化复用现有 RAG 基础设施，Agent 是上层编排 |
| **Observable** | 每个关键路径有 Trace span，工具调用有审计日志 |
| **Graceful-Degradation** | LLM 不可用时返回纯检索结果，组件故障不阻塞核心流程 |
| **Memory-Aware** | 短期上下文工程 + 长期四类记忆，Agent 有状态、可个性化 |
| **Skill-Driven** | 复杂任务通过预定义 Skill 编排，渐进式披露控制 Token 成本 |

### 1.3 核心技术栈

| 组件 | 选型 | 版本 |
|------|------|------|
| 语言 | Python | >= 3.10 |
| Web 框架 | FastAPI + Uvicorn | latest |
| LLM SDK | openai (兼容多家 API) | >= 1.0 |
| 向量存储 | ChromaDB + Milvus Lite（双 Provider） | >= 0.4.0 / >= 2.4.0 |
| PPT 解析 | python-pptx | latest |
| PDF 解析 | MarkItDown + PyMuPDF | latest |
| Word 解析 | python-docx | latest |
| 数学公式 | lxml (OMML→LaTeX) | latest |
| 前端公式渲染 | KaTeX | >= 0.16 |
| 数据校验 | Pydantic | >= 2.0 |
| SSE | sse-starlette | latest |
| 测试 | pytest + pytest-asyncio | latest |

---

## 2. 系统架构与模块设计

### 2.1 架构分层

```
Layer 0  [Web Frontend]        → 只依赖 Server HTTP API
Layer 1  [FastAPI Server]      → 只依赖 Agent Core
Layer 2  [Agent Core]          → 依赖 LlmService + ToolRegistry + ConversationStore + Memory + Skill
Layer 3  [Agent Tools]         → 依赖 RAG Infrastructure + LlmService + Memory
Layer 4  [RAG Infrastructure]  → 复用现有（HybridSearch, IngestionPipeline, ChromaDB, BM25）
Layer 5  [Libs]                → 复用现有 + 新增 PptxLoader
```

**依赖规则**：上层可依赖下层，下层不可依赖上层。同层通过抽象接口交互。

### 2.2 目录结构

```
MODULAR-RAG-MCP-SERVER/
├── config/
│   ├── settings.yaml                  # 主配置（扩展 agent 段）
│   └── prompts/                       # Prompt 模板
│       └── system_prompt.txt
│
├── src/
│   ├── agent/                         # ===== 新增：Agent 核心层 =====
│   │   ├── __init__.py
│   │   ├── agent.py                   # [D3] Agent 主类
│   │   ├── config.py                  # [A3] AgentConfig
│   │   ├── types.py                   # [A2] Agent 层类型定义
│   │   ├── conversation.py            # [D1] ConversationStore
│   │   ├── prompt_builder.py          # [D2] SystemPromptBuilder
│   │   │
│   │   ├── llm/                       # ===== LLM 服务层 =====
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # [B1] LlmService ABC
│   │   │   ├── openai_service.py      # [B2] OpenAI / Azure OpenAI
│   │   │   ├── deepseek_service.py    # [B3] DeepSeek
│   │   │   ├── ollama_service.py      # [B3] Ollama
│   │   │   └── factory.py             # [B4] LlmServiceFactory
│   │   │
│   │   ├── tools/                     # ===== 工具层 =====
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # [C1] Tool ABC + ToolRegistry
│   │   │   ├── knowledge_query.py     # [E1] KnowledgeQueryTool
│   │   │   ├── review_summary.py      # [E3] ReviewSummaryTool
│   │   │   ├── quiz_generator.py      # [E4] QuizGeneratorTool
│   │   │   ├── quiz_evaluator.py      # [E4] QuizEvaluatorTool
│   │   │   └── document_ingest.py     # [E2] DocumentIngestTool
│   │   │
│   │   ├── hooks/                     # ===== 扩展点 =====
│   │   │   ├── __init__.py
│   │   │   ├── lifecycle.py           # [C2] LifecycleHook ABC
│   │   │   ├── middleware.py          # [C3] LlmMiddleware ABC
│   │   │   ├── rate_limit.py          # [I1] RateLimitHook
│   │   │   └── retry_middleware.py    # [I1] RetryMiddleware
│   │   │
│   │   ├── memory/                    # ===== Memory 系统 =====
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # [F1] Memory 抽象接口
│   │   │   ├── context_filter.py      # [F1] ContextEngineeringFilter
│   │   │   ├── student_profile.py     # [F2] StudentProfileMemory
│   │   │   ├── error_memory.py        # [F2] ErrorMemory
│   │   │   ├── knowledge_map.py       # [F2] KnowledgeMapMemory
│   │   │   ├── skill_memory.py        # [F2] SkillMemory
│   │   │   └── enhancer.py            # [F3] LlmContextEnhancer + MemoryRecordHook
│   │   │
│   │   └── skills/                    # ===== Skill 系统 =====
│   │       ├── __init__.py
│   │       ├── registry.py            # [G1] SkillRegistry
│   │       ├── workflow.py            # [G2] SkillWorkflowHandler
│   │       └── definitions/           # [G3] 预定义 Skill
│   │           ├── exam_prep/
│   │           │   └── SKILL.md
│   │           ├── error_review/
│   │           │   └── SKILL.md
│   │           ├── chapter_deep_dive/
│   │           │   └── SKILL.md
│   │           ├── quiz_drill/
│   │           │   └── SKILL.md
│   │           └── knowledge_check/
│   │               └── SKILL.md
│   │
│   ├── server/                        # ===== 新增：FastAPI 服务 =====
│   │   ├── __init__.py
│   │   ├── app.py                     # [H1] FastAPI 应用
│   │   ├── routes.py                  # [H1] API 路由
│   │   ├── chat_handler.py            # [H1] ChatHandler
│   │   └── models.py                  # [H1] 请求/响应 Pydantic 模型
│   │
│   ├── web/                           # ===== 新增：前端 =====
│   │   ├── index.html                 # [H3] 主页面
│   │   ├── style.css                  # [H3] 样式
│   │   └── app.js                     # [H3] SSE 客户端
│   │
│   ├── libs/loader/
│   │   ├── pptx_loader.py            # [E2] PPT Loader → [J2] OMML 公式增强
│   │   ├── pdf_loader.py             # 复用 → [J3] 公式后处理 + 结构检测
│   │   ├── docx_loader.py            # [J4] 新增：Word Loader
│   │   └── math_utils.py             # [J1] 新增：公式工具模块
│   │
│   ├── libs/vector_store/
│   │   ├── milvus_store.py            # [J11] 新增：Milvus Lite 存储
│   │   └── ...                        # 其他复用现有
│   │
│   ├── ingestion/chunking/
│   │   ├── semantic_splitter.py       # [J5] 新增：语义分块
│   │   ├── structure_splitter.py      # [J6] 新增：结构感知分块
│   │   └── ...                        # 其他复用现有
│   │
│   ├── core/                          # 复用现有 → [J1] types.py 增加 DocumentSection
│   ├── ingestion/                     # 复用现有 → [J17] Pipeline 更新
│   ├── libs/                          # 复用现有
│   ├── mcp_server/                    # 复用现有 → [J13] 工具解耦
│   └── observability/                 # 复用现有
│
├── tests/
│   ├── unit/
│   │   ├── test_agent_types.py        # [A2]
│   │   ├── test_agent_config.py       # [A3]
│   │   ├── test_llm_service.py        # [B1-B4]
│   │   ├── test_tool_base.py          # [C1]
│   │   ├── test_tool_registry.py      # [C1]
│   │   ├── test_lifecycle_hooks.py    # [C2]
│   │   ├── test_middleware.py         # [C3]
│   │   ├── test_conversation.py       # [D1]
│   │   ├── test_prompt_builder.py     # [D2]
│   │   ├── test_agent.py             # [D3]
│   │   ├── test_knowledge_query.py    # [E1]
│   │   ├── test_pptx_loader.py        # [E2]
│   │   ├── test_review_summary.py     # [E3]
│   │   ├── test_quiz_tools.py         # [E4]
│   │   ├── test_context_filter.py     # [F1]
│   │   ├── test_memory_stores.py      # [F2]
│   │   ├── test_memory_enhancer.py    # [F3]
│   │   ├── test_skill_registry.py     # [G1]
│   │   ├── test_skill_workflow.py     # [G2]
│   │   ├── test_math_utils.py         # [J1]
│   │   ├── test_docx_loader.py        # [J4]
│   │   ├── test_semantic_splitter.py   # [J5]
│   │   ├── test_structure_splitter.py  # [J6]
│   │   ├── test_parent_child.py        # [J8]
│   │   ├── test_milvus_store.py        # [J11]
│   │   └── test_metadata_enricher.py   # [J14]
│   │
│   ├── integration/
│   │   ├── test_agent_tool_loop.py    # [D3] Agent + Tools 集成
│   │   ├── test_agent_memory.py       # [F3] Agent + Memory 集成
│   │   ├── test_agent_skill.py        # [G2] Agent + Skill 集成
│   │   └── test_server_sse.py         # [H1] FastAPI SSE 集成
│   │
│   ├── e2e/
│   │   ├── test_chat_flow.py          # 完整对话流
│   │   ├── test_upload_flow.py        # 文件上传流
│   │   └── test_quiz_flow.py          # 习题练习流
│   │
│   └── fixtures/
│       ├── sample.pptx                # PPT 测试文件
│       ├── sample_chunks.json         # Mock 检索结果
│       └── mock_llm_responses.json    # Mock LLM 响应
```

### 2.3 核心数据类型（`src/agent/types.py`）

```python
# --- 流式事件 ---
class StreamEvent(BaseModel):
    type: str           # "text_delta" | "tool_start" | "tool_result" | "error" | "done"
    content: Optional[str]
    tool_name: Optional[str]
    metadata: dict

# --- LLM 层 ---
class LlmMessage(BaseModel):
    role: str           # "system" | "user" | "assistant" | "tool"
    content: Optional[str]
    tool_calls: Optional[list[ToolCallData]]
    tool_call_id: Optional[str]

class LlmRequest(BaseModel):
    messages: list[LlmMessage]
    tools: Optional[list[dict]]
    temperature: float
    max_tokens: Optional[int]
    stream: bool
    metadata: dict

class LlmResponse(BaseModel):
    content: Optional[str]
    tool_calls: Optional[list[ToolCallData]]
    usage: Optional[dict]
    error: Optional[str]

class LlmStreamChunk(BaseModel):
    delta_content: Optional[str]
    delta_tool_calls: Optional[list]
    finish_reason: Optional[str]

# --- 工具层 ---
class ToolCallData(BaseModel):
    id: str
    name: str
    arguments: dict

class ToolContext(BaseModel):
    user_id: str
    conversation_id: str
    request_id: str
    metadata: dict

class ToolResult(BaseModel):
    success: bool
    result_for_llm: str
    error: Optional[str]
    metadata: dict

# --- 会话层 ---
class Message(BaseModel):
    role: str
    content: str
    timestamp: datetime
    tool_calls: Optional[list[ToolCallData]]
    tool_call_id: Optional[str]

class Conversation(BaseModel):
    id: str
    user_id: str
    messages: list[Message]
    created_at: datetime
    updated_at: datetime
```

---

## 3. 测试方案

### 3.1 测试分层

| 层级 | 目录 | 目标 | 外部依赖 |
|------|------|------|---------|
| Unit | `tests/unit/` | 单模块逻辑正确性 | 全部 Mock |
| Integration | `tests/integration/` | 模块间协作 | 允许本地 DB / 文件系统 |
| E2E | `tests/e2e/` | 完整用户场景 | 需要 LLM API (标记 `@pytest.mark.llm`) |

### 3.2 Mock 策略

| 被 Mock 对象 | Mock 方式 |
|-------------|----------|
| LlmService | `MockLlmService`：返回预设 `LlmResponse`（含 tool_calls） |
| HybridSearch | 返回固定 `RetrievalResult` 列表 |
| IngestionPipeline | 返回固定 `PipelineResult` |
| ConversationStore | `MemoryConversationStore`：内存字典 |
| Memory Stores | 内存字典实现 |

### 3.3 Pytest 标记

```python
markers = [
    "unit: 单元测试 (快速, 无外部依赖)",
    "integration: 集成测试 (本地 DB/文件)",
    "e2e: 端到端测试 (完整流程)",
    "llm: 需要真实 LLM API",
    "slow: 慢速测试",
]
```

---

## 4. 项目排期

### 4.1 阶段总览

| 阶段 | 名称 | 任务数 | 预估工时 | 依赖 |
|------|------|--------|---------|------|
| A | 工程骨架与核心类型 | 3 | 0.5 天 | 无 |
| B | LLM 服务层 | 4 | 1 天 | A |
| C | Tool 基础设施与扩展点 | 3 | 0.5 天 | A |
| D | Agent 核心 | 3 | 1.5 天 | B, C |
| E | RAG 工具集 | 4 | 2 天 | D |
| F | Memory 记忆系统 | 3 | 2 天 | D, E |
| G | Skill 技能系统 | 3 | 1.5 天 | D, E, F |
| H | Server 与前端 | 3 | 1.5 天 | D |
| I | 稳定性与可观测 | 3 | 1 天 | D |
| J | RAG 深度优化 | 19 | 4 天 | A-I |
| K | 记忆系统深度升级（个性化复习助手） | 9 | 3 天 | D, F, J |
| L | 前端 UI 优化 + 会话管理完善 | 7 | 1.5 天 | D, H |
| M | RAG 深度优化 II — 面试无死角 | 16 | 3 天 | A-L |

### 4.2 进度跟踪

#### 阶段 A：工程骨架与核心类型

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| A1 | 初始化 Agent 目录结构 | [ ] | - | |
| A2 | Agent 核心类型定义 | [ ] | - | |
| A3 | Agent 配置扩展 | [ ] | - | |

#### 阶段 B：LLM 服务层

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| B1 | LlmService 抽象基类 | [ ] | - | |
| B2 | OpenAI LlmService 实现 | [ ] | - | |
| B3 | DeepSeek + Ollama LlmService | [ ] | - | |
| B4 | LlmServiceFactory | [ ] | - | |

#### 阶段 C：Tool 基础设施与扩展点

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| C1 | Tool 基类 + ToolRegistry | [ ] | - | |
| C2 | LifecycleHook 基类 | [ ] | - | |
| C3 | LlmMiddleware 基类 | [ ] | - | |

#### 阶段 D：Agent 核心

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| D1 | ConversationStore | [ ] | - | |
| D2 | SystemPromptBuilder | [ ] | - | |
| D3 | Agent 主类（Tool Loop + Streaming） | [ ] | - | |

#### 阶段 E：RAG 工具集

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| E1 | KnowledgeQueryTool | [ ] | - | |
| E2 | PptxLoader + DocumentIngestTool | [ ] | - | |
| E3 | ReviewSummaryTool | [ ] | - | |
| E4 | QuizGeneratorTool + QuizEvaluatorTool | [ ] | - | |

#### 阶段 F：Memory 记忆系统

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| F1 | 短期记忆：上下文工程 | [ ] | - | |
| F2 | 长期记忆：四类 Memory Store | [ ] | - | |
| F3 | Memory 与 Agent 集成 | [ ] | - | |

#### 阶段 G：Skill 技能系统

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| G1 | SkillRegistry + 渐进式加载 | [ ] | - | |
| G2 | SkillWorkflowHandler 集成 | [ ] | - | |
| G3 | 预定义 Skill 编写 | [ ] | - | |

#### 阶段 H：Server 与前端

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| H1 | FastAPI + SSE + ChatHandler | [ ] | - | |
| H2 | 文件上传端点 | [ ] | - | |
| H3 | Web UI 前端 | [ ] | - | |

#### 阶段 I：稳定性与可观测

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| I1 | 限流 + 重试 + 熔断 | [ ] | - | |
| I2 | 审计日志 | [ ] | - | |
| I3 | 可观测性（Metrics + Trace） | [ ] | - | |

#### 阶段 J：RAG 深度优化

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| J1 | DocumentSection 数据类 + 公式工具模块 | [ ] | - | |
| J2 | PptxLoader 增强：OMML 公式提取 + 结构化 sections | [ ] | - | |
| J3 | PdfLoader 增强：公式后处理 + 章节检测 + 习题区域识别 | [ ] | - | |
| J4 | DocxLoader 新建：Word 文档解析 + OMML 公式提取 | [ ] | - | |
| J5 | SemanticSplitter：基于 embedding 余弦相似度检测语义断点 | [ ] | - | |
| J6 | StructureAwareSplitter：按文档结构优先切分 + 内部语义细分 | [ ] | - | |
| J7 | SplitterFactory 注册 semantic 和 structure provider | [ ] | - | |
| J8 | DocumentChunker 生成 parent-child 层级 chunk | [ ] | - | |
| J9 | ChromaStore HNSW 参数调优（M/ef_construction/search_ef） | [ ] | - | |
| J10 | ChromaStore 多 collection 管理 + score_threshold + 复杂 metadata filter | [ ] | - | |
| J11 | MilvusStore 实现（Milvus Lite 嵌入式模式） | [ ] | - | |
| J12 | MilvusStore 注册 + settings.yaml 配置 | [ ] | - | |
| J13 | MCP 工具解耦（消除 chromadb 直接依赖） | [ ] | - | |
| J14 | MetadataEnricher 增强：content_type/has_formula/chapter/difficulty 标注 | [ ] | - | |
| J15 | 前端 KaTeX 集成：数学公式渲染 + dark mode 适配 | [ ] | - | |
| J16 | knowledge_query 工具支持 parent 回溯检索 + metadata filter | [ ] | - | |
| J17 | pipeline 和 settings.yaml 更新支持新分块策略和 .docx | [ ] | - | |
| J18 | 重新入库课件并端到端测试检索效果 | [ ] | - | |
| J19 | 系统设计文档补充向量存储设计决策说明 | [ ] | - | |

#### 阶段 K：记忆系统深度升级（个性化复习助手）

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| K1 | SessionMemory 会话摘要存储 | [x] | 2026-03-04 | 借鉴 CoPaw ReMe 每日日志理念 |
| K2 | MemoryRecordHook 增强：LLM/Rule 双模式学习数据提取 | [x] | 2026-03-04 | |
| K3 | StudentProfile preferences 结构化 + 偏好提取 | [x] | 2026-03-04 | |
| K4 | ReviewScheduleHook 主动复习推荐 | [x] | 2026-03-04 | Ebbinghaus 衰减调度 |
| K5 | StudentProfile 自动同步 weak/strong/accuracy | [x] | 2026-03-04 | 修复 total_sessions 覆写 bug |
| K6 | get_memory_summary 注入优化 | [x] | 2026-03-04 | 更结构化、更个性化 |
| K7 | ContextEngineeringFilter 接入 Agent + Level 3 LLM 压缩 | [x] | 2026-03-04 | 借鉴 CoPaw Compaction |
| K8 | 服务端接线 + settings.yaml 更新 | [x] | 2026-03-04 | |
| K9 | 端到端测试 | [x] | 2026-03-04 | 多轮对话验证记忆持久化 |

#### 阶段 L：前端 UI 优化 + 会话管理完善

| 任务编号 | 任务名称 | 状态 | 完成日期 | 备注 |
|---------|---------|------|---------|------|
| L1 | Conversation 模型加 title 字段 | [x] | 2026-03-04 | |
| L2 | 新增 list/delete 会话 API 端点 | [x] | 2026-03-04 | ConversationStore.delete() |
| L3 | Agent 自动生成会话标题 + done metadata | [x] | 2026-03-04 | 首条消息截取前30字 |
| L4 | HTML 重构为 ChatGPT 两栏布局 | [x] | 2026-03-04 | 侧边栏 + 主聊天区 |
| L5 | CSS 全面重写 ChatGPT 风格 | [x] | 2026-03-04 | 深色侧边栏 + light/dark 主题 |
| L6 | JS 侧边栏会话管理 + 新建/切换/删除 | [x] | 2026-03-04 | 移动端折叠支持 |
| L7 | DEV_SPEC.md + 复习文档 | [x] | 2026-03-04 | docs/FRONTEND_UI.md |

#### 总体进度

| 阶段 | 总任务 | 已完成 | 进度 |
|------|--------|--------|------|
| A 工程骨架 | 3 | 0 | 0% |
| B LLM 服务 | 4 | 0 | 0% |
| C Tool 基础 | 3 | 0 | 0% |
| D Agent 核心 | 3 | 0 | 0% |
| E RAG 工具 | 4 | 0 | 0% |
| F Memory | 3 | 0 | 0% |
| G Skill | 3 | 0 | 0% |
| H Server | 3 | 0 | 0% |
| I 稳定性 | 3 | 0 | 0% |
| J RAG 深度优化 | 19 | 0 | 0% |
| K 记忆系统升级 | 9 | 9 | 100% |
| L 前端 UI 优化 | 7 | 7 | 100% |
| **总计** | **64** | **16** | **25%** |

---

## 5. 各阶段详细任务

---

### 阶段 A：工程骨架与核心类型

---

### A1：初始化 Agent 目录结构

- **目标**：创建 2.2 节所述的 Agent 层目录骨架与空 `__init__.py`，确保所有包可导入。
- **修改文件**：
  - `src/agent/__init__.py`
  - `src/agent/llm/__init__.py`
  - `src/agent/tools/__init__.py`
  - `src/agent/hooks/__init__.py`
  - `src/agent/memory/__init__.py`
  - `src/agent/skills/__init__.py`
  - `src/server/__init__.py`
  - `src/web/` (空目录)
  - `tests/unit/`, `tests/integration/`, `tests/e2e/`, `tests/fixtures/`
- **实现类/函数**：无（仅骨架）
- **验收标准**：
  - 目录结构与 2.2 节一致
  - `python -c "import src.agent; import src.agent.llm; import src.agent.tools"` 成功
- **测试方法**：`python -m compileall src/agent`

---

### A2：Agent 核心类型定义

- **目标**：定义 Agent 层所有核心数据类型（2.3 节），使用 Pydantic BaseModel，为后续模块提供类型契约。
- **修改文件**：
  - `src/agent/types.py`
- **实现类/函数**：
  - `StreamEvent`：流式事件（type, content, tool_name, metadata）
  - `LlmMessage`：LLM 消息（role, content, tool_calls, tool_call_id）
  - `LlmRequest` / `LlmResponse` / `LlmStreamChunk`：LLM 请求响应
  - `ToolCallData`：工具调用指令（id, name, arguments）
  - `ToolContext`：工具执行上下文（user_id, conversation_id, request_id, metadata）
  - `ToolResult`：工具执行结果（success, result_for_llm, error, metadata）
  - `Message` / `Conversation`：会话消息与会话
- **验收标准**：
  - 所有类型可正常实例化和序列化（`.model_dump_json()`）
  - 类型间引用关系正确（如 `LlmMessage.tool_calls` 引用 `ToolCallData`）
  - 字段校验生效（如 `StreamEvent.type` 限定枚举值）
- **测试方法**：`pytest -q tests/unit/test_agent_types.py`
  - 测试每个类型的构造、序列化、反序列化
  - 测试字段校验（无效值应抛 ValidationError）

---

### A3：Agent 配置扩展

- **目标**：在 `settings.yaml` 中增加 `agent` 配置段，定义 `AgentConfig` Pydantic 模型，从配置文件加载。
- **修改文件**：
  - `config/settings.yaml`（新增 `agent:` 段）
  - `src/agent/config.py`
- **实现类/函数**：
  - `AgentConfig(BaseModel)`：
    - `max_tool_iterations: int = 10`
    - `stream_responses: bool = True`
    - `tool_timeout: int = 30`
    - `max_context_messages: int = 40`
    - `max_context_tokens: int = 8000`
    - `temperature: float = 0.7`
    - `max_tokens: Optional[int] = None`
    - `system_prompt_path: str = "config/prompts/system_prompt.txt"`
    - `skills_dir: str = "src/agent/skills/definitions"`
    - `memory_enabled: bool = True`
    - `conversation_store_dir: str = "data/conversations"`
  - `load_agent_config(settings) -> AgentConfig`
- **验收标准**：
  - `settings.yaml` 中 `agent:` 段可被正确解析
  - 缺省值生效
  - 无效配置抛出明确错误
- **测试方法**：`pytest -q tests/unit/test_agent_config.py`
  - 测试默认值加载
  - 测试自定义值覆盖
  - 测试无效值校验

---

### 阶段 B：LLM 服务层

---

### B1：LlmService 抽象基类

- **目标**：定义支持 Tool Calling + Streaming 的 LLM 服务抽象接口。这是对现有 `BaseLLM`（仅支持简单 chat）的增强，专为 Agent 场景设计。
- **修改文件**：
  - `src/agent/llm/base.py`
- **实现类/函数**：
  - `LlmService(ABC)`：
    - `async def send_request(self, request: LlmRequest) -> LlmResponse`
    - `async def stream_request(self, request: LlmRequest) -> AsyncGenerator[LlmStreamChunk, None]`
    - `def validate_tools(self, tools: list[dict]) -> list[str]`（校验 tool schema 兼容性）
- **验收标准**：
  - 抽象方法无法直接实例化
  - 类型签名与 2.3 节类型定义匹配
  - `validate_tools` 能检测无效 schema 并返回错误列表
- **测试方法**：`pytest -q tests/unit/test_llm_service.py::TestLlmServiceABC`
  - 测试抽象类不可实例化
  - 测试具体子类必须实现所有方法

---

### B2：OpenAI LlmService 实现

- **目标**：实现基于 OpenAI SDK 的 LlmService，支持 Tool Calling（function calling）和流式输出。兼容 OpenAI 和 Azure OpenAI。
- **修改文件**：
  - `src/agent/llm/openai_service.py`
- **实现类/函数**：
  - `OpenAILlmService(LlmService)`：
    - 构造函数：从 settings 读取 `api_key`, `model`, `base_url`（Azure 时为 endpoint）
    - `send_request()`：调用 `client.chat.completions.create()` with `tools` 参数
    - `stream_request()`：调用 `client.chat.completions.create(stream=True)`，逐块 yield `LlmStreamChunk`
    - 将 `tool_calls` 从 OpenAI 格式转为 `ToolCallData`
    - 支持 Azure：通过 `AzureOpenAI` client
- **验收标准**：
  - 非流式调用返回正确的 `LlmResponse`（含 content 或 tool_calls）
  - 流式调用 yield 多个 `LlmStreamChunk`，累积后与非流式一致
  - Azure 认证参数正确传递
  - API 错误（429/500/timeout）封装为 `LlmResponse(error=...)`
- **测试方法**：`pytest -q tests/unit/test_llm_service.py::TestOpenAILlmService`
  - Mock `openai.AsyncOpenAI`
  - 测试非流式 tool_calls 解析
  - 测试流式 chunk 累积
  - 测试 API 错误处理
  - 标记 `@pytest.mark.llm` 的真实 API 测试（可选）

---

### B3：DeepSeek + Ollama LlmService

- **目标**：实现 DeepSeek 和 Ollama 的 LlmService。两者均兼容 OpenAI API 格式，可复用 B2 的核心逻辑。
- **修改文件**：
  - `src/agent/llm/deepseek_service.py`
  - `src/agent/llm/ollama_service.py`
- **实现类/函数**：
  - `DeepSeekLlmService(OpenAILlmService)`：覆盖 `base_url` 为 DeepSeek endpoint
  - `OllamaLlmService(LlmService)`：
    - 使用 `openai.AsyncOpenAI(base_url="http://localhost:11434/v1")`
    - Ollama 的 tool calling 支持有限，需做兼容处理
- **验收标准**：
  - DeepSeek 继承 OpenAI 实现，仅覆盖连接配置
  - Ollama 在 tool calling 不可用时 graceful fallback
- **测试方法**：`pytest -q tests/unit/test_llm_service.py::TestDeepSeekLlmService`
  - Mock OpenAI client，验证 base_url 正确
  - 测试 Ollama fallback 逻辑

---

### B4：LlmServiceFactory

- **目标**：工厂模式根据 `settings.yaml` 的 `llm.provider` 创建对应 LlmService 实例。
- **修改文件**：
  - `src/agent/llm/factory.py`
- **实现类/函数**：
  - `LlmServiceFactory`：
    - `@staticmethod create(settings: Settings) -> LlmService`
    - 支持 provider: `"openai"`, `"azure"`, `"deepseek"`, `"ollama"`
    - 未知 provider 抛 `ValueError`
- **验收标准**：
  - 每种 provider 正确创建对应 Service 实例
  - 缺少必要配置（如 api_key）时抛明确错误
- **测试方法**：`pytest -q tests/unit/test_llm_service.py::TestLlmServiceFactory`
  - 测试每种 provider 的创建
  - 测试无效 provider 报错
  - 测试缺少配置报错

---

### 阶段 C：Tool 基础设施与扩展点

---

### C1：Tool 基类 + ToolRegistry

- **目标**：实现参考 Vanna.ai 的 `Tool[T]` 泛型基类和 `ToolRegistry` 工具注册中心。工具通过 Pydantic model 定义参数 schema，Registry 负责注册、校验、分发执行。
- **修改文件**：
  - `src/agent/tools/base.py`
- **实现类/函数**：
  - `Tool(ABC, Generic[T])`：
    - `name: str` (property, abstract)
    - `description: str` (property, abstract)
    - `get_args_schema() -> Type[T]` (abstract)
    - `async execute(context: ToolContext, args: T) -> ToolResult` (abstract)
    - `get_schema() -> dict`：从 Pydantic `model_json_schema()` 自动生成 JSON Schema
  - `ToolRegistry`：
    - `register(tool: Tool) -> None`
    - `get_tool(name: str) -> Tool`
    - `get_all_schemas() -> list[dict]`：返回所有工具的 JSON Schema（给 LLM 的 `tools` 参数）
    - `async execute(tool_call: ToolCallData, context: ToolContext) -> ToolResult`：
      1. 查找工具
      2. Pydantic 校验参数
      3. 调用 `tool.execute(context, args)`
      4. 异常处理 → `ToolResult(success=False, error=...)`
- **验收标准**：
  - 可定义具体 Tool 子类并注册到 Registry
  - `get_all_schemas()` 输出符合 OpenAI function calling 格式
  - 参数校验失败返回 `ToolResult(success=False)`
  - 执行异常不会抛出，而是封装为错误 ToolResult
- **测试方法**：`pytest -q tests/unit/test_tool_base.py tests/unit/test_tool_registry.py`
  - 创建 MockTool，测试注册、查找、Schema 生成
  - 测试参数校验（有效/无效）
  - 测试执行成功和异常路径
  - 测试重复注册报错

---

### C2：LifecycleHook 基类

- **目标**：定义请求生命周期钩子接口，支持在消息处理和工具执行的前后插入自定义逻辑。
- **修改文件**：
  - `src/agent/hooks/lifecycle.py`
- **实现类/函数**：
  - `LifecycleHook(ABC)`：
    - `async before_message(user_id: str, message: str) -> Optional[str]`：返回修改后的消息或 None
    - `async after_message(conversation: Conversation) -> None`
    - `async before_tool(tool_name: str, context: ToolContext) -> None`：可抛异常阻止执行
    - `async after_tool(tool_name: str, result: ToolResult) -> Optional[ToolResult]`：返回修改后的结果或 None
- **验收标准**：
  - 接口方法签名完整
  - 默认实现为 pass-through（不修改）
  - 子类可选择性覆盖
- **测试方法**：`pytest -q tests/unit/test_lifecycle_hooks.py`
  - 测试默认 Hook 不修改数据
  - 测试 before_tool 抛异常时阻止执行
  - 测试 after_tool 返回修改后的 result

---

### C3：LlmMiddleware 基类

- **目标**：定义 LLM 请求/响应中间件接口，支持在 LLM 调用前后插入逻辑（日志、缓存、重试、计费）。
- **修改文件**：
  - `src/agent/hooks/middleware.py`
- **实现类/函数**：
  - `LlmMiddleware(ABC)`：
    - `async before_llm_request(request: LlmRequest) -> LlmRequest`
    - `async after_llm_response(request: LlmRequest, response: LlmResponse) -> LlmResponse`
- **验收标准**：
  - 接口方法签名完整
  - 默认实现为 pass-through
  - 可链式应用多个 Middleware
- **测试方法**：`pytest -q tests/unit/test_middleware.py`
  - 测试 Middleware 链式执行顺序
  - 测试 request 修改在后续 Middleware 可见
  - 测试 Middleware 异常处理

---

### 阶段 D：Agent 核心

---

### D1：ConversationStore

- **目标**：实现会话持久化存储，支持创建、加载、更新、列出会话。使用文件系统 JSON 存储，按 user_id 隔离。
- **修改文件**：
  - `src/agent/conversation.py`
- **实现类/函数**：
  - `ConversationStore(ABC)`：
    - `async create(user_id: str) -> Conversation`
    - `async get(conversation_id: str, user_id: str) -> Optional[Conversation]`
    - `async update(conversation: Conversation) -> None`
    - `async list_conversations(user_id: str, limit: int = 20) -> list[Conversation]`
  - `FileConversationStore(ConversationStore)`：
    - 存储路径：`{base_dir}/{user_id_hash[:12]}/{conversation_id}.json`
    - JSON 序列化/反序列化
  - `MemoryConversationStore(ConversationStore)`：
    - 内存字典实现（测试用）
- **验收标准**：
  - 会话创建后可通过 ID 加载
  - 用户只能访问自己的会话（user_id 校验）
  - 并发安全（文件写入使用临时文件 + rename）
  - JSON 文件可直接阅读（调试友好）
- **测试方法**：`pytest -q tests/unit/test_conversation.py`
  - 测试 CRUD 全流程
  - 测试用户隔离（A 不能访问 B 的会话）
  - 测试不存在的会话返回 None
  - 测试 MemoryConversationStore 行为一致

---

### D2：SystemPromptBuilder

- **目标**：构建 Agent 的系统提示，包含角色设定、课程上下文、工具描述、输出格式约束。支持动态注入 Memory 和 Skill 上下文。
- **修改文件**：
  - `src/agent/prompt_builder.py`
  - `config/prompts/system_prompt.txt`（模板文件）
- **实现类/函数**：
  - `SystemPromptBuilder`：
    - `build(tool_schemas: list[dict], memory_context: str = "", active_skill: str = "") -> str`
    - 从 `system_prompt.txt` 加载基础模板
    - 注入 `{tool_descriptions}`, `{memory_context}`, `{active_skill}` 占位符
- **验收标准**：
  - 默认 prompt 包含数据库课程专家角色设定
  - 工具描述正确注入
  - Memory 和 Skill 上下文可选注入
  - prompt 总长度可控（不超过 max_context_tokens 的 30%）
- **测试方法**：`pytest -q tests/unit/test_prompt_builder.py`
  - 测试默认 prompt 生成
  - 测试工具描述注入
  - 测试 Memory 上下文注入
  - 测试 Skill SOP 注入

---

### D3：Agent 主类（Tool Loop + Streaming）

- **目标**：实现 Agent 核心类，包含 ReAct 工具循环、流式输出、会话管理、扩展点调用。这是整个系统的核心编排器。
- **修改文件**：
  - `src/agent/agent.py`
- **实现类/函数**：
  - `Agent`：
    - 构造函数注入：`llm_service`, `tool_registry`, `conversation_store`, `config`, `prompt_builder`, `lifecycle_hooks`, `llm_middlewares`, `conversation_filters`
    - `async chat(message: str, user_id: str, conversation_id: Optional[str]) -> AsyncGenerator[StreamEvent, None]`：
      1. 加载/创建 Conversation
      2. 执行 `before_message` hooks
      3. 添加 user message
      4. 构建 SystemPrompt
      5. 获取 ToolSchemas
      6. **Tool Loop**（`while iterations < max_tool_iterations`）：
         - 执行 Middlewares `before_llm_request`
         - 调用 `llm_service.send_request()` 或 `stream_request()`
         - 执行 Middlewares `after_llm_response`
         - 如果有 `tool_calls`：
           - yield `StreamEvent(type="tool_start")`
           - 执行 `before_tool` hooks
           - `tool_registry.execute()` with timeout
           - 执行 `after_tool` hooks
           - yield `StreamEvent(type="tool_result")`
           - 将结果添加到消息历史，继续循环
         - 否则：yield `StreamEvent(type="text_delta", content=...)`，break
      7. yield `StreamEvent(type="done")`
      8. 保存 Conversation
      9. 执行 `after_message` hooks
- **验收标准**：
  - 无 tool_calls 时直接返回文本
  - 有 tool_calls 时执行工具并继续循环
  - 达到 max_tool_iterations 时停止并返回警告
  - 工具超时返回错误 ToolResult（不中断循环）
  - LifecycleHook 和 Middleware 按正确顺序执行
  - 异常不泄露，封装为 StreamEvent(type="error")
  - 会话正确保存（含 tool_calls 消息）
- **测试方法**：
  - `pytest -q tests/unit/test_agent.py`
    - Mock LlmService 返回预设响应（无 tool / 有 tool / 多轮 tool）
    - Mock ToolRegistry 返回预设 ToolResult
    - 验证 StreamEvent 序列正确
    - 验证 Hook/Middleware 调用顺序
    - 验证超时和异常处理
  - `pytest -q tests/integration/test_agent_tool_loop.py`
    - 使用 MockLlmService + 真实 ToolRegistry + MockTool
    - 验证完整工具循环流程

---

### 阶段 E：RAG 工具集

---

### E1：KnowledgeQueryTool

- **目标**：实现知识库检索工具，复用现有 `HybridSearch`（Dense + Sparse + RRF），返回带引用的检索结果。
- **修改文件**：
  - `src/agent/tools/knowledge_query.py`
- **实现类/函数**：
  - `KnowledgeQueryArgs(BaseModel)`：`query: str`, `top_k: int = 5`, `collection: str = "default"`
  - `KnowledgeQueryTool(Tool[KnowledgeQueryArgs])`：
    - 内部复用 `QueryKnowledgeHubTool` 的初始化逻辑（`_ensure_initialized`）
    - `execute()`：调用 `HybridSearch.search()` → 可选 Rerank → 格式化为 `ToolResult`
    - `result_for_llm` 包含：检索到的文本片段 + 来源引用
- **验收标准**：
  - 调用 `HybridSearch` 返回正确结果
  - 空结果时返回 "未找到相关内容"
  - 异常时 graceful 返回错误信息
  - `result_for_llm` 格式清晰，LLM 可理解
- **测试方法**：`pytest -q tests/unit/test_knowledge_query.py`
  - Mock HybridSearch 返回固定 RetrievalResult
  - 测试正常检索、空结果、异常路径
  - 测试 result_for_llm 格式

---

### E2：PptxLoader + DocumentIngestTool

- **目标**：(1) 实现 PPT/PPTX 文档加载器，与现有 `PdfLoader` 输出格式一致。(2) 实现文档入库工具，支持 Agent 触发文件入库。
- **修改文件**：
  - `src/libs/loader/pptx_loader.py`
  - `src/agent/tools/document_ingest.py`
- **实现类/函数**：
  - `PptxLoader(BaseLoader)`：
    - `load(file_path) -> Document`
    - 提取：每页标题 + 文本内容 + speaker notes + 表格 + 图片
    - 输出 Markdown 格式，图片占位符 `[IMAGE: {image_id}]`
    - 元数据：`source_path`, `doc_type="pptx"`, `title`, `slide_count`, `images`
  - `DocumentIngestArgs(BaseModel)`：`file_path: str`, `collection: str = "default"`
  - `DocumentIngestTool(Tool[DocumentIngestArgs])`：
    - 根据文件扩展名选择 Loader（.pdf → PdfLoader, .pptx → PptxLoader）
    - 调用 `IngestionPipeline.run()` 执行全流程
    - 返回入库结果（chunk 数、图片数）
- **验收标准**：
  - PptxLoader 正确解析包含文本、表格、图片的 PPT
  - 输出 Document 对象通过 `__post_init__` 校验
  - DocumentIngestTool 正确选择 Loader
  - 已处理文件跳过（SHA256 去重）
- **测试方法**：
  - `pytest -q tests/unit/test_pptx_loader.py`
    - 使用 `tests/fixtures/sample.pptx` 测试解析
    - 测试空 PPT、仅图片 PPT、包含表格的 PPT
  - `pytest -q tests/unit/test_document_ingest.py` (Mock Pipeline)

---

### E3：ReviewSummaryTool

- **目标**：实现考点复习总结工具。检索指定章节/主题的知识库内容，调用 LLM 生成结构化考点摘要。集成 Memory 实现个性化（标注薄弱知识点）。
- **修改文件**：
  - `src/agent/tools/review_summary.py`
- **实现类/函数**：
  - `ReviewSummaryArgs(BaseModel)`：`topic: str`, `chapter: Optional[str]`, `include_weak_points: bool = True`
  - `ReviewSummaryTool(Tool[ReviewSummaryArgs])`：
    - 注入 `HybridSearch` + `LlmService` + 可选 `ErrorMemory` + `KnowledgeMapMemory`
    - `execute()` 流程：
      1. 调用 HybridSearch 检索相关内容
      2. 如果 `include_weak_points` 且 Memory 可用：查询薄弱知识点
      3. 构建总结 prompt：要求 LLM 生成 "核心概念 / 重要定理 / 易错点 / 章节关联"
      4. 调用 LLM 生成摘要
      5. **降级**：LLM 不可用时返回原始检索结果
    - `result_for_llm`：Markdown 格式的考点摘要
- **验收标准**：
  - 正常路径返回结构化考点摘要
  - 薄弱知识点在摘要中标注 ⚠️
  - LLM 不可用时降级为原始检索结果
  - 无检索结果时返回 "未找到相关内容"
- **测试方法**：`pytest -q tests/unit/test_review_summary.py`
  - Mock HybridSearch + LlmService
  - 测试正常总结、带薄弱点标注、LLM 降级、空结果

---

### E4：QuizGeneratorTool + QuizEvaluatorTool

- **目标**：(1) 习题生成工具：基于知识库内容生成指定题型和数量的习题。(2) 答案评判工具：评判用户答案的正确性并给出解析。两者均集成 Memory 实现自适应。
- **修改文件**：
  - `src/agent/tools/quiz_generator.py`
  - `src/agent/tools/quiz_evaluator.py`
- **实现类/函数**：
  - `QuizGeneratorArgs(BaseModel)`：
    - `topic: str`
    - `question_type: str = "选择题"`（"选择题" | "填空题" | "简答题" | "SQL题"）
    - `count: int = 3`
    - `difficulty: int = 3`（1-5）
  - `QuizGeneratorTool(Tool[QuizGeneratorArgs])`：
    - 注入 `HybridSearch` + `LlmService` + 可选 `ErrorMemory` + `KnowledgeMapMemory`
    - `execute()` 流程：
      1. 检索相关知识
      2. 如果 Memory 可用：获取薄弱知识点，调整出题权重
      3. 构建出题 prompt（含题型要求、难度、知识范围、答案和解析）
      4. 调用 LLM 生成题目
      5. 解析 LLM 输出为结构化 JSON
    - `result_for_llm`：JSON 格式题目列表
  - `QuizEvaluatorArgs(BaseModel)`：
    - `question: str`
    - `user_answer: str`
    - `correct_answer: str`
    - `question_type: str`
    - `topic: str`
    - `concepts: list[str] = []`
  - `QuizEvaluatorTool(Tool[QuizEvaluatorArgs])`：
    - 注入 `LlmService` + 可选 `ErrorMemory` + `KnowledgeMapMemory`
    - `execute()` 流程：
      1. 调用 LLM 评判答案（对/错/部分正确）
      2. 生成解析说明
      3. 如果 Memory 可用：
         - 错误 → 写入 ErrorMemory
         - 更新 KnowledgeMapMemory（答对 +0.1 / 答错 -0.15）
    - `result_for_llm`：评判结果 + 解析
- **验收标准**：
  - QuizGenerator 生成的题目格式正确（JSON 含 question, options, answer, explanation）
  - QuizEvaluator 正确判断对/错/部分正确
  - Memory 更新正确（错题写入、掌握度更新）
  - LLM 不可用时 graceful 返回错误
- **测试方法**：`pytest -q tests/unit/test_quiz_tools.py`
  - Mock LlmService 返回预设题目/评判结果
  - 测试各题型生成
  - 测试评判 + Memory 更新
  - 测试 LLM 降级

---

### 阶段 F：Memory 记忆系统

---

### F1：短期记忆 — 上下文工程

- **目标**：实现 `ConversationFilter` 接口和 `ContextEngineeringFilter`，通过渐进式压缩策略管理短期记忆（上下文窗口）。
- **修改文件**：
  - `src/agent/memory/base.py`（ConversationFilter 接口）
  - `src/agent/memory/context_filter.py`
- **实现类/函数**：
  - `ConversationFilter(ABC)`：
    - `filter_messages(messages: list[Message]) -> list[Message]`
  - `ContextEngineeringFilter(ConversationFilter)`：
    - Level 1 **滑动窗口**：保留最近 `max_messages` 条
    - Level 2 **工具结果卸载**：超过 2000 字符的 tool 消息卸载到文件，仅保留摘要引用
    - Level 3 **历史摘要**：超出窗口的历史消息生成 LLM 摘要（可选，需 LlmService）
    - Level 4 **Token 预算**：按 system_prompt + memory + history + current 分配 token 预算
    - `_offload_to_file(content: str) -> str`：卸载到 `data/context_offload/{ref_id}.txt`
    - `_summarize_old(messages: list[Message]) -> str`：LLM 摘要
- **验收标准**：
  - 40 条消息以下不做任何处理
  - 超过 40 条时滑动窗口生效
  - 大型工具结果被卸载，消息中仅保留引用
  - 卸载文件可通过 ref_id 找回
  - LLM 摘要失败时 fallback 到简单截断
- **测试方法**：`pytest -q tests/unit/test_context_filter.py`
  - 构造不同长度的消息列表
  - 验证滑动窗口行为
  - 验证工具结果卸载和引用
  - 验证 LLM 摘要 fallback

---

### F2：长期记忆 — 四类 Memory Store

- **目标**：实现四类长期记忆的存储和检索。使用 SQLite 做结构化存储 + ChromaDB 做语义检索（可选）。
- **修改文件**：
  - `src/agent/memory/student_profile.py`
  - `src/agent/memory/error_memory.py`
  - `src/agent/memory/knowledge_map.py`
  - `src/agent/memory/skill_memory.py`
- **实现类/函数**：

  **StudentProfileMemory**：
  - `StudentProfile(BaseModel)`：user_id, preferences, weak_topics, strong_topics, learning_pace, total_sessions, total_quizzes, overall_accuracy, last_active, notes
  - `get_profile(user_id: str) -> StudentProfile`
  - `update_profile(user_id: str, updates: dict) -> None`
  - 存储：SQLite `data/memory/profiles.db`

  **ErrorMemory**：
  - `ErrorRecord(BaseModel)`：id, user_id, question, question_type, topic, concepts, user_answer, correct_answer, explanation, error_type, difficulty, mastered, created_at, mastered_at
  - `add_error(user_id: str, record: ErrorRecord) -> None`
  - `get_errors(user_id: str, topic: Optional[str], mastered: Optional[bool], limit: int) -> list[ErrorRecord]`
  - `mark_mastered(error_id: str) -> None`
  - `get_weak_concepts(user_id: str) -> list[str]`：统计高频错误概念
  - 存储：SQLite `data/memory/errors.db`

  **KnowledgeMapMemory**：
  - `KnowledgeNode(BaseModel)`：concept, chapter, mastery_level, quiz_count, correct_count, last_reviewed, review_interval_days
  - `get_node(user_id: str, concept: str) -> Optional[KnowledgeNode]`
  - `update_mastery(user_id: str, concept: str, correct: bool) -> None`：答对 +0.1, 答错 -0.15
  - `get_weak_nodes(user_id: str, threshold: float = 0.5) -> list[KnowledgeNode]`
  - `get_due_for_review(user_id: str) -> list[KnowledgeNode]`：超过 review_interval 的节点
  - `apply_decay(user_id: str) -> int`：艾宾浩斯衰减
  - 存储：SQLite `data/memory/knowledge_map.db`

  **SkillMemory**：
  - `ToolUsageRecord(BaseModel)`：question_pattern, tool_chain, tool_args, quality_score
  - `save_usage(user_id: str, record: ToolUsageRecord) -> None`
  - `search_similar(user_id: str, question: str, limit: int = 3) -> list[ToolUsageRecord]`
  - 存储：SQLite `data/memory/skill_memory.db` + 可选 ChromaDB 语义检索

- **验收标准**：
  - 每类 Memory 的 CRUD 操作正确
  - SQLite 表自动创建（首次访问时）
  - 用户隔离（所有查询按 user_id 过滤）
  - 掌握度计算符合公式（+0.1/-0.15，clamp 到 [0, 1]）
  - 艾宾浩斯衰减逻辑正确
- **测试方法**：`pytest -q tests/unit/test_memory_stores.py`
  - 每类 Memory 独立测试 CRUD
  - 测试掌握度更新计算
  - 测试用户隔离
  - 测试 get_weak_concepts / get_due_for_review

---

### F3：Memory 与 Agent 集成

- **目标**：实现 `LlmContextEnhancer` 在对话开始时注入 Memory 上下文，实现 `MemoryRecordHook` 在对话结束时写入 Memory。
- **修改文件**：
  - `src/agent/memory/enhancer.py`
- **实现类/函数**：
  - `MemoryContextEnhancer`：
    - `enhance_system_prompt(base_prompt: str, user_id: str) -> str`
      - Retrieve StudentProfile → 注入学生画像
      - Retrieve ErrorMemory → 注入未掌握错题数
      - Retrieve KnowledgeMap → 注入需复习知识点
      - Retrieve SkillMemory → 注入类似问题工具链建议
    - `get_memory_summary(user_id: str) -> str`：格式化所有 Memory 为文本
  - `MemoryRecordHook(LifecycleHook)`：
    - `after_message(conversation: Conversation)`：
      - 从对话中提取学习状态变化 → 更新 StudentProfile
      - 从对话中提取成功的工具链 → 保存到 SkillMemory
- **验收标准**：
  - 对话开始时 system prompt 包含 Memory 上下文
  - Memory 为空时不注入（不影响正常对话）
  - 对话结束时 StudentProfile 正确更新
  - 工具链保存正确
- **测试方法**：
  - `pytest -q tests/unit/test_memory_enhancer.py`
    - Mock Memory stores 返回预设数据
    - 验证 prompt 注入内容
  - `pytest -q tests/integration/test_agent_memory.py`
    - 使用 MockLlmService + 真实 Memory stores
    - 验证完整 Memory 读写循环

---

### 阶段 G：Skill 技能系统

---

### G1：SkillRegistry + 渐进式加载

- **目标**：实现 Skill 注册中心和三层渐进式加载机制（元数据 → 指令 → 资源）。
- **修改文件**：
  - `src/agent/skills/registry.py`
- **实现类/函数**：
  - `SkillMetadata(BaseModel)`：name, description, trigger_patterns, tools_required, memory_required, estimated_tokens, difficulty
  - `SkillInstruction(BaseModel)`：name, steps, output_format, quality_checks
  - `SkillRegistry`：
    - `__init__(skills_dir: str)`
    - `_load_all_metadata()`：Level 1，启动时解析所有 SKILL.md 的 YAML frontmatter
    - `get_skill_descriptions_for_prompt() -> str`：返回简短描述列表（注入 system prompt）
    - `match_skill(user_message: str) -> Optional[str]`：关键词匹配
    - `load_instruction(skill_name: str) -> SkillInstruction`：Level 2，加载完整 SOP
    - `load_resource(skill_name: str, resource_path: str) -> str`：Level 3，加载资源文件
- **验收标准**：
  - 启动时仅加载元数据（< 1KB）
  - 匹配正确的 Skill
  - 无匹配时返回 None
  - 指令和资源按需加载
  - 无效 SKILL.md 跳过并 log warning
- **测试方法**：`pytest -q tests/unit/test_skill_registry.py`
  - 创建临时 skills 目录，放置测试 SKILL.md
  - 测试元数据加载
  - 测试关键词匹配
  - 测试指令加载
  - 测试资源加载
  - 测试无效文件处理

---

### G2：SkillWorkflowHandler 集成

- **目标**：实现 WorkflowHandler，在消息到达 LLM 前检查是否匹配 Skill，如匹配则将 Skill SOP 注入 system prompt。
- **修改文件**：
  - `src/agent/skills/workflow.py`
- **实现类/函数**：
  - `SkillWorkflowHandler`：
    - `__init__(skill_registry: SkillRegistry)`
    - `try_handle(user_message: str, user_id: str) -> WorkflowResult`：
      - 调用 `skill_registry.match_skill()`
      - 如匹配：加载 Skill 指令 → 返回 `WorkflowResult(should_skip_llm=False, skill_instruction=..., matched_skill=...)`
      - 如未匹配：返回 `WorkflowResult(should_skip_llm=False, skill_instruction=None)`
    - 处理 `/help` 命令：返回可用 Skill 列表
  - `WorkflowResult(BaseModel)`：
    - `should_skip_llm: bool`
    - `skill_instruction: Optional[str]`：注入到 system prompt 的 [Active Skill] 区
    - `matched_skill: Optional[str]`
    - `direct_response: Optional[str]`：/help 等直接返回
  - Agent 集成点：在构建 SystemPrompt 前调用 `try_handle`，将 `skill_instruction` 传给 `PromptBuilder`
- **验收标准**：
  - 匹配到 Skill 时 system prompt 包含 [Active Skill] 区
  - 未匹配时不影响正常流程
  - /help 返回 Skill 列表
  - Skill 加载失败时 log warning 并继续
- **测试方法**：
  - `pytest -q tests/unit/test_skill_workflow.py`
  - `pytest -q tests/integration/test_agent_skill.py`

---

### G3：预定义 Skill 编写

- **目标**：编写 5 个预定义 Skill 的 SKILL.md 文件，包含完整的 SOP 步骤和模板。
- **修改文件**：
  - `src/agent/skills/definitions/exam_prep/SKILL.md`
  - `src/agent/skills/definitions/error_review/SKILL.md`
  - `src/agent/skills/definitions/chapter_deep_dive/SKILL.md`
  - `src/agent/skills/definitions/quiz_drill/SKILL.md`
  - `src/agent/skills/definitions/knowledge_check/SKILL.md`
- **每个 Skill 包含**：
  - YAML frontmatter：name, description, trigger_patterns, tools_required, memory_required
  - Markdown body：分步 SOP、输出格式、质量自检
- **验收标准**：
  - 每个 SKILL.md 的 YAML frontmatter 可被 `SkillRegistry` 正确解析
  - trigger_patterns 覆盖常见用户表达
  - SOP 步骤中引用的工具名与 ToolRegistry 中一致
- **测试方法**：`pytest -q tests/unit/test_skill_registry.py::test_load_predefined_skills`
  - 测试所有预定义 Skill 可被加载
  - 测试 trigger_patterns 匹配

---

### 阶段 H：Server 与前端

---

### H1：FastAPI + SSE + ChatHandler

- **目标**：实现 FastAPI 应用和 SSE 流式聊天端点，将 Agent 的 `AsyncGenerator[StreamEvent]` 转为 SSE 事件流。
- **修改文件**：
  - `src/server/app.py`
  - `src/server/routes.py`
  - `src/server/chat_handler.py`
  - `src/server/models.py`
- **实现类/函数**：
  - `ChatRequest(BaseModel)`：message, conversation_id, user_id
  - `ChatStreamChunk(BaseModel)`：type, content, tool_name, metadata, timestamp
  - `ChatHandler`：
    - `__init__(agent: Agent)`
    - `async handle_stream(request: ChatRequest) -> AsyncGenerator[ChatStreamChunk, None]`
  - FastAPI 路由：
    - `POST /api/chat`：SSE 端点，使用 `sse-starlette` 的 `EventSourceResponse`
    - `GET /api/conversations/{conversation_id}`：获取会话历史
    - `GET /api/health`：健康检查
    - `GET /`：静态 Web UI
  - `create_app(agent: Agent) -> FastAPI`：应用工厂
- **验收标准**：
  - SSE 端点正确发送 `data: {json}\n\n` 格式
  - 流式输出每个 StreamEvent 立即发送（不缓冲）
  - CORS 正确配置
  - 健康检查返回 200
  - 会话历史端点返回正确数据
- **测试方法**：
  - `pytest -q tests/unit/test_server_models.py`
  - `pytest -q tests/integration/test_server_sse.py`
    - 使用 `httpx.AsyncClient` + `TestClient` 测试 SSE
    - 验证 SSE 事件格式和顺序

---

### H2：文件上传端点

- **目标**：实现文件上传 API，接收 PPT/PDF 文件并触发入库流程。
- **修改文件**：
  - `src/server/routes.py`（新增上传路由）
- **实现类/函数**：
  - `POST /api/upload`：
    - 接收 `UploadFile`
    - 校验：文件类型 (.pdf/.pptx)、文件大小 (< 50MB)
    - 保存到 `data/uploads/{user_id}/`
    - 异步调用 `DocumentIngestTool.execute()`
    - 返回 `{success, filename, chunks, images}`
- **验收标准**：
  - 正确接收和保存文件
  - 非法文件类型返回 400
  - 超大文件返回 413
  - 入库成功返回 chunk 数量
  - 入库失败返回错误信息
- **测试方法**：`pytest -q tests/integration/test_server_sse.py::test_upload`
  - 使用 `httpx.AsyncClient` 上传测试文件
  - 测试文件类型校验
  - 测试大小校验

---

### H3：Web UI 前端

- **目标**：实现单页面 Web UI（纯 HTML/CSS/JS），支持聊天对话（SSE 流式）、文件上传、习题交互。
- **修改文件**：
  - `src/web/index.html`
  - `src/web/style.css`
  - `src/web/app.js`
- **实现要点**：
  - **聊天界面**：
    - 消息输入框 + 发送按钮
    - SSE 连接 `/api/chat`，逐事件渲染
    - Markdown 渲染（使用 marked.js CDN）
    - 代码高亮（使用 highlight.js CDN）
    - 工具调用状态展示（loading spinner + 工具名）
  - **文件上传**：
    - 拖拽上传区域
    - 进度条展示
    - 成功/失败提示
  - **习题模式**：
    - 选择题渲染为按钮卡片
    - 用户点击/输入后提交评判
    - 正确/错误反馈动画
  - **设计**：
    - 响应式布局
    - 深色/浅色主题切换
    - 现代简洁风格
- **验收标准**：
  - 对话流式渲染流畅，无闪烁
  - 文件上传功能完整
  - 习题可交互
  - 移动端可用
- **测试方法**：手动测试 + 浏览器开发工具检查
  - 验证 SSE 连接建立和消息接收
  - 验证 Markdown 渲染
  - 验证文件上传

---

### 阶段 I：稳定性与可观测

---

### I1：限流 + 重试 + 熔断

- **目标**：实现 `RateLimitHook`（限流）、`RetryMiddleware`（LLM 重试）、`CircuitBreaker`（熔断器）。
- **修改文件**：
  - `src/agent/hooks/rate_limit.py`
  - `src/agent/hooks/retry_middleware.py`
- **实现类/函数**：
  - `TokenBucket`：令牌桶算法
  - `RateLimitHook(LifecycleHook)`：per-user 限流，20 次/分钟
  - `RetryMiddleware(LlmMiddleware)`：LLM 错误指数退避重试，最多 3 次
  - `CircuitBreaker`：三态（CLOSED/OPEN/HALF_OPEN），failure_threshold=5, cooldown=30s
- **验收标准**：
  - 超出限流抛 RateLimitExceeded
  - 429/500 错误触发重试（1s, 2s, 4s）
  - 连续 5 次失败触发熔断
  - 熔断后请求立即拒绝
  - 30s 后进入 HALF_OPEN 状态
- **测试方法**：`pytest -q tests/unit/test_rate_limit.py tests/unit/test_retry.py tests/unit/test_circuit_breaker.py`

---

### I2：审计日志

- **目标**：实现 JSONL 格式的审计日志，记录所有工具调用、LLM 请求、文件上传操作。
- **修改文件**：
  - `src/agent/hooks/audit.py`
- **实现类/函数**：
  - `AuditEvent(BaseModel)`：timestamp, user_id, action, tool_name, parameters, result_summary, duration_ms, metadata
  - `FileAuditLogger`：
    - `log_event(event: AuditEvent) -> None`：写入 JSONL 文件
    - `_sanitize_params(params: dict) -> dict`：对 password/secret/token 等字段脱敏
  - `AuditHook(LifecycleHook)`：
    - `before_tool` / `after_tool` 中记录审计事件
- **验收标准**：
  - 每次工具调用产生一条审计记录
  - 敏感字段脱敏
  - JSONL 文件可直接 `jq` 查询
- **测试方法**：`pytest -q tests/unit/test_audit.py`

---

### I3：可观测性（Metrics + Trace）

- **目标**：实现 Metrics 收集器和 Trace span 管理，支持性能分析和瓶颈定位。
- **修改文件**：
  - `src/agent/hooks/observability.py`
- **实现类/函数**：
  - `Span(BaseModel)`：id, name, start_time, end_time, attributes, parent_id
  - `MetricsCollector`：
    - `record_counter(name, value, tags)`
    - `record_histogram(name, value, tags)`
    - `create_span(name) -> Span`
    - `end_span(span)`
    - 定期写入 `logs/metrics.jsonl`
  - Agent 中的埋点：
    - `agent.send_message`（整体耗时）
    - `agent.tool.execute`（每个工具耗时）
    - `llm.request`（每次 LLM 耗时 + token 数）
    - `retrieval.search`（检索耗时）
- **验收标准**：
  - 每次请求产生完整的 Span 链
  - 耗时准确（误差 < 10ms）
  - Metrics 文件可用于后续分析
- **测试方法**：`pytest -q tests/unit/test_observability.py`

---

### 阶段 J：RAG 深度优化

> **设计动机**：面试中 RAG 系统常被认为"做得浅"，主要体现在：文档解析丢失公式/结构信息、分块策略单一、向量存储无参数调优、不支持生产级数据库、前端无法渲染数学公式。本阶段系统性解决这些问题。

---

### J1：DocumentSection 数据类 + 公式工具模块

- **目标**：(1) 在 `core/types.py` 新增 `DocumentSection` 数据类，为结构化解析提供统一表示。(2) 新建 `math_utils.py` 提供 OMML→LaTeX 转换和 Unicode 数学符号标准化。
- **修改文件**：
  - `src/core/types.py`
  - `src/libs/loader/math_utils.py`（新建）
- **实现类/函数**：
  - `DocumentSection(BaseModel)`：
    - `title: str` — 章节标题
    - `level: int` — 标题层级（1=章, 2=节, 3=小节）
    - `content: str` — 正文内容（Markdown 格式，公式用 `$...$` 包裹）
    - `content_type: str` — "concept" / "exercise" / "definition" / "example" / "formula"
    - `page_or_slide: int` — 来源页码或 slide 编号
    - `has_formula: bool` — 是否包含数学公式
    - `images: list[dict]` — 关联图片元数据
  - `omml_to_latex(element: etree.Element) -> str`：
    - 解析 OMML XML 节点，将常见标签映射为 LaTeX
    - 支持分数 `oMath/f` → `\frac{}{}`、上下标 `sSup/sSub` → `^{}_{}`、根号 `rad` → `\sqrt{}`、希腊字母、矩阵等
    - 未识别标签提取纯文本 fallback
  - `unicode_math_to_latex(text: str) -> str`：
    - 将 Unicode 数学符号映射为 LaTeX 命令
    - 覆盖：希腊字母（α→`\alpha`）、运算符（≤→`\leq`）、上下标（²→`^{2}`）、特殊符号（∞→`\infty`）
  - `normalize_latex(text: str) -> str`：
    - 统一公式定界符：`\(...\)` → `$...$`，`\[...\]` → `$$...$$`
    - 清理冗余空白
- **验收标准**：
  - `DocumentSection` 可正常实例化和序列化
  - `omml_to_latex` 可将标准 OMML 分数/上下标/根号节点转为正确 LaTeX
  - `unicode_math_to_latex` 正确映射常见数学符号（α, β, ≤, ∑, ∫ 等）
  - `normalize_latex` 统一不同来源的公式定界符
- **测试方法**：`pytest -q tests/unit/test_math_utils.py`
  - 构造 OMML XML 片段测试 `omml_to_latex`
  - 测试 Unicode 字符串映射
  - 测试定界符归一化

---

### J2：PptxLoader 增强 — OMML 公式提取 + 结构化解析

- **目标**：增强 PptxLoader，使其能 (1) 从 shape XML 中提取 OMML 公式并转为 LaTeX，(2) 输出结构化的 `DocumentSection` 列表。
- **修改文件**：
  - `src/libs/loader/pptx_loader.py`
- **实现要点**：
  - **OMML 提取**：遍历 `shape._element` 的 XML，查找 `{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath` 节点，调用 `omml_to_latex()` 转换
  - **结构化 sections**：每个 slide 生成一个 `DocumentSection`：
    - `title`：slide 标题（从 title placeholder 获取）
    - `level`：根据字体大小或占位符类型推断层级（标题 slide → level 1，内容 slide → level 2）
    - `content`：文本 + 表格 + 公式（LaTeX）+ 图片占位符 + 备注
    - `content_type`：根据关键词规则标注（含"例题/练习" → exercise，含"定义/定理" → definition）
    - `has_formula`：检测是否包含 `$...$` 或 `$$...$$`
  - Document.metadata 中新增 `sections: list[dict]`（序列化的 DocumentSection 列表）
- **验收标准**：
  - 包含公式的 PPT slide 中公式被正确转为 LaTeX（如 `$E = mc^2$`）
  - 无公式的 slide 不受影响
  - 每个 slide 生成对应的 DocumentSection
  - content_type 标注合理
- **测试方法**：`pytest -q tests/unit/test_pptx_loader.py`
  - 使用包含公式的测试 PPTX（可通过 python-pptx 构造）
  - 验证公式提取结果
  - 验证结构化 sections 输出

---

### J3：PdfLoader 增强 — 公式后处理 + 章节检测 + 习题识别

- **目标**：增强 PdfLoader，对 MarkItDown 的输出做后处理，修复公式乱码、检测章节结构、识别习题区域。
- **修改文件**：
  - `src/libs/loader/pdf_loader.py`
- **实现要点**：
  - **MathFormulaPostProcessor**（新增内部类或方法）：
    - 调用 `unicode_math_to_latex()` 修复常见乱码
    - 正则检测已有 LaTeX 片段并调用 `normalize_latex()` 标准化
    - 可选 fallback：对检测到公式密集但无法解析的页面，用 Vision LLM 做公式 OCR（`config.vision_llm.enabled` 控制开关）
  - **章节检测**：
    - 正则匹配 `^#{1,3} ` 提取 Markdown 标题
    - 匹配 `^第[一二三四五六七八九十\d]+章` 或 `^\d+\.\d+` 等中文/英文章节模式
    - 输出 `DocumentSection` 列表
  - **习题区域识别**：
    - 关键词检测：`习题`、`练习`、`思考题`、`Exercise`、`Problem`
    - 每道习题拆为独立 section，`content_type = "exercise"`
    - 如果检测到答案区域（`答案`、`解答`、`Solution`），与对应习题关联
  - Document.metadata 中新增 `sections: list[dict]`
- **验收标准**：
  - Unicode 数学符号被正确映射为 LaTeX
  - 章节结构被正确提取
  - 习题区域被识别并标注为 "exercise"
  - 非习题 PDF 不受影响
- **测试方法**：`pytest -q tests/unit/test_pdf_loader.py`
  - 使用 `tests/fixtures/generate_complex_pdf.py` 生成包含公式和习题的测试 PDF
  - 验证公式后处理、章节检测、习题识别

---

### J4：DocxLoader — Word 文档解析

- **目标**：新建 DocxLoader，支持 .docx 格式文档解析，包含 OMML 公式提取。
- **修改文件**：
  - `src/libs/loader/docx_loader.py`（新建）
- **实现类/函数**：
  - `DocxLoader(BaseLoader)`：
    - `load(file_path) -> Document`
    - 使用 `python-docx` 解析段落、表格、标题样式
    - 遍历段落的 XML（`paragraph._element`），查找 OMML 公式节点，调用 `omml_to_latex()` 转换
    - 标题层级：从 `paragraph.style.name`（Heading 1 → level 1, Heading 2 → level 2）获取
    - 输出结构化 `DocumentSection` 列表
    - 图片提取：使用 `docx.opc.part` 获取嵌入图片
    - 元数据：`source_path`, `doc_type="docx"`, `title`, `page_count`, `sections`, `images`
- **验收标准**：
  - 正确解析包含标题、段落、表格、公式的 Word 文档
  - OMML 公式转换为 LaTeX
  - 标题层级正确识别
  - 图片正确提取
- **测试方法**：`pytest -q tests/unit/test_docx_loader.py`
  - 使用 `python-docx` 构造包含公式的测试 .docx
  - 验证解析结果

---

### J5：SemanticSplitter — 基于 Embedding 的语义分块

- **目标**：新建 SemanticSplitter，通过 embedding 余弦相似度检测语义断点，实现语义感知的文本分块。
- **修改文件**：
  - `src/ingestion/chunking/semantic_splitter.py`（新建）
- **实现类/函数**：
  - `SemanticSplitter(BaseSplitter)`：
    - 构造参数：`embedding_client`, `similarity_threshold: float = 0.5`, `min_chunk_size: int = 100`, `max_chunk_size: int = 1500`
    - `split(document: Document) -> list[Chunk]` 流程：
      1. 按句子/段落预切分（正则 `[。！？\n\n]`）
      2. 对每个预切分片段做 embedding
      3. 计算相邻片段的余弦相似度
      4. 在相似度低于 `similarity_threshold` 的位置切分（语义断点）
      5. 合并过小的 chunk，拆分过大的 chunk
    - 性能优化：embedding 批量调用，缓存避免重复计算
- **验收标准**：
  - 语义相近的段落被合并为一个 chunk
  - 语义不同的段落被切分到不同 chunk
  - chunk 大小在 `[min_chunk_size, max_chunk_size]` 范围内
  - embedding 调用次数合理（批量而非逐句）
- **测试方法**：`pytest -q tests/unit/test_semantic_splitter.py`
  - Mock embedding client 返回可控向量
  - 构造语义变化明显的文本验证切分结果
  - 验证 chunk 大小约束

---

### J6：StructureAwareSplitter — 结构感知分块

- **目标**：新建 StructureAwareSplitter，优先按文档结构切分，内部再做语义细分。
- **修改文件**：
  - `src/ingestion/chunking/structure_splitter.py`（新建）
- **实现类/函数**：
  - `StructureAwareSplitter(BaseSplitter)`：
    - 构造参数：`max_chunk_size: int = 1500`, `fallback_splitter: BaseSplitter`（用于内部细分）
    - `split(document: Document) -> list[Chunk]` 流程：
      1. 如果 Document.metadata 包含 `sections`：按 sections 切分
      2. 否则 fallback 到 Markdown 标题切分（`^#{1,3} `）
      3. 每个 section 如果超过 `max_chunk_size`：调用 `fallback_splitter` 做内部细分
      4. 每个 chunk 继承 section 的元数据（title, level, content_type, has_formula）
    - 特殊处理：
      - 习题（content_type="exercise"）保持题目 + 答案在同一 chunk
      - 公式（has_formula=True）的 chunk 不在公式中间切断
- **验收标准**：
  - 有结构化 sections 的文档按 section 切分
  - 无结构化信息的文档按 Markdown 标题 fallback
  - 过大 section 被内部细分
  - 习题的题目和答案不被拆开
  - 公式不被从中间切断
- **测试方法**：`pytest -q tests/unit/test_structure_splitter.py`
  - 构造包含 sections 的 Document 对象
  - 验证结构化切分、fallback 切分、大 section 细分
  - 验证习题和公式保护

---

### J7：SplitterFactory 注册新 Provider

- **目标**：在 SplitterFactory 中注册 `semantic` 和 `structure` 两种新的分块策略 provider。
- **修改文件**：
  - `src/ingestion/chunking/splitter_factory.py`
- **实现要点**：
  - 注册 `"semantic"` → `SemanticSplitter`
  - 注册 `"structure"` → `StructureAwareSplitter`
  - `settings.yaml` 中 `ingestion.splitter` 支持新值
- **验收标准**：
  - `SplitterFactory.create("semantic", settings)` 返回 SemanticSplitter
  - `SplitterFactory.create("structure", settings)` 返回 StructureAwareSplitter
  - 未知 provider 抛出明确错误
- **测试方法**：`pytest -q tests/unit/test_splitter_factory.py`

---

### J8：Parent-Child 层级 Chunk

- **目标**：修改 DocumentChunker，在细粒度 chunk 之上生成粗粒度 parent chunk，形成 parent-child 层级索引。检索时命中 child chunk 可回溯到 parent 获取完整上下文。
- **修改文件**：
  - `src/ingestion/chunking/document_chunker.py`
  - `src/core/types.py`（Chunk 类增加 `parent_id` 字段）
- **实现要点**：
  - `Chunk.parent_id: Optional[str]` — 指向 parent chunk 的 ID（None 表示 parent 自身）
  - 两层索引策略：
    - **Parent chunk**（粗粒度）：由整个 section 或相邻 sections 合并生成，`chunk_size ≈ 2000-3000`
    - **Child chunk**（细粒度）：在 parent 内部切分，`chunk_size ≈ 500-1000`
  - 两层 chunk 都入库到向量数据库，child 的 metadata 包含 `parent_id`
  - 检索时搜 child，拿到 parent_id 后回溯 parent 获取完整上下文
- **验收标准**：
  - 每个 child chunk 有正确的 `parent_id`
  - parent chunk 覆盖所有 child 内容
  - 通过 `parent_id` 可查到对应的 parent chunk
- **测试方法**：`pytest -q tests/unit/test_parent_child_chunking.py`
  - 构造多 section 文档，验证 parent-child 关系
  - 验证 parent_id 链路

---

### J9：ChromaStore HNSW 参数调优

- **目标**：将 ChromaDB 的 HNSW 参数从默认值调优为经过考量的值，并暴露到 `settings.yaml`。
- **修改文件**：
  - `src/libs/vector_store/chroma_store.py`
  - `config/settings.yaml`
- **实现要点**：
  - 从 `settings.yaml` 读取 HNSW 参数，传入 collection metadata：
    ```python
    metadata={
        "hnsw:space": hnsw_cfg.space,           # "cosine"
        "hnsw:construction_ef": hnsw_cfg.construction_ef,  # 200（默认100）
        "hnsw:M": hnsw_cfg.M,                   # 32（默认16）
        "hnsw:search_ef": hnsw_cfg.search_ef,   # 100（默认10）
    }
    ```
  - settings.yaml 新增：
    ```yaml
    vector_store:
      hnsw:
        space: "cosine"
        M: 32
        construction_ef: 200
        search_ef: 100
    ```
- **面试要点**：
  - `M=32` vs 默认 `16`：增加图的连通性，提高召回率，代价是内存和构建时间
  - `construction_ef=200` vs 默认 `100`：构建时探索更多候选，索引质量更高
  - `search_ef=100` vs 默认 `10`：搜索时考察更多节点，召回更准，延迟略增
  - 教育场景（< 50 万 chunks）下这些参数充分够用
- **验收标准**：
  - 参数从配置读取，非硬编码
  - collection 创建时参数生效
  - 参数不合法时有校验
- **测试方法**：`pytest -q tests/unit/test_chroma_store.py`

---

### J10：ChromaStore 多 Collection + 高级查询

- **目标**：增强 ChromaStore 支持动态 collection 切换、score_threshold 过滤、复杂 metadata filter。
- **修改文件**：
  - `src/libs/vector_store/chroma_store.py`
- **实现要点**：
  - `switch_collection(name: str)` — 动态切换当前 collection
  - `query()` 增加 `score_threshold: Optional[float]` 参数，过滤低于阈值的结果
  - `_build_where_clause()` 增强：支持 `$and`/`$or` 组合过滤
  - `list_collections() -> list[dict]` — 列出所有 collection 及其统计信息
- **验收标准**：
  - 可在运行时切换 collection
  - score_threshold 正确过滤低相关性结果
  - 复杂 metadata filter（`$and`/`$or`）正确执行
- **测试方法**：`pytest -q tests/unit/test_chroma_store.py`

---

### J11：MilvusStore 实现 — Milvus Lite 嵌入式

- **目标**：新建 MilvusStore，使用 Milvus Lite 嵌入式模式（无需 Docker），实现 BaseVectorStore 全部接口。
- **修改文件**：
  - `src/libs/vector_store/milvus_store.py`（新建）
- **实现类/函数**：
  - `MilvusStore(BaseVectorStore)`：
    - 构造函数：从 settings 读取 `uri`（Milvus Lite 使用本地文件路径，如 `./data/db/milvus.db`）、`dim`（向量维度）
    - **Collection Schema**（显式定义）：
      - `id`: VARCHAR(256), primary key
      - `vector`: FLOAT_VECTOR(dim)
      - `text`: VARCHAR(65535)
      - `content_type`: VARCHAR(64)
      - `source_path`: VARCHAR(1024)
      - `metadata_json`: VARCHAR(65535)（其余 metadata 序列化为 JSON）
    - **索引**：HNSW, metric_type="COSINE", M=32, efConstruction=200
    - `upsert(records)` — 使用 `client.upsert()`
    - `query(vector, top_k, filters)` — 使用 `client.search()`, 支持标量过滤表达式
    - `delete(ids)` — 使用 `client.delete()`
    - `clear()` — 删除并重建 collection
    - `get_by_ids(ids)` — 使用 `client.get()`
    - `delete_by_metadata(filter_dict)` — 使用标量过滤表达式删除
  - Milvus 标量过滤表达式构建：`_build_filter_expr(filters) -> str`
    - 等值：`field == "value"`
    - 组合：`field1 == "v1" and field2 == "v2"`
- **面试要点**：
  - 显式 Schema vs ChromaDB schemaless：生产中强类型约束更安全
  - Milvus Lite vs Standalone vs Cluster：不同部署形态的取舍
  - 标量索引：Milvus 对 metadata 字段建索引，filter 性能远优于 ChromaDB 的暴力扫描
  - 为什么两个都支持：插件化设计，一行配置切换，上层零修改
- **验收标准**：
  - `MilvusStore` 实现 BaseVectorStore 全部 6 个方法
  - upsert + query 的结果与 ChromaStore 语义一致（同向量检索返回相同排序）
  - 标量过滤正确执行
  - Milvus Lite 模式无需外部服务
- **测试方法**：`pytest -q tests/unit/test_milvus_store.py`
  - 测试 CRUD 全流程
  - 测试标量过滤
  - 测试与 ChromaStore 结果一致性
- **依赖**：`pip install "pymilvus>=2.4.0"`

---

### J12：MilvusStore 注册 + 配置

- **目标**：在 VectorStoreFactory 注册 Milvus provider，在 settings.yaml 中增加 Milvus 配置段。
- **修改文件**：
  - `src/libs/vector_store/__init__.py`
  - `config/settings.yaml`
- **实现要点**：
  - `__init__.py` 增加：
    ```python
    try:
        from src.libs.vector_store.milvus_store import MilvusStore
        VectorStoreFactory.register_provider('milvus', MilvusStore)
    except ImportError:
        pass
    ```
  - `settings.yaml` 增加：
    ```yaml
    vector_store:
      provider: "chroma"  # Options: chroma, milvus
      milvus:
        uri: "./data/db/milvus.db"
        dim: 768
    ```
  - 切换只需改 `provider: "milvus"`
- **验收标准**：
  - `VectorStoreFactory.list_providers()` 包含 `["chroma", "milvus"]`
  - `provider: "milvus"` 时正确创建 MilvusStore
  - `provider: "chroma"` 时行为不变
- **测试方法**：`pytest -q tests/unit/test_vector_store_factory.py`

---

### J13：MCP 工具解耦

- **目标**：重构 `list_collections` 和 `get_document_summary` MCP 工具，消除对 chromadb 的直接 `import` 依赖，改为通过 VectorStoreFactory 获取实例。
- **修改文件**：
  - `src/mcp_server/tools/list_collections.py`
  - `src/mcp_server/tools/get_document_summary.py`
- **实现要点**：
  - 将 `import chromadb; chromadb.PersistentClient(...)` 替换为 `VectorStoreFactory.create(settings)`
  - 在 BaseVectorStore 上增加 `list_collections()` 和 `get_collection_stats()` 方法（ChromaStore 和 MilvusStore 各自实现）
- **验收标准**：
  - 两个 MCP 工具不再直接 `import chromadb`
  - 使用 Milvus provider 时两个工具仍正常工作
- **测试方法**：手动验证 MCP 工具在两种 provider 下可用

---

### J14：MetadataEnricher 增强 — 内容类型标注规则引擎

- **目标**：增强 MetadataEnricher 的规则引擎，对每个 chunk 做细粒度内容类型标注。
- **修改文件**：
  - `src/ingestion/transform/metadata_enricher.py`
- **实现要点**：
  - 新增标注字段：
    - `content_type`："concept" / "exercise" / "definition" / "theorem" / "example" / "formula" / "summary"
    - `has_formula`: bool — 检测 `$...$` 或 `$$...$$`
    - `chapter`: str — 从上下文推断章节编号
    - `difficulty`: int (1-5) — 基于关键词的难度估算（"简单/基础" → 1-2, "复杂/高级/证明" → 4-5）
  - 规则优先级：section 元数据 > 关键词匹配 > 默认值 "concept"
  - 习题结构化：检测并解析 "题号 + 题干 + 选项 + 答案" 模式
- **验收标准**：
  - 定义/定理类内容标注为 "definition"/"theorem"
  - 习题标注为 "exercise"
  - 含公式的 chunk `has_formula=True`
  - 章节信息从 section 元数据继承
- **测试方法**：`pytest -q tests/unit/test_metadata_enricher.py`
  - 构造不同类型文本的 Chunk，验证标注结果

---

### J15：前端 KaTeX 集成

- **目标**：在前端引入 KaTeX，使 LLM 输出的 LaTeX 公式能正确渲染为数学公式。
- **修改文件**：
  - `src/web/index.html`
  - `src/web/app.js`
  - `src/web/style.css`
- **实现要点**：
  - **index.html**：引入 KaTeX CDN（CSS + JS + auto-render 扩展）
  - **app.js**：修改 `renderMarkdown()` 函数：
    1. 先用 `marked.parse()` 渲染 Markdown
    2. 创建临时 DOM 元素
    3. 调用 `renderMathInElement()` 渲染数学公式
    4. 支持定界符：`$$...$$`（块级）、`$...$`（行内）、`\[...\]`、`\(...\)`
    5. `throwOnError: false` 避免无效 LaTeX 导致整段内容消失
  - **style.css**：
    - `.katex` 颜色继承 `var(--text)` 以适配 dark mode
    - 块级公式（`.katex-display`）居中、上下 margin
    - 行内公式垂直对齐调整
- **验收标准**：
  - `$E=mc^2$` 渲染为行内公式
  - `$$\frac{a}{b}$$` 渲染为块级居中公式
  - dark mode 下公式清晰可读
  - 无效 LaTeX 显示原文不报错
  - 流式输出时公式逐步渲染
- **测试方法**：手动测试
  - 在聊天框输入包含公式的问题，验证 LLM 回复中公式正确渲染
  - 验证 dark/light 主题切换

---

### J16：knowledge_query 支持 Parent 回溯 + Metadata Filter

- **目标**：增强 KnowledgeQueryTool，支持 parent chunk 回溯检索和 metadata 过滤。
- **修改文件**：
  - `src/agent/tools/knowledge_query.py`
- **实现要点**：
  - 检索到 child chunk 后，如果 `parent_id` 存在：
    1. 通过 `vector_store.get_by_ids([parent_id])` 获取 parent chunk
    2. 用 parent chunk 的完整上下文替代或补充 child chunk
  - 新增 metadata filter 参数：
    - `content_type: Optional[str]` — 过滤特定类型（如 "exercise"）
    - `chapter: Optional[str]` — 过滤特定章节
  - `result_for_llm` 中标注内容类型和来源章节
- **验收标准**：
  - 检索 child chunk 后能正确回溯到 parent
  - metadata filter 正确过滤
  - 无 parent_id 的 chunk 正常返回
- **测试方法**：`pytest -q tests/unit/test_knowledge_query.py`
  - Mock 包含 parent_id 的检索结果
  - 验证回溯逻辑
  - 验证 metadata filter

---

### J17：Pipeline 和 Settings 更新

- **目标**：更新 IngestionPipeline 支持新的分块策略、.docx 格式、HNSW 配置。
- **修改文件**：
  - `src/ingestion/pipeline.py`
  - `config/settings.yaml`
- **实现要点**：
  - Pipeline 中增加 DocxLoader 的文件扩展名映射（`.docx` → `DocxLoader`）
  - `settings.yaml` 中 `ingestion.splitter` 支持 `"structure"` 和 `"semantic"` 值
  - 新增 `ingestion.parent_child_enabled: true` 开关
  - 新增 `ingestion.semantic_splitter` 配置段（`similarity_threshold`, `min_chunk_size`, `max_chunk_size`）
- **验收标准**：
  - `.docx` 文件可通过 Pipeline 入库
  - `splitter: "structure"` 使用 StructureAwareSplitter
  - `splitter: "semantic"` 使用 SemanticSplitter
  - parent_child 开关可控
- **测试方法**：集成测试，上传 .docx/.pdf/.pptx 文件验证全流程

---

### J18：重新入库 + 端到端测试

- **目标**：清空现有知识库，使用新的解析和分块策略重新入库所有课件，端到端验证效果。
- **步骤**：
  1. 清空 ChromaDB 数据目录
  2. 使用 `splitter: "structure"` + `parent_child_enabled: true` 重新入库 `docs/computer_internet/` 下所有 PPT
  3. 验证检索效果：
     - 查询包含公式的知识点，确认公式正确返回
     - 查询习题，确认 content_type 过滤有效
     - 查询概念，确认 parent chunk 回溯有效
  4. 验证前端：公式正确渲染、各类内容显示正常
  5. 切换 `provider: "milvus"` 重复验证
- **验收标准**：
  - 所有课件成功入库
  - 公式内容在检索结果中正确保留
  - 前端数学公式正确渲染
  - ChromaDB 和 Milvus 两种 provider 都能正常工作
- **测试方法**：手动端到端测试 + 自动化烟测脚本

---

### J19：系统设计文档补充

- **目标**：在 `docs/SYSTEM_DESIGN.md` 中补充 RAG 深度优化的设计决策说明，为面试准备。
- **修改文件**：
  - `docs/SYSTEM_DESIGN.md`
- **补充内容**：
  - **文档解析设计**：为什么需要结构化解析、OMML→LaTeX 的技术选型
  - **分块策略对比**：Recursive vs Semantic vs Structure-Aware 的优劣势和适用场景
  - **Parent-Child 索引**：设计原理、检索回溯流程、与 Late Chunking 的对比
  - **向量数据库设计决策**：
    - 为什么同时支持 ChromaDB 和 Milvus（开发效率 vs 生产能力）
    - HNSW 参数选择依据
    - 数据量增长时的迁移方案
    - 标量过滤索引的价值
  - **内容类型标注**：规则引擎 vs LLM 标注的 trade-off
- **验收标准**：文档清晰、面试可直接引用
- **测试方法**：人工审查

---

### 阶段 K：记忆系统深度升级（个性化复习助手）

> 借鉴阿里 CoPaw ReMe (Remember Me, Refine Me) 记忆框架理念，将 Agent 从"无状态问答机器人"升级为"有长期记忆的个性化复习助手"。

---

### K1：SessionMemory 会话摘要存储

- **目标**：新建 `SessionMemory` 组件，为每次会话存储结构化摘要，实现"你上次问过 TCP"的能力。
- **新建文件**：
  - `src/agent/memory/session_memory.py`
- **实现类/函数**：
  - `SessionSummary(BaseModel)`：
    - `session_id: str` — 关联 Conversation ID
    - `user_id: str`
    - `timestamp: datetime`
    - `topics: list[str]` — 讨论的知识点
    - `key_questions: list[str]` — 用户问的关键问题
    - `mastery_observations: dict[str, str]` — {"TCP": "weak", "IP": "strong"}
    - `preference_snapshot: dict` — 本次偏好快照
    - `summary_text: str` — 一句话摘要
  - `SessionMemory`：
    - `__init__(db_dir)` — SQLite: `session_summaries(id, user_id, data, created_at)`
    - `async save_session(user_id, summary: SessionSummary)`
    - `async get_recent_sessions(user_id, limit=5) -> list[SessionSummary]`
    - `async get_topic_history(user_id, topic) -> list[SessionSummary]` — 搜索包含特定话题的会话
    - `async search_sessions(user_id, query) -> list[SessionSummary]` — 关键词匹配
- **设计理念**：借鉴 CoPaw 的 `memory/YYYY-MM-DD.md` 每日日志，但使用 SQLite 结构化存储，支持快速查询。
- **验收标准**：
  - 会话摘要可正确存取
  - 可按话题、时间查询历史会话
- **测试方法**：单元测试

---

### K2：MemoryRecordHook 增强 — LLM/Rule 双模式学习数据提取

- **目标**：增强 `MemoryRecordHook.after_message()`，从对话中自动提取学习数据并分发到各记忆存储。
- **修改文件**：
  - `src/agent/memory/enhancer.py`
- **实现内容**：
  - 给 `MemoryRecordHook` 注入 `error_memory`、`knowledge_map`、`session_memory`（当前只有 `student_profile` 和 `skill_memory`）
  - **LLM 提取模式** (`extraction_mode: "llm"`)：
    - 将对话摘要发给 LLM，返回结构化 JSON：
      ```json
      {
        "topics_discussed": ["TCP三次握手", "拥塞控制"],
        "weak_points_observed": ["不理解慢启动阈值"],
        "strong_points_observed": ["TCP报文格式熟练"],
        "user_preference": {"detail_level": "concise", "style": "exam_focused"},
        "key_questions": ["TCP为什么需要三次握手？"],
        "quiz_accuracy": 0.6,
        "summary": "本次主要复习了TCP运输层协议"
      }
      ```
  - **Rule 提取模式** (`extraction_mode: "rule"`)：
    - 用正则提取话题（匹配章节名、协议名等关键词）
    - 偏好检测：匹配"简洁点"→concise，"详细讲一下"→detailed，"考点"→exam_focused
    - 从 QuizEvaluatorTool 结果中提取正确率
  - **Both 模式** (`extraction_mode: "both"`)：LLM 优先，失败回退到 Rule
  - 提取后分发更新：
    - → `SessionMemory.save_session()` — 存储本次会话摘要
    - → `StudentProfile.update_profile()` — 更新偏好、弱点
    - → `KnowledgeMapMemory.update_mastery()` — 同步掌握度
- **验收标准**：
  - 会话结束后自动提取学习数据
  - LLM 失败时回退到 Rule 模式
  - 提取结果正确分发到各存储
- **测试方法**：构造模拟对话数据进行测试

---

### K3：StudentProfile preferences 结构化 + 偏好学习

- **目标**：结构化用户偏好字段，使系统能记住并应用"要简洁 / 要详细 / 要考点版"。
- **修改文件**：
  - `src/agent/memory/student_profile.py`
- **实现内容**：
  - 扩展 `StudentProfile.preferences` 默认结构：
    ```python
    preferences: dict = {
        "detail_level": "normal",    # concise / normal / detailed
        "style": "default",          # default / exam_focused / example_heavy
        "quiz_difficulty": "medium", # easy / medium / hard
    }
    ```
  - 在 K2 的提取逻辑中填充偏好值
  - 在 K6 的注入逻辑中将偏好翻译为 system prompt 指令
- **验收标准**：
  - 用户说"简洁点"后偏好被更新
  - 后续对话中 system prompt 包含偏好信息
- **测试方法**：单元测试

---

### K4：ReviewScheduleHook — 主动复习推荐

- **目标**：新建 `before_message` 钩子，在新会话开始时主动推荐复习内容，实现"你上次 TCP 掌握不好，今天复习一下"。
- **新建文件**：
  - `src/agent/hooks/review_schedule.py`
- **实现类/函数**：
  - `ReviewScheduleHook(LifecycleHook)`：
    - `__init__(knowledge_map, error_memory, session_memory)`
    - `async before_message(user_id, message) -> Optional[str]`
      1. 检测是否为新会话（conversation.messages 为空时首次触发）
      2. 调用 `knowledge_map.apply_decay(user_id)` — Ebbinghaus 衰减（K5 防重复机制）
      3. 调用 `knowledge_map.get_due_for_review(user_id)` — 到期复习节点
      4. 调用 `error_memory.get_errors(user_id, mastered=False, limit=3)` — 未掌握错题
      5. 调用 `session_memory.get_recent_sessions(user_id, limit=2)` — 上次话题
      6. 构建推荐文本注入 system prompt（不修改用户消息，通过新的机制传给 prompt_builder）
- **推荐文本格式**：
  ```
  ### 主动复习建议
  - 上次你学习了"TCP三次握手"，掌握度 0.3（薄弱），建议今天复习
  - 错题提醒：你在"计算子网掩码"上连续错了2次，需要巩固
  - 到期复习："ICMP协议"已3天未复习，掌握度可能衰减
  ```
- **验收标准**：
  - 新会话开始时自动检查并生成复习建议
  - 复习建议出现在 system prompt 中引导 Agent 行为
  - 没有到期项时不产生噪音
- **测试方法**：单元测试 + 手动多轮对话验证

---

### K5：StudentProfile 自动同步 + 修复

- **目标**：修复 `total_sessions` 覆写 bug，实现 `weak_topics`、`strong_topics`、`overall_accuracy` 自动同步。
- **修改文件**：
  - `src/agent/memory/enhancer.py`（`MemoryRecordHook.after_message`）
- **实现内容**：
  - **修复 total_sessions**：改为 `total_sessions: profile.total_sessions + 1` 累加
  - **同步 weak_topics**：从 `KnowledgeMapMemory.get_weak_nodes()` + `ErrorMemory.get_weak_concepts()` 合并去重
  - **同步 strong_topics**：从 `KnowledgeMapMemory` 中 mastery_level >= 0.8 的节点提取
  - **同步 overall_accuracy**：从 `KnowledgeMapMemory` 的 correct_count / quiz_count 加权平均
- **验收标准**：
  - `total_sessions` 正确累加
  - `weak_topics` 和 `strong_topics` 反映真实学习状态
- **测试方法**：单元测试

---

### K6：get_memory_summary 注入优化

- **目标**：优化 `MemoryContextEnhancer.get_memory_summary()` 的输出格式，使 system prompt 更聚焦、更个性化。
- **修改文件**：
  - `src/agent/memory/enhancer.py`
- **实现内容**：
  - 新增 `session_memory` 引用
  - 输出格式优化为分节结构：
    ```
    ## 学生记忆上下文
    ### 学习偏好
    - 回答风格: 简洁考点版
    - 难度偏好: 中等

    ### 上次学习
    - 话题: TCP三次握手, 拥塞控制
    - 关键问题: "TCP为什么需要三次握手？"
    - 掌握情况: TCP报文格式(强), 慢启动(弱)

    ### 需要复习的知识点
    - 慢启动阈值 (掌握度: 0.3, 已到复习时间)
    - ICMP协议 (掌握度: 0.4, 3天未复习)

    ### 错题提醒
    - 计算子网掩码 (错2次, 未掌握)
    ```
  - 加入 `session_memory.get_recent_sessions()` 信息
- **验收标准**：
  - system prompt 中的记忆上下文信息丰富、格式清晰
  - Agent 回答时能引用上次学习话题
- **测试方法**：打印 system prompt 检查

---

### K7：ContextEngineeringFilter 接入 Agent + Level 3 压缩

- **目标**：将已实现的 `ContextEngineeringFilter` 接入 Agent 的 `_build_llm_messages`，并实现 Level 3 LLM 压缩。
- **修改文件**：
  - `src/agent/agent.py`（`_build_llm_messages`）
  - `src/agent/memory/context_filter.py`（Level 3）
- **实现内容**：
  - **Agent 接入**：在 `_build_llm_messages` 中用 `ContextEngineeringFilter.filter_messages()` 替代简单的 `[-max_context_messages:]` 切片
  - **Level 3 LLM 压缩**（借鉴 CoPaw Compaction）：
    - 当消息数超过 `compaction_threshold_messages`（默认 30）时触发
    - 将旧消息压缩为一条 `[COMPACTED_SUMMARY]` 摘要
    - 保留最近 N 条消息原文
    - 摘要内容包含：学习目标、已解决问题、关键发现、下一步
  - 给 `ContextEngineeringFilter` 注入可选的 `LlmService` 用于 Level 3
- **验收标准**：
  - 短对话（< 30 条消息）不做任何处理
  - 长对话自动压缩，不丢失关键上下文
  - LLM 不可用时 fallback 到 Level 1-2
- **测试方法**：构造长对话消息列表进行测试

---

### K8：服务端接线 + 配置更新

- **目标**：将所有新组件注册到 FastAPI 应用中，更新配置文件。
- **修改文件**：
  - `src/server/app.py`
  - `config/settings.yaml`
- **实现内容**：
  - 初始化 `SessionMemory` 并注入到 `MemoryContextEnhancer` 和 `MemoryRecordHook`
  - 给 `MemoryRecordHook` 传入 `error_memory`、`knowledge_map`、`session_memory`、`llm_service`
  - 初始化 `ReviewScheduleHook` 并添加到 Agent 的 `hooks` 列表
  - 初始化 `ContextEngineeringFilter` 并注入到 Agent
  - settings.yaml 新增：
    ```yaml
    memory:
      extraction_mode: "both"  # llm / rule / both
      session_memory_enabled: true
      review_schedule_enabled: true
      decay_on_session_start: true
      compaction_enabled: true
      compaction_threshold_messages: 30
    ```
- **验收标准**：
  - 系统正常启动，所有新组件初始化成功
  - 配置项生效
- **测试方法**：启动服务器检查日志

---

### K9：端到端测试

- **目标**：验证记忆系统升级的完整效果。
- **测试流程**：
  1. 启动系统，进行第一轮对话：问 TCP 相关问题 + 做几道题
  2. 检查 `data/memory/` 下各 SQLite 数据库更新
  3. 开始第二轮对话（新会话）：观察是否出现主动复习建议
  4. 测试偏好记忆：在对话中说"简洁点"，检查后续是否生效
  5. 测试长对话压缩：发送 30+ 条消息，检查上下文压缩
- **验收标准**：
  - 新会话能看到"你上次学习了 TCP"的提示
  - 薄弱知识点被正确识别和推荐
  - 用户偏好被记住并影响回答风格
  - 长对话不丢失关键上下文
- **测试方法**：手动端到端测试 + 检查 SQLite 数据

---

### 阶段 L：前端 UI 优化 + 会话管理完善

> 将前端从简陋单页聊天升级为 ChatGPT 风格的现代 UI

---

### L1：Conversation 模型扩展

- **目标**：给 `Conversation` 模型添加 `title` 字段，支持侧边栏标题显示。
- **修改文件**：`src/agent/types.py`
- **实现**：`title: str = ""` 字段
- **验收标准**：现有对话可序列化/反序列化，title 默认为空字符串

---

### L2：会话列表与删除 API

- **目标**：新增 REST API 端点，支持前端侧边栏展示和管理会话。
- **修改文件**：
  - `src/agent/conversation.py` — `ConversationStore.delete()` 抽象方法 + 两个实现类
  - `src/server/routes.py` — 新增 `GET /api/conversations` 和 `DELETE /api/conversations/{id}`
- **API 规格**：
  - `GET /api/conversations?user_id=xxx&limit=50` → `[{id, title, updated_at, message_count}]`
  - `DELETE /api/conversations/{id}?user_id=xxx` → `{success: true}`
- **验收标准**：API 正确返回列表、删除后列表更新

---

### L3：会话标题自动生成

- **目标**：新会话的第一条用户消息自动截取前 30 字符作为标题。
- **修改文件**：`src/agent/agent.py`
- **实现**：在 `chat()` 中 append 用户消息后检查 `conversation.title`，为空则截取
- **输出**：`done` 事件 metadata 包含 `title` 字段
- **验收标准**：新对话自动生成标题，前端接收后更新侧边栏

---

### L4：HTML 结构重构

- **目标**：ChatGPT 风格两栏布局
- **重写文件**：`src/web/index.html`
- **布局**：
  - 左侧 `sidebar`：新建对话按钮 + 会话历史列表 + 底部上传/主题切换
  - 右侧 `main-panel`：顶部标题栏（含移动端汉堡菜单）+ 聊天区 + 输入区
  - 欢迎页面含快捷操作按钮（出题、复习、考点总结、帮助）
- **验收标准**：页面结构完整，语义化 HTML

---

### L5：CSS 全面重写

- **目标**：ChatGPT 配色 + 响应式设计
- **重写文件**：`src/web/style.css`
- **设计规范**：
  - 侧边栏：`#171717` 深灰背景，白色文字，`260px` 宽度
  - 聊天区：`#f7f7f8` 浅灰背景（dark: `#212121`）
  - 消息：用户/AI 头像（圆角方块）+ 分隔线风格
  - 输入框：圆角胶囊形，focus 时 primary 边框发光
  - 移动端：侧边栏 transform 折叠，overlay 遮罩
- **验收标准**：light/dark 主题均正常，移动端适配

---

### L6：JavaScript 逻辑增强

- **目标**：会话管理全功能
- **重写文件**：`src/web/app.js`
- **新增功能**：
  - 侧边栏会话列表：启动时调用 `GET /api/conversations` 加载
  - 新建对话：清空 `conversationId`，显示欢迎页
  - 切换对话：点击侧边栏项，加载历史消息渲染
  - 删除对话：调用 `DELETE` API，刷新列表
  - 标题更新：从 `done` 事件 metadata 读取 title
  - 快捷建议按钮：点击直接发送预设消息
  - 移动端侧边栏折叠/展开
  - 思考动画：三点跳动 indicator
- **验收标准**：多会话切换流畅，删除/新建正确

---

### L7：文档更新

- **目标**：更新 DEV_SPEC.md 阶段 L + 编写前端复习文档
- **文件**：
  - `DEV_SPEC.md`
  - `docs/FRONTEND_UI.md`（新建）
- **验收标准**：文档完整、面试可引用

---

## 6. 配置文件变更

### settings.yaml 新增段

```yaml
# =============================================================================
# Agent Configuration
# =============================================================================
agent:
  max_tool_iterations: 10
  stream_responses: true
  tool_timeout: 30
  max_context_messages: 40
  max_context_tokens: 8000
  temperature: 0.7
  system_prompt_path: "config/prompts/system_prompt.txt"
  skills_dir: "src/agent/skills/definitions"
  memory_enabled: true
  conversation_store_dir: "data/conversations"

# =============================================================================
# Server Configuration
# =============================================================================
server:
  host: "0.0.0.0"
  port: 8000
  cors_origins: ["*"]
  max_upload_size_mb: 50
  upload_dir: "data/uploads"

# =============================================================================
# Memory Configuration
# =============================================================================
memory:
  enabled: true
  db_dir: "data/memory"
  profile_enabled: true
  error_memory_enabled: true
  knowledge_map_enabled: true
  skill_memory_enabled: true
  decay_interval_hours: 24
```

### settings.yaml 新增段（阶段 J）

```yaml
# --- Vector Store 扩展 ---
vector_store:
  provider: "chroma"  # Options: chroma, milvus
  persist_directory: "./data/db/chroma"
  collection_name: "knowledge_hub"
  hnsw:
    space: "cosine"
    M: 32
    construction_ef: 200
    search_ef: 100
  milvus:
    uri: "./data/db/milvus.db"
    dim: 768

# --- Ingestion 扩展 ---
ingestion:
  splitter: "structure"  # Options: recursive, semantic, structure
  parent_child_enabled: true
  semantic_splitter:
    similarity_threshold: 0.5
    min_chunk_size: 100
    max_chunk_size: 1500
```

### pyproject.toml 新增依赖

```toml
dependencies = [
    # 现有依赖...
    "pyyaml>=6.0",
    "langchain-text-splitters>=0.3.0",
    "chromadb>=0.4.0",
    "mcp>=1.0.0",
    # 阶段 A-I 新增
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0",
    "sse-starlette>=1.0.0",
    "python-multipart>=0.0.5",
    "python-pptx>=0.6.21",
    "pydantic>=2.0.0",
    "openai>=1.0.0",
    "httpx>=0.24.0",
    # 阶段 J 新增
    "python-docx>=1.0.0",
    "pymilvus>=2.4.0",
    "lxml>=4.9.0",
]
```

---

## 7. 可扩展性与未来展望

| 方向 | 扩展方式 |
|------|---------|
| **新 LLM** | 实现 `LlmService` 子类，注册到 Factory |
| **新工具** | 实现 `Tool[T]` 子类，注册到 ToolRegistry |
| **新 Skill** | 在 `skills/definitions/` 下创建 SKILL.md |
| **新 Memory** | 实现新 Memory Store，注入 Enhancer |
| **新 Loader** | 实现 `BaseLoader` 子类，添加到 Pipeline |
| **新分块策略** | 实现 `BaseSplitter` 子类，注册到 SplitterFactory |
| **新向量数据库** | 实现 `BaseVectorStore` 子类，注册到 VectorStoreFactory |
| **新前端** | 调用 `/api/chat` SSE 端点即可 |
| **多课程** | 通过 collection 隔离不同课程的知识库 |
| **集群部署** | 将 VectorStore 切换到 Milvus Cluster，ConversationStore 和 Memory 迁移到 Redis/PostgreSQL |

---

## 阶段 M：RAG 深度优化 II — 面试无死角

> 目标：针对 RAG 系统逐一消除面试深度拷打中可能暴露的薄弱点，涵盖**中文分词、查询增强、检索质量、工程优化、评估体系**五大维度共 16 项任务。

### M 阶段总览

| 分组 | 任务 | 说明 | 状态 |
|------|------|------|------|
| **Group 1: 基础修复** | M1 | SparseEncoder + QueryProcessor 接入 jieba 中文分词 | ✅ 完成 |
| | M2 | SemanticSplitter batch embed bug 修复 | ✅ 完成 |
| | M3 | DocxLoader 死代码 + CrossEncoder timeout 修复 | ✅ 完成 |
| | M4 | BM25 索引 mtime 缓存 | ✅ 完成 |
| **Group 2: 查询增强** | M5 | QueryEnhancer — Query Rewriting (LLM 查询改写) | ✅ 完成 |
| | M6 | HyDE (Hypothetical Document Embedding) | ✅ 完成 |
| | M7 | Multi-Query Retrieval (查询分解) | ✅ 完成 |
| **Group 3: 检索质量** | M8 | MMR 多样性控制 | ✅ 完成 |
| | M9 | Reranker 集成到 HybridSearch 管线 | ✅ 完成 |
| | M10 | Contextual Retrieval (chunk 上下文注入) | ✅ 完成 |
| **Group 4: 工程优化** | M11 | Embedding LRU 缓存 | ✅ 完成 |
| | M12 | RRF 可配权重 (dense_weight / sparse_weight) | ✅ 完成 |
| | M13 | Chunk 语义去重 (SimHash) | ✅ 完成 |
| **Group 5: 评估体系** | M14 | 检索评估指标 (Hit Rate / MRR / NDCG / P@k / R@k) | ✅ 完成 |
| | M15 | Golden Test Set + 自动化评估脚本 | ✅ 完成 |
| | M16 | DEV_SPEC.md 阶段 M + 复习文档 | ✅ 完成 |

---

### M1: SparseEncoder + QueryProcessor 接入 jieba 中文分词

**问题**: `_tokenize` 使用 `re.findall(r'\b[\w-]+\b', text)` 无法正确切分中文。
**方案**: 引入 `jieba.cut` 对 CJK 文本进行分词，保留英文 regex fallback。
**文件**:
- `src/ingestion/embedding/sparse_encoder.py` — 重写 `_tokenize`，添加中文停用词表
- `src/core/query_engine/query_processor.py` — `_tokenize` 同步替换为 jieba
- `pyproject.toml` — 新增 `jieba>=0.42.1` 依赖
**测试**: `SparseEncoder.encode()` 对中文文本输出精确分词结果

---

### M2: SemanticSplitter batch embed bug 修复

**问题**: `_embed_batch` 实现为逐条 `embedder.embed(t)` 调用（返回单向量），但 `split_text` 期望返回向量列表。
**方案**: 改为 `embedder.embed(texts)` 批量调用。
**文件**: `src/libs/splitter/semantic_splitter.py`

---

### M3: DocxLoader 死代码 + CrossEncoder timeout 修复

**问题**:
1. `docx_loader.py` 行 213: `if False` 永远跳过一个分支，属于死代码
2. `cross_encoder_reranker.py` 声明了 `timeout` 参数但从未在 `_score_pairs` 中使用
**方案**:
1. 删除 `if False` 分支
2. 使用 `concurrent.futures.ThreadPoolExecutor` + `future.result(timeout=self.timeout)` 实现超时
**文件**:
- `src/libs/loader/docx_loader.py`
- `src/libs/reranker/cross_encoder_reranker.py`

---

### M4: BM25 索引 mtime 缓存

**问题**: `SparseRetriever._ensure_index_loaded` 每次查询都重新从磁盘加载 BM25 索引 JSON。
**方案**: 基于文件 mtime 的缓存，仅在文件变更时重新加载。
**文件**: `src/core/query_engine/sparse_retriever.py`

---

### M5-M7: 查询增强模块 (QueryEnhancer)

**新文件**: `src/core/query_engine/query_enhancer.py`

统一模块包含三种策略：

| 策略 | 说明 | Prompt |
|------|------|--------|
| M5: Query Rewriting | LLM 改写查询为更适合检索的形式 | `config/prompts/query_rewrite.txt` |
| M6: HyDE | 生成假设性文档，用其 embedding 替代原始查询 embedding | `config/prompts/hyde.txt` |
| M7: Multi-Query | 将复杂问题拆分为 2-4 个独立子查询 | `config/prompts/multi_query.txt` |

所有策略均为 `async def`，通过 `settings.retrieval.*_enabled` 开关控制。

---

### M8: MMR 多样性控制

**新文件**: `src/core/query_engine/mmr.py`
**算法**: `MMR = λ · sim(d, q) - (1-λ) · max sim(d, S)`
**参数**: `mmr_lambda` (0.7 = 偏向相关性，0.3 = 偏向多样性)
**集成**: 在 `HybridSearch` 融合后可选调用

---

### M9: Reranker 集成到 HybridSearch 管线

**修改**: `HybridSearch.__init__` 新增 `reranker: BaseReranker` 参数
**流程**: 在 fusion 之后、top_k 截断之前，若 `config.rerank_enabled=True` 则调用 `reranker.rerank()`
**容错**: reranker 失败时 fallback 到原始排序
**文件**: `src/core/query_engine/hybrid_search.py`

---

### M10: Contextual Retrieval (chunk 上下文注入)

**新文件**: `src/ingestion/transform/contextual_enricher.py`
**原理**: 参考 Anthropic "Contextual Retrieval"，为每个 chunk 添加文档级上下文前缀
**模式**:
- `rule`: 基于标题/文件名/节标题拼接（无 LLM 开销）
- `llm`: LLM 生成 1-2 句定位描述
**配置**: `settings.retrieval.contextual_enrichment: rule|llm|off`

---

### M11: Embedding LRU 缓存

**新文件**: `src/libs/embedding/cached_embedding.py`
**设计**: 透明代理模式包装 `BaseEmbedding`，SHA-256 哈希文本作为缓存 key
**容量**: `max_size=4096`（可通过 `settings.retrieval.embedding_cache_size` 配置）
**监控**: 每 200 次调用输出 hit/miss 统计

---

### M12: RRF 可配权重

**修改**: `HybridSearch._fuse_results` 改用 `fusion.fuse_with_weights(weights=[dense_weight, sparse_weight])`
**配置**: `settings.retrieval.dense_weight / sparse_weight`
**文件**: `src/core/query_engine/hybrid_search.py`

---

### M13: Chunk 语义去重 (SimHash)

**新文件**: `src/ingestion/transform/chunk_dedup.py`
**算法**: 64-bit SimHash + Hamming 距离阈值 (默认 3)
**集成点**: Ingestion Pipeline 在 chunking 后、encoding 前去重
**配置**: `settings.retrieval.dedup_enabled / dedup_threshold`

---

### M14: 检索评估指标

**新文件**: `src/libs/evaluator/retrieval_metrics.py`
**指标**: Hit Rate, MRR, NDCG@k, Precision@k, Recall@k
**接口**: 实现 `BaseEvaluator`，可直接注入 `EvalRunner`

---

### M15: Golden Test Set + 自动化评估脚本

**新文件**:
- `tests/fixtures/golden_test_set.json` — 10 条计算机网络知识点测试用例
- `scripts/run_eval.py` — CLI 脚本，初始化 HybridSearch + RetrievalMetricsEvaluator，运行评估并输出 JSON 报告

**用法**: `python scripts/run_eval.py --top-k 10 --output data/eval_report.json`

---

### M16: 文档输出

- 更新 `DEV_SPEC.md` — 阶段 M 详细 spec（本节）
- 新建 `docs/RAG_DEEP_OPTIMIZATION_II.md` — 复习与面试话术文档

### settings.yaml 新增配置项

```yaml
retrieval:
  dense_weight: 1.0          # RRF 中 dense 路权重
  sparse_weight: 1.0         # RRF 中 sparse 路权重
  rerank_enabled: false       # 是否启用 Reranker
  rerank_top_k: 5            # Reranker 返回 top-k
  mmr_enabled: false          # 是否启用 MMR 多样性控制
  mmr_lambda: 0.7            # MMR λ 参数
  query_rewrite_enabled: false
  hyde_enabled: false
  multi_query_enabled: false
  contextual_enrichment: "rule"  # rule / llm / off
  embedding_cache_size: 4096
  dedup_enabled: true
  dedup_threshold: 3          # SimHash Hamming 距离阈值
```
