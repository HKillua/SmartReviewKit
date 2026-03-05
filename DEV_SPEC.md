# Developer Specification (DEV_SPEC)

# Database Course Agent — 数据库课程智能学习助手

> 基于 RAG + Memory + Skill 的 ReAct Agent，参考 Vanna.ai v2.0 架构  
> 系统设计详见：[docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md)

---

## 1. 项目概述

### 1.1 项目定位

本项目在现有 MODULAR-RAG-MCP-SERVER 的 RAG 基础设施之上，构建一个面向数据库课程的智能学习 Agent。Agent 具备考点复习、习题生成与评判、智能问答、动态知识库管理四大核心能力，并通过 Memory 记忆系统和 Skill 技能系统实现个性化学习体验。

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
| 向量存储 | ChromaDB (复用) | >= 0.4.0 |
| PPT 解析 | python-pptx | latest |
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
│   │   └── pptx_loader.py            # [E2] 新增：PPT Loader
│   │
│   ├── core/                          # 复用现有
│   ├── ingestion/                     # 复用现有
│   ├── libs/                          # 复用现有
│   ├── mcp_server/                    # 复用现有
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
│   │   └── test_skill_workflow.py     # [G2]
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
| **总计** | **29** | **0** | **0%** |

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

### pyproject.toml 新增依赖

```toml
dependencies = [
    # 现有依赖...
    "pyyaml>=6.0",
    "langchain-text-splitters>=0.3.0",
    "chromadb>=0.4.0",
    "mcp>=1.0.0",
    # 新增依赖
    "fastapi>=0.100.0",
    "uvicorn>=0.20.0",
    "sse-starlette>=1.0.0",
    "python-multipart>=0.0.5",
    "python-pptx>=0.6.21",
    "pydantic>=2.0.0",
    "openai>=1.0.0",
    "httpx>=0.24.0",
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
| **新前端** | 调用 `/api/chat` SSE 端点即可 |
| **多课程** | 通过 collection 隔离不同课程的知识库 |
| **集群部署** | 将 ConversationStore 和 Memory 迁移到 Redis/PostgreSQL |
