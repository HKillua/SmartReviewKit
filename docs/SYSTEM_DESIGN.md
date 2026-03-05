# Database Course Agent - 系统设计文档

> **项目定位**：基于 RAG + Memory + Skill 的数据库课程智能学习助手  
> **架构参考**：Vanna.ai v2.0 Agent 架构 + Mem0 长期记忆 + Agent Skill 渐进式披露  
> **版本**：v1.1  
> **日期**：2026-03-04

---

## 目录

1. [目标与约束](#1-目标与约束)
2. [模块切分](#2-模块切分)
3. [数据流](#3-数据流)
4. [稳定性设计](#4-稳定性设计)
5. [多租户隔离](#5-多租户隔离)
6. [指标与演进路线](#6-指标与演进路线)
7. [Memory 记忆系统设计](#7-memory-记忆系统设计) **(核心亮点)**
8. [Skill 技能系统设计](#8-skill-技能系统设计) **(核心亮点)**

---

## 1. 目标与约束

### 1.1 业务目标

| 目标 | 说明 |
|------|------|
| **考点复习** | 用户输入章节/主题，Agent 自动检索知识库并生成结构化考点摘要 |
| **习题生成** | 基于知识库内容生成选择题、填空题、简答题、SQL 题，支持难度分级 |
| **答案评判** | 用户提交答案后，Agent 判断对错并给出详细解析 |
| **智能问答** | 针对课程任意内容进行自由问答，回答带知识库引用 |
| **动态知识库** | 支持上传新 PPT/PDF 自动入库，增量更新知识库 |

### 1.2 性能约束

| 指标 | 目标值 | 说明 |
|------|--------|------|
| **首字延迟 (TTFT)** | < 2s | 用户发消息到看到第一个字符 |
| **端到端延迟 (P95)** | < 15s | 完整回复（含检索+LLM 推理） |
| **检索延迟** | < 500ms | HybridSearch 单次查询 |
| **文档入库** | < 60s/文档 | 单个 PPT/PDF 从上传到可检索 |
| **并发用户** | 10-50 | 面试演示 + 小规模教学场景 |
| **知识库规模** | 1-100 篇课件 | 约 1000-10000 个 chunk |

### 1.3 SLA 定义

```
┌─────────────────────────────────────────────────────┐
│  SLA Tier: Development / Demo                       │
│                                                     │
│  可用性目标:  99% (允许每月 ~7h 停机)                 │
│  数据持久性:  本地持久化, 无跨区复制                    │
│  恢复目标:    RTO < 30min, RPO < 1h                  │
│  降级策略:    LLM 不可用时返回纯检索结果               │
└─────────────────────────────────────────────────────┘
```

### 1.4 合规约束

| 约束 | 措施 |
|------|------|
| **API Key 安全** | 环境变量注入，禁止硬编码，`.env` 在 `.gitignore` 中 |
| **用户数据** | 会话数据本地存储，不外传；LLM 调用仅传必要上下文 |
| **审计日志** | 记录每次工具调用、LLM 请求、文件上传操作 |
| **内容安全** | 输入/输出内容过滤（通过 LifecycleHook 实现） |

### 1.5 预算约束

| 资源 | 预算策略 |
|------|---------|
| **LLM API** | 可插拔设计：演示用 Ollama 本地（零成本），正式用 OpenAI/DeepSeek |
| **Embedding** | 优先本地模型（Ollama），备选 Azure text-embedding-ada-002 |
| **向量存储** | ChromaDB 本地持久化（零成本） |
| **部署** | 单机 Docker Compose，无云服务费用 |

---

## 2. 模块切分

### 2.1 架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Web Frontend (SPA)                            │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│   │ Chat UI  │  │ Quiz UI  │  │ Upload   │  │ Review Summary   │   │
│   └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
└───────────────────────────┬──────────────────────────────────────────┘
                            │ SSE / REST
┌───────────────────────────▼──────────────────────────────────────────┐
│                     FastAPI Server Layer                              │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│   │ POST /chat   │  │ POST /upload │  │ GET /health  │              │
│   │   (SSE)      │  │  (multipart) │  │              │              │
│   └──────┬───────┘  └──────┬───────┘  └──────────────┘              │
│          │                  │                                        │
│   ┌──────▼───────┐  ┌──────▼───────┐                                │
│   │ ChatHandler  │  │ UploadHandler│                                │
│   └──────┬───────┘  └──────┬───────┘                                │
└──────────┼─────────────────┼────────────────────────────────────────┘
           │                 │
┌──────────▼─────────────────▼────────────────────────────────────────┐
│                        Agent Core                                    │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │  Agent                                                       │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │   │
│   │  │ ToolRegistry │  │ LlmService   │  │ ConversationStore│  │   │
│   │  └──────┬───────┘  └──────┬───────┘  └──────────────────┘  │   │
│   │         │                 │                                  │   │
│   │  ┌──────▼───────────────────────────────────────────────┐   │   │
│   │  │              Tool Loop (ReAct)                        │   │   │
│   │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │   │   │
│   │  │  │ Thought  │→│ Action  │→│ Observe │→ (loop)       │   │   │
│   │  │  └─────────┘  └─────────┘  └─────────┘              │   │   │
│   │  └──────────────────────────────────────────────────────┘   │   │
│   │                                                              │   │
│   │  Extension Points:                                           │   │
│   │  ┌────────────┐ ┌────────────┐ ┌────────────┐              │   │
│   │  │ Lifecycle  │ │ Middleware │ │ Context    │              │   │
│   │  │ Hooks      │ │            │ │ Enrichers  │              │   │
│   │  └────────────┘ └────────────┘ └────────────┘              │   │
│   └─────────────────────────────────────────────────────────────┘   │
└──────────┬──────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────────┐
│                     Agent Tools Layer                                 │
│                                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│   │ Knowledge    │  │ Review       │  │ QuizGenerator            │ │
│   │ QueryTool    │  │ SummaryTool  │  │ Tool                     │ │
│   └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘ │
│          │                 │                      │                  │
│   ┌──────────────┐  ┌──────────────┐                                │
│   │ QuizEvaluator│  │ Document     │                                │
│   │ Tool         │  │ IngestTool   │                                │
│   └──────┬───────┘  └──────┬───────┘                                │
└──────────┼─────────────────┼────────────────────────────────────────┘
           │                 │
┌──────────▼─────────────────▼────────────────────────────────────────┐
│                RAG Infrastructure (Existing - Reuse)                  │
│                                                                      │
│   ┌───────────────────────────────────────────────┐                 │
│   │  Retrieval Engine                              │                 │
│   │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ │                 │
│   │  │ Dense  │ │ Sparse │ │ RRF    │ │Reranker│ │                 │
│   │  │Retriever│ │Retriever│ │Fusion │ │        │ │                 │
│   │  └────────┘ └────────┘ └────────┘ └────────┘ │                 │
│   └───────────────────────────────────────────────┘                 │
│                                                                      │
│   ┌───────────────────────────────────────────────┐                 │
│   │  Ingestion Pipeline                            │                 │
│   │  Load → Chunk → Transform → Encode → Store     │                 │
│   │  ┌────────┐ ┌────────┐                         │                 │
│   │  │PdfLoad │ │PptxLoad│  (New)                  │                 │
│   │  └────────┘ └────────┘                         │                 │
│   └───────────────────────────────────────────────┘                 │
│                                                                      │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             │
│   │ ChromaDB │ │ BM25 Idx │ │ Embedding│ │ Settings │             │
│   └──────────┘ └──────────┘ └──────────┘ └──────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块职责矩阵

| 模块 | 职责 | 对外接口 | 依赖 |
|------|------|---------|------|
| **Web Frontend** | 聊天 UI、习题交互、文件上传 | SSE/REST 调用 Server | 无后端依赖 |
| **FastAPI Server** | HTTP 路由、SSE 流式、文件接收、CORS | `ChatHandler.handle_stream()` | Agent Core |
| **ChatHandler** | 请求解析、流式包装、错误格式化 | `Agent.chat()` | Agent |
| **Agent** | ReAct 工具循环、会话管理、系统提示 | `chat(msg, conv_id) -> AsyncGen[StreamEvent]` | LlmService, ToolRegistry, ConversationStore |
| **LlmService** | LLM 调用抽象（含 Tool Calling + Streaming） | `send_request()`, `stream_request()` | OpenAI/DeepSeek/Ollama SDK |
| **ToolRegistry** | 工具注册、权限校验、参数验证、执行分发 | `register()`, `execute()`, `get_schemas()` | Tool 实例 |
| **KnowledgeQueryTool** | 知识库混合检索 | `execute(query, top_k)` | HybridSearch |
| **ReviewSummaryTool** | 考点复习生成 | `execute(topic, chapter)` | HybridSearch, LlmService |
| **QuizGeneratorTool** | 习题生成 | `execute(topic, type, count, difficulty)` | HybridSearch, LlmService |
| **QuizEvaluatorTool** | 答案评判 | `execute(question, user_answer, ref_answer)` | LlmService |
| **DocumentIngestTool** | 文档入库 | `execute(file_path, collection)` | IngestionPipeline |
| **IngestionPipeline** | 文档处理全流程 | `run(file_path)` | Loaders, Chunker, Encoders, Storage |
| **PptxLoader** | PPT 解析 | `load(file_path) -> Document` | python-pptx |
| **ConversationStore** | 会话持久化 | CRUD 接口 | 文件系统 / SQLite |
| **LifecycleHooks** | 请求前后拦截 | `before_message()`, `after_tool()` 等 | - |
| **LlmMiddleware** | LLM 请求拦截 | `before_llm_request()`, `after_llm_response()` | - |

### 2.3 模块层级与依赖规则

```
Layer 0  [Web Frontend]        → 只依赖 Server HTTP API
Layer 1  [FastAPI Server]      → 只依赖 Agent Core
Layer 2  [Agent Core]          → 依赖 LlmService + ToolRegistry + ConversationStore
Layer 3  [Agent Tools]         → 依赖 RAG Infrastructure + LlmService
Layer 4  [RAG Infrastructure]  → 依赖 Libs (Embedding, VectorStore, Loader)
Layer 5  [Libs]                → 依赖外部 SDK (OpenAI, ChromaDB, python-pptx)
```

**依赖规则**：上层可依赖下层，下层不可依赖上层。同层模块通过接口交互，不直接引用实现。

---

## 3. 数据流

### 3.1 核心请求流：用户对话

```
用户 → Web UI → FastAPI → ChatHandler → Agent → [Tool Loop] → 流式响应 → Web UI

详细步骤：

1. [Web UI]     用户输入 "帮我总结第3章考点，出2道选择题"
                 │
                 ▼ POST /api/chat (SSE)
2. [FastAPI]    解析请求，构建 ChatRequest
                 │  {message, conversation_id, user_id}
                 ▼
3. [ChatHandler] 创建 RequestContext，调用 Agent
                 │
                 ▼
4. [Agent]       ┌─ 加载/创建 Conversation
                 │  添加 user message
                 │
                 ├─ 构建 SystemPrompt（角色设定 + 课程上下文）
                 │
                 ├─ 获取 ToolSchemas（5 个工具的 JSON Schema）
                 │
                 ├─ 调用 LLM (with tools)
                 │  ← LLM 返回: tool_call(KnowledgeQueryTool, {query: "第3章"})
                 │
                 ├─ yield StreamEvent(type="tool_start", tool="KnowledgeQueryTool")
                 │
                 ├─ 执行 KnowledgeQueryTool
                 │  │ → HybridSearch.search("第3章", top_k=10)
                 │  │   → DenseRetriever (embedding → ChromaDB)
                 │  │   → SparseRetriever (BM25)
                 │  │   → RRF Fusion
                 │  │   → Reranker (optional)
                 │  │ ← 返回 10 个 RetrievalResult
                 │  ← ToolResult(result_for_llm="第3章相关内容...")
                 │
                 ├─ yield StreamEvent(type="tool_result", summary="检索到10条结果")
                 │
                 ├─ 将 tool result 反馈 LLM
                 │  ← LLM 返回: tool_call(ReviewSummaryTool, {topic: "第3章"})
                 │
                 ├─ 执行 ReviewSummaryTool
                 │  │ → 将检索结果 + prompt 发送 LLM 生成考点摘要
                 │  ← ToolResult(result_for_llm="考点: 1. 关系模型...")
                 │
                 ├─ 将 tool result 反馈 LLM
                 │  ← LLM 返回: tool_call(QuizGeneratorTool, {topic: "第3章", type: "选择题", count: 2})
                 │
                 ├─ 执行 QuizGeneratorTool
                 │  │ → 将检索结果 + prompt 发送 LLM 生成题目
                 │  ← ToolResult(result_for_llm="题目JSON...")
                 │
                 ├─ 将 tool result 反馈 LLM
                 │  ← LLM 返回: 最终文本回复（综合考点 + 题目）
                 │
                 ├─ yield StreamEvent(type="text_delta", content="## 第3章考点总结\n...")
                 │  ... (流式逐字输出)
                 │
                 └─ 保存 Conversation
                         │
                         ▼
5. [FastAPI]    SSE 逐条发送 StreamEvent
                         │
                         ▼
6. [Web UI]     实时渲染 Markdown + 题目卡片
```

### 3.2 文件上传流：文档入库

```
用户 → Web UI → FastAPI → IngestionPipeline → 知识库更新

详细步骤：

1. [Web UI]     拖拽上传 "第4章-事务管理.pptx"
                 │
                 ▼ POST /api/upload (multipart/form-data)
2. [FastAPI]    接收文件，保存到 data/uploads/
                 │  校验文件类型 (.pdf, .pptx)
                 │  校验文件大小 (< 50MB)
                 │
                 ▼ 异步任务
3. [Pipeline]   ┌─ SHA256 完整性检查（跳过已处理）
                 │
                 ├─ 选择 Loader
                 │  .pptx → PptxLoader
                 │  .pdf  → PdfLoader
                 │
                 ├─ Load → Document (Markdown + 图片占位符)
                 │
                 ├─ Chunk → 1000 字/块, 200 字重叠
                 │
                 ├─ Transform
                 │  → ChunkRefiner (规则/LLM)
                 │  → MetadataEnricher (标题/标签/摘要)
                 │  → ImageCaptioner (Vision LLM, optional)
                 │
                 ├─ Encode
                 │  → DenseEncoder (text-embedding-ada-002)
                 │  → SparseEncoder (BM25 词频)
                 │
                 └─ Store
                    → ChromaDB (dense vectors)
                    → BM25 Index (sparse)
                    → ImageStorage (图片索引)

4. [Response]   返回入库结果
                 {success: true, chunks: 42, images: 8}
```

### 3.3 上下文构建流

```
                    SystemPrompt 构建流程
                    ━━━━━━━━━━━━━━━━━━━

┌─────────────────────────────────────────────────────────┐
│  Base System Prompt (角色设定)                            │
│                                                         │
│  "你是一位数据库课程学习助手。你的任务是帮助学生         │
│   理解数据库原理，提供考点总结和练习题..."                │
│                                                         │
│  + 课程元信息（已入库章节列表、文档数量）                 │
│  + 可用工具描述                                          │
│  + 输出格式约束（Markdown、题目 JSON Schema）             │
│                                                         │
│  ┌────────────────────────────────────────────────┐     │
│  │ LlmContextEnhancer (可选)                      │     │
│  │ → 搜索 AgentMemory 中相关上下文                │     │
│  │ → 注入 "Relevant Context from Memory" 段        │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  ┌────────────────────────────────────────────────┐     │
│  │ Conversation History (最近 N 轮)               │     │
│  │ → ConversationFilter 裁剪过长历史              │     │
│  │ → 保留 system + 最近 20 条 user/assistant       │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  ┌────────────────────────────────────────────────┐     │
│  │ Tool Schemas (JSON Schema)                      │     │
│  │ → 5 个工具的 name + description + parameters    │     │
│  └────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### 3.4 工具执行内部流

```
Agent 主循环
━━━━━━━━━━

while iterations < max_tool_iterations (10):
    │
    ├─ [Middleware] before_llm_request(request)
    │   → 可注入: 缓存检查、token 计数、请求日志
    │
    ├─ [LlmService] send_request(request) / stream_request(request)
    │   → 调用 OpenAI/DeepSeek/Ollama API
    │   → 返回 LlmResponse {content, tool_calls}
    │
    ├─ [Middleware] after_llm_response(request, response)
    │   → 可注入: 响应缓存、成本记录、内容过滤
    │
    ├─ if response.tool_calls:
    │   │
    │   ├─ for each tool_call:
    │   │   │
    │   │   ├─ [LifecycleHook] before_tool(tool, context)
    │   │   │   → 可注入: 配额检查、参数审计
    │   │   │
    │   │   ├─ [ToolRegistry] execute(tool_call, context)
    │   │   │   ├─ 权限校验 (access_groups ∩ user.groups)
    │   │   │   ├─ Pydantic 参数校验
    │   │   │   ├─ transform_args (可做 RLS/过滤)
    │   │   │   └─ tool.execute(context, args) → ToolResult
    │   │   │
    │   │   ├─ [LifecycleHook] after_tool(result)
    │   │   │   → 可注入: 结果审计、敏感信息脱敏
    │   │   │
    │   │   └─ yield StreamEvent(tool_result)
    │   │
    │   ├─ 将所有 ToolResult 添加到消息历史
    │   └─ 重建 LlmRequest，继续循环
    │
    └─ else (无 tool_calls):
        ├─ yield StreamEvent(text_delta, content)  (流式)
        └─ break
```

### 3.5 数据写回路径

| 写回目标 | 触发时机 | 数据内容 |
|---------|---------|---------|
| **ConversationStore** | 每轮对话结束 | 完整消息历史（user + assistant + tool_calls + tool_results） |
| **ChromaDB + BM25** | DocumentIngestTool 执行 | 新文档的向量和稀疏索引 |
| **AuditLog** | 每次工具调用 | 用户 ID、工具名、参数、结果、耗时 |
| **Trace** | 每次请求 | 各阶段耗时、检索结果、LLM token 用量 |

---

## 4. 稳定性设计

### 4.1 总体策略

```
┌────────────────────────────────────────────────────────┐
│                   稳定性防护层级                         │
│                                                        │
│  L1  入口层   │ 限流 → 超时 → 输入校验                  │
│  L2  Agent层  │ 工具循环上限 → 工具超时 → 错误恢复       │
│  L3  LLM层   │ 重试 → 降级 → 熔断                      │
│  L4  存储层   │ 连接池 → 失败重试 → 只读降级             │
└────────────────────────────────────────────────────────┘
```

### 4.2 限流 (Rate Limiting)

通过 `LifecycleHook.before_message()` 实现：

```python
class RateLimitHook(LifecycleHook):
    """基于令牌桶的限流，per-user 粒度"""

    def __init__(self, max_requests_per_minute: int = 20):
        self.limiter = {}  # user_id -> TokenBucket

    async def before_message(self, user, message):
        bucket = self.limiter.setdefault(user.id, TokenBucket(rate=20, capacity=20))
        if not bucket.consume(1):
            raise RateLimitExceeded(
                f"请求过于频繁，请在 {bucket.next_available_in_ms}ms 后重试"
            )
        return None
```

| 维度 | 限制 | 实现方式 |
|------|------|---------|
| 对话请求 | 20 次/分钟/用户 | TokenBucket in LifecycleHook |
| 文件上传 | 5 次/分钟/用户 | FastAPI Depends |
| 单文件大小 | ≤ 50MB | FastAPI UploadFile 校验 |
| LLM Token | 10000 tokens/请求 | AgentConfig.max_tokens |

### 4.3 超时 (Timeout)

| 层级 | 超时值 | 实现 |
|------|--------|------|
| HTTP 请求 | 120s | FastAPI/Uvicorn timeout |
| 单次 LLM 调用 | 60s | httpx.Client(timeout=60) |
| 单次工具执行 | 30s | asyncio.wait_for(tool.execute(), timeout=30) |
| 文档入库 | 300s | Pipeline 级别 |
| Agent 工具循环 | max_tool_iterations=10 | AgentConfig 配置 |

```python
# Agent 中的工具执行超时
async def _execute_tool_with_timeout(self, tool_call, context, timeout=30):
    try:
        return await asyncio.wait_for(
            self.tool_registry.execute(tool_call, context),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        return ToolResult(
            success=False,
            error=f"工具 {tool_call.name} 执行超时 ({timeout}s)",
            result_for_llm=f"Tool {tool_call.name} timed out after {timeout}s."
        )
```

### 4.4 重试 (Retry)

通过 `LlmMiddleware` + `ErrorRecoveryStrategy` 实现：

```python
class RetryMiddleware(LlmMiddleware):
    """LLM 调用自动重试"""

    def __init__(self, max_retries=3, backoff_base=1.0):
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    async def before_llm_request(self, request):
        request.metadata["retry_count"] = 0
        return request

    async def after_llm_response(self, request, response):
        if response.error and request.metadata["retry_count"] < self.max_retries:
            retry = request.metadata["retry_count"] + 1
            delay = self.backoff_base * (2 ** (retry - 1))  # 指数退避
            await asyncio.sleep(delay)
            request.metadata["retry_count"] = retry
            # 触发重试（通过抛出特定异常让 Agent 重发）
        return response
```

| 错误类型 | 重试策略 | 最大次数 |
|---------|---------|---------|
| LLM API 429 (Rate Limit) | 指数退避 1s, 2s, 4s | 3 次 |
| LLM API 500/502/503 | 指数退避 | 3 次 |
| LLM API 超时 | 立即重试 | 2 次 |
| 工具执行失败 | 不重试，返回错误给 LLM 决定 | 0 次 |
| Embedding API 失败 | 指数退避 | 3 次 |
| ChromaDB 连接失败 | 固定间隔 2s | 3 次 |

### 4.5 熔断 (Circuit Breaker)

```python
class CircuitBreaker:
    """三态熔断器：CLOSED → OPEN → HALF_OPEN"""

    States: CLOSED, OPEN, HALF_OPEN

    ┌────────┐  failure_count >= threshold  ┌────────┐
    │ CLOSED │ ──────────────────────────→ │  OPEN  │
    │(正常)  │                              │(拒绝)  │
    └────────┘ ←────────────────────────── └────────┘
                  probe success               │ cooldown 到期
                                              ▼
                                         ┌──────────┐
                                         │HALF_OPEN │
                                         │(探测)    │
                                         └──────────┘

    配置：
    - failure_threshold: 5（连续 5 次失败触发熔断）
    - cooldown_seconds: 30（熔断后 30s 进入半开状态）
    - half_open_max_calls: 1（半开状态最多放行 1 个请求）
```

应用位置：

| 熔断器 | 保护目标 | 降级行为 |
|--------|---------|---------|
| `llm_circuit_breaker` | LLM API | 返回 "AI 服务暂时不可用，请稍后重试" |
| `embedding_circuit_breaker` | Embedding API | 使用缓存结果或返回空检索 |

### 4.6 降级 (Graceful Degradation)

```
┌──────────────────────────────────────────────────────┐
│              降级策略矩阵                              │
│                                                      │
│  组件失败        │ 降级行为                            │
│  ───────────────┼──────────────────────────────────  │
│  LLM 不可用     │ 返回纯检索结果（无总结/出题能力）     │
│  Embedding 不可用│ 仅用 BM25 稀疏检索（精度下降）       │
│  Reranker 不可用 │ 跳过重排，使用 RRF 融合分数          │
│  ChromaDB 异常   │ 返回 "知识库暂时不可用" 提示          │
│  Vision LLM 不可用│ 跳过图片描述，保留文本内容           │
│  文件上传失败    │ 返回错误 + 重试指引                   │
└──────────────────────────────────────────────────────┘
```

实现方式（参考现有项目的 graceful degradation 模式）：

```python
# ReviewSummaryTool 降级示例
async def execute(self, context: ToolContext, args: ReviewSummaryArgs) -> ToolResult:
    # 1. 检索知识库内容
    results = await self._search(args.topic)
    if not results:
        return ToolResult(success=False, error="未找到相关内容")

    # 2. 尝试 LLM 生成摘要
    try:
        summary = await self.llm_service.generate_summary(results)
        return ToolResult(success=True, result_for_llm=summary)
    except LlmUnavailableError:
        # 降级：返回原始检索结果（无 LLM 总结）
        fallback = self._format_raw_results(results)
        return ToolResult(
            success=True,
            result_for_llm=f"[AI总结暂不可用，以下为原始检索结果]\n{fallback}",
            metadata={"degraded": True}
        )
```

---

## 5. 多租户隔离

### 5.1 隔离架构

```
┌─────────────────────────────────────────────────────────┐
│                    多租户隔离模型                         │
│                                                         │
│   ┌──────────────────────────────────────────────────┐  │
│   │  Layer 1: 身份识别                                │  │
│   │  UserResolver → 从 Request 中提取/验证用户身份     │  │
│   │  支持: Cookie, JWT, API Key, Session              │  │
│   └──────────────────────────────────────────────────┘  │
│                                                         │
│   ┌──────────────────────────────────────────────────┐  │
│   │  Layer 2: 数据隔离                                │  │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │  │
│   │  │会话隔离   │ │文件隔离   │ │知识库隔离          │ │  │
│   │  │per-user   │ │per-user   │ │per-collection    │ │  │
│   │  │conversation│ │upload dir │ │(可选 per-tenant) │ │  │
│   │  └──────────┘ └──────────┘ └──────────────────┘ │  │
│   └──────────────────────────────────────────────────┘  │
│                                                         │
│   ┌──────────────────────────────────────────────────┐  │
│   │  Layer 3: 权限控制                                │  │
│   │  Tool.access_groups ∩ User.group_memberships      │  │
│   │  ┌──────────────┐  ┌─────────────────────────┐   │  │
│   │  │ student 组   │  │ admin 组                 │   │  │
│   │  │ - 检索       │  │ - 检索 + 入库 + 管理     │   │  │
│   │  │ - 复习       │  │ - 系统配置               │   │  │
│   │  │ - 习题       │  │ - 审计日志查看           │   │  │
│   │  │ - 问答       │  │                          │   │  │
│   │  └──────────────┘  └─────────────────────────┘   │  │
│   └──────────────────────────────────────────────────┘  │
│                                                         │
│   ┌──────────────────────────────────────────────────┐  │
│   │  Layer 4: 审计日志                                │  │
│   │  每次操作记录: user_id, action, params, result,    │  │
│   │  timestamp, duration, ip_address                  │  │
│   └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 5.2 租户级配置

```python
# 配置结构 (config/settings.yaml 扩展)
tenants:
  default:
    llm_provider: "openai"
    llm_model: "gpt-4o"
    max_tokens_per_request: 10000
    collections: ["database_course"]
    rate_limit: 20  # requests/minute

  premium:
    llm_provider: "openai"
    llm_model: "gpt-4o"
    max_tokens_per_request: 20000
    collections: ["database_course", "advanced_db"]
    rate_limit: 60

# User → Tenant 映射
users:
  alice:
    tenant: "default"
    groups: ["student"]
  bob:
    tenant: "premium"
    groups: ["student", "admin"]
```

### 5.3 密钥隔离

| 密钥类型 | 隔离方式 | 存储位置 |
|---------|---------|---------|
| LLM API Key | 全局共享（单租户场景） / 租户级（多租户场景） | 环境变量 / Vault |
| Embedding API Key | 全局共享 | 环境变量 |
| 用户 Session Token | Per-user，HttpOnly Cookie | 浏览器 |
| Admin API Key | 独立管理接口 | 环境变量 |

### 5.4 数据隔离

```python
# ConversationStore: 按 user_id 隔离
class FileConversationStore(ConversationStore):
    def _get_user_dir(self, user: User) -> Path:
        return self.base_dir / hashlib.sha256(user.id.encode()).hexdigest()[:12]

    async def get_conversation(self, conversation_id, user):
        path = self._get_user_dir(user) / f"{conversation_id}.json"
        # 用户只能访问自己目录下的会话

# 文件上传: 按 user_id 隔离
upload_dir = f"data/uploads/{user.id}/"

# 知识库: 按 collection 隔离 (所有用户共享课程知识库)
# 但 Agent Memory 按 user_id 隔离（每个学生的学习记忆独立）
```

### 5.5 审计日志

```python
class AuditEvent(BaseModel):
    timestamp: datetime
    user_id: str
    action: str           # "tool_call", "llm_request", "file_upload", "login"
    tool_name: Optional[str]
    parameters: dict      # 脱敏后的参数
    result_summary: str   # 成功/失败 + 摘要
    duration_ms: float
    ip_address: Optional[str]
    metadata: dict

# 审计日志实现（写入 JSONL 文件，可后续接入 ELK）
class FileAuditLogger(AuditLogger):
    async def log_event(self, event: AuditEvent):
        with open(self.log_path, "a") as f:
            f.write(event.model_dump_json() + "\n")
```

审计覆盖范围：

| 事件 | 记录内容 |
|------|---------|
| 用户登录 | user_id, ip, timestamp |
| 对话请求 | user_id, message (截断), conversation_id |
| 工具调用 | tool_name, args (脱敏), result_summary, duration |
| LLM 请求 | model, token_count, cost_estimate, duration |
| 文件上传 | filename, size, collection, ingestion_result |
| 权限拒绝 | user_id, tool_name, required_groups, user_groups |

---

## 6. 指标与演进路线

### 6.1 核心监控指标

#### 6.1.1 业务指标

| 指标 | 采集方式 | 告警阈值 |
|------|---------|---------|
| 日活用户数 (DAU) | 审计日志统计 | - |
| 对话轮次/用户 | ConversationStore 统计 | - |
| 习题正确率 | QuizEvaluatorTool 结果聚合 | - |
| 知识库文档数 | ChromaDB collection stats | - |
| 用户满意度 | (可选) 拇指点赞/踩 | < 70% 满意触发 review |

#### 6.1.2 性能指标

| 指标 | 采集方式 | 告警阈值 |
|------|---------|---------|
| TTFT (首字延迟) | Agent span 计时 | P95 > 3s |
| E2E 延迟 | ChatHandler 计时 | P95 > 20s |
| 检索延迟 | HybridSearch trace | P95 > 1s |
| LLM 调用延迟 | LlmMiddleware 计时 | P95 > 10s |
| 工具执行延迟 | ToolRegistry 计时 | P95 > 5s |
| 文档入库耗时 | Pipeline trace | > 120s |

#### 6.1.3 可靠性指标

| 指标 | 采集方式 | 告警阈值 |
|------|---------|---------|
| LLM 调用成功率 | Middleware 统计 | < 95% |
| 工具执行成功率 | ToolRegistry 统计 | < 90% |
| 检索命中率 | 非空结果占比 | < 80% |
| 熔断器状态 | CircuitBreaker 状态 | OPEN |
| 错误率 | 5xx 响应占比 | > 5% |

#### 6.1.4 成本指标

| 指标 | 采集方式 | 告警阈值 |
|------|---------|---------|
| LLM Token 消耗/日 | LlmMiddleware 累计 | > 100K tokens |
| Embedding API 调用/日 | EmbeddingFactory 计数 | > 1000 |
| 存储空间使用 | 磁盘监控 | > 80% |

### 6.2 可观测性实现

```python
# ObservabilityProvider 实现
class MetricsCollector(ObservabilityProvider):
    """基于 Prometheus 风格的指标收集"""

    def __init__(self):
        self.counters = {}     # 计数器
        self.histograms = {}   # 直方图（延迟分布）
        self.gauges = {}       # 瞬时值

    async def record_metric(self, name, value, unit, tags):
        # 写入内存 + 定期刷盘到 metrics.jsonl
        ...

    async def create_span(self, name, attributes):
        span = Span(name=name, start_time=time.time(), attributes=attributes)
        return span

# 在 Agent 中已有的埋点位置（参考 Vanna 的设计）：
# - agent.user_resolution
# - agent.send_message (整体)
# - agent.tool.execute (每个工具)
# - llm.request / llm.stream (每次 LLM 调用)
# - agent.hook.before_message / after_message
```

### 6.3 演进路线

```
┌─────────────────────────────────────────────────────────────────┐
│                        演进路线图                                │
│                                                                 │
│  Phase 1: MVP (当前)              预计: 1-2 周                   │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  ✓ Agent Core (ReAct 工具循环)                                   │
│  ✓ 5 个核心工具 (检索/复习/出题/评判/入库)                        │
│  ✓ LlmService (OpenAI + Ollama)                                 │
│  ✓ PPT Loader                                                   │
│  ✓ FastAPI + SSE 流式                                            │
│  ✓ 基础 Web UI (聊天 + 上传)                                     │
│  ✓ 复用现有 RAG (HybridSearch + ChromaDB + BM25)                 │
│                                                                 │
│  Phase 2: 打磨体验                预计: 1 周                     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  □ 习题模式专属 UI (选择题卡片、SQL 编辑器)                       │
│  □ 会话历史持久化 (FileConversationStore)                         │
│  □ 考点总结导出 (Markdown / PDF)                                 │
│  □ 错题本功能 (AgentMemory 记录错题)                              │
│  □ 多 LLM 切换 (DeepSeek, Azure)                                │
│                                                                 │
│  Phase 3: 生产加固              预计: 1-2 周                     │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  □ 限流 + 熔断 + 降级完整实现                                     │
│  □ 审计日志持久化                                                 │
│  □ 多租户基础 (用户注册/登录)                                     │
│  □ Docker Compose 一键部署                                       │
│  □ 健康检查 + 基础监控 Dashboard                                  │
│  □ 自动化测试 (unit + integration)                               │
│                                                                 │
│  Phase 4: 智能增强              预计: 持续迭代                    │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  □ 自适应出题 (根据错题率调整难度)                                 │
│  □ 知识图谱 (章节/概念关系可视化)                                 │
│  □ Retrieval 评估 (hit_rate, MRR, faithfulness)                  │
│  □ Prompt 自动优化 (A/B 测试框架)                                 │
│  □ 多模态 (课件图片理解、手写公式识别)                             │
│  □ 协作学习 (学习小组、排行榜)                                    │
│                                                                 │
│  Phase 5: 规模化               预计: 按需                        │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  □ Kubernetes 部署                                               │
│  □ 向量数据库迁移 (Qdrant / Milvus)                              │
│  □ Redis 缓存层                                                  │
│  □ 多课程支持 (不限于数据库)                                      │
│  □ LLM Gateway (统一代理 + 成本控制)                              │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 持续优化机制

| 优化方向 | 方法 | 指标驱动 |
|---------|------|---------|
| **检索质量** | 定期运行评估脚本 (hit_rate, MRR)，调整 chunk_size、top_k、RRF 权重 | 检索命中率 < 80% 触发 |
| **Prompt 工程** | 版本化 system prompt，A/B 测试不同 prompt 的回答质量 | 用户满意度 |
| **出题质量** | 收集用户反馈（题目太简单/太难/有误），迭代 prompt | 习题评分分布 |
| **成本控制** | 监控 token 用量，对高频相似查询增加缓存 (LlmMiddleware) | 日均 token > 阈值 |
| **延迟优化** | 分析 trace 瀑布图，识别瓶颈（通常是 LLM 调用） | P95 > SLA 触发 |
| **知识库更新** | 监控文档版本，课件更新时自动触发重入库 | 文件 SHA256 变化 |

---

## 7. Memory 记忆系统设计

> 参考：[AI Agent 记忆系统：从短期到长期的技术架构与实践](https://mp.weixin.qq.com/s/mftM6jr0YiFxRATeNvm5Qg)（阿里云·翼严）、Mem0、Vanna AgentMemory

### 7.1 为什么需要 Memory — 教育场景的独特价值

传统 RAG 只做"检索-生成"，**没有记忆**：每次对话都是一张白纸。但教育场景天然需要"记住学生"：

| 痛点 | Memory 解决方案 |
|------|----------------|
| 学生重复问同一知识点，Agent 每次都从头解释 | **长期记忆**记住"该学生在 B+树 概念上反复提问"，主动加强该知识点 |
| 出题难度不匹配（太简单或太难） | **错题记忆**分析历史正确率，自适应调整难度 |
| 复习时不知道哪些是薄弱环节 | **知识掌握图谱**记录每个知识点的掌握程度 |
| 长对话超出 LLM 上下文窗口 | **上下文工程**策略（压缩、卸载、摘要） |

### 7.2 两层记忆架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Memory Architecture                               │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  短期记忆 (Short-Term / Session Memory)                       │  │
│  │                                                               │  │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                │  │
│  │  │ User Msg  │  │ Assistant │  │ Tool Call │                │  │
│  │  │           │  │ Msg       │  │ + Result  │                │  │
│  │  └───────────┘  └───────────┘  └───────────┘                │  │
│  │                                                               │  │
│  │  上下文工程策略:                                               │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │  │
│  │  │ 滑动窗口 │  │ 摘要压缩 │  │ 工具结果 │                   │  │
│  │  │ (最近N轮) │  │ (LLM摘要)│  │ 卸载     │                   │  │
│  │  └──────────┘  └──────────┘  └──────────┘                   │  │
│  └───────────────────────────────────┬───────────────────────────┘  │
│                                      │                              │
│                          Record ↓    ↑ Retrieve                     │
│                                      │                              │
│  ┌───────────────────────────────────▼───────────────────────────┐  │
│  │  长期记忆 (Long-Term / Cross-Session Memory)                  │  │
│  │                                                               │  │
│  │  ┌─────────────────┐  ┌─────────────────┐                   │  │
│  │  │ 学生画像记忆     │  │ 错题记忆         │                   │  │
│  │  │ (Student Profile)│  │ (Error Memory)  │                   │  │
│  │  │                 │  │                  │                   │  │
│  │  │ - 学习偏好      │  │ - 题目+错误答案   │                   │  │
│  │  │ - 薄弱知识点    │  │ - 正确答案+解析   │                   │  │
│  │  │ - 学习进度      │  │ - 错误类型分类    │                   │  │
│  │  │ - 偏好出题类型  │  │ - 掌握状态(已纠正?)│                  │  │
│  │  └─────────────────┘  └─────────────────┘                   │  │
│  │                                                               │  │
│  │  ┌─────────────────┐  ┌─────────────────┐                   │  │
│  │  │ 知识掌握图谱     │  │ 工具经验记忆     │                   │  │
│  │  │ (Knowledge Map) │  │ (Skill Memory)  │                   │  │
│  │  │                 │  │                  │                   │  │
│  │  │ - 章节→掌握度   │  │ - 成功的工具组合  │                   │  │
│  │  │ - 概念→正确率   │  │ - 优质prompt模板  │                   │  │
│  │  │ - 最后复习时间   │  │ - 题目难度校准    │                   │  │
│  │  └─────────────────┘  └─────────────────┘                   │  │
│  │                                                               │  │
│  │  存储层: ChromaDB(语义检索) + SQLite(结构化查询+审计)          │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 短期记忆：上下文工程策略

随着对话增长，需要智能化地管理上下文窗口。参考 AgentScope 的 AutoContextMemory 思路，采用**渐进式压缩**：

```
策略优先级（从轻量到重量）：

Level 1: 滑动窗口
  → 保留最近 20 轮对话 (约 40 条消息)
  → 超出部分移入 "历史区"

Level 2: 工具结果卸载
  → 将 ToolResult 中的大文本卸载到文件
  → 消息中仅保留摘要引用: "[检索结果: 10条, 详见 ref_001]"
  → 需要时可通过 ReadFileTool 恢复

Level 3: 历史对话摘要
  → 对超出窗口的历史对话使用 LLM 生成摘要
  → 摘要注入 system prompt 的 "Previous Context" 区
  → 原始消息保留在 ConversationStore (可追溯)

Level 4: Token 预算管理
  → 设定 max_context_tokens = 8000
  → system_prompt (固定) + memory_injection + history + current_message
  → 各部分按优先级分配预算
```

实现位置：通过 `ConversationFilter` 接口：

```python
class ContextEngineeringFilter(ConversationFilter):
    """渐进式上下文压缩"""

    def __init__(self, max_messages=40, max_tokens=8000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens

    def filter_messages(self, messages: list[Message]) -> list[Message]:
        # Level 1: 滑动窗口
        if len(messages) > self.max_messages:
            kept = messages[-self.max_messages:]
            summarized = self._summarize_old(messages[:-self.max_messages])
            kept.insert(0, Message(role="system", content=f"[历史摘要] {summarized}"))
            messages = kept

        # Level 2: 卸载大型工具结果
        for msg in messages:
            if msg.role == "tool" and len(msg.content) > 2000:
                ref_id = self._offload_to_file(msg.content)
                msg.content = f"[工具结果已卸载, ref={ref_id}, 摘要: {msg.content[:200]}...]"

        return messages
```

### 7.4 长期记忆：四类记忆详细设计

#### 7.4.1 学生画像记忆 (StudentProfileMemory)

```python
class StudentProfile(BaseModel):
    user_id: str
    preferences: dict          # {"preferred_quiz_type": "选择题", "difficulty": "中等"}
    weak_topics: list[str]     # ["B+树", "事务隔离级别", "范式分解"]
    strong_topics: list[str]   # ["ER模型", "SQL基础"]
    learning_pace: str         # "fast" | "medium" | "slow"
    total_sessions: int
    total_quizzes: int
    overall_accuracy: float
    last_active: datetime
    notes: str                 # LLM 从对话中提取的自由文本观察

# Record 时机：每次对话结束时
# Retrieve 时机：每次对话开始时注入 system prompt
```

**Record 流程**（对话结束后触发）：

```
对话历史 → LLM 提取关键信息 → 更新 StudentProfile

Prompt 示例:
"分析以下对话，提取学生的学习状态更新：
- 学生提了哪些问题？暴露了哪些薄弱点？
- 学生对哪些概念表现出已掌握？
- 学生的偏好有无变化？
输出 JSON 格式的增量更新。"
```

**Retrieve 注入**（对话开始时）：

```
注入到 system prompt:
"当前学生画像：
- 薄弱知识点: B+树, 事务隔离级别
- 偏好题型: 选择题
- 整体正确率: 72%
- 上次学习: 第3章 关系代数

请根据学生画像个性化你的回答。对薄弱知识点提供更详细的解释。"
```

#### 7.4.2 错题记忆 (ErrorMemory)

```python
class ErrorRecord(BaseModel):
    id: str
    user_id: str
    question: str              # 原始题目
    question_type: str         # "选择题" | "填空题" | "SQL题" | "简答题"
    topic: str                 # "第3章-关系代数"
    concepts: list[str]        # ["自然连接", "投影", "选择"]
    user_answer: str           # 学生的错误答案
    correct_answer: str        # 正确答案
    explanation: str           # 解析
    error_type: str            # "概念混淆" | "计算错误" | "语法错误" | "遗漏"
    difficulty: int            # 1-5
    mastered: bool             # 是否已在后续答对（艾宾浩斯复习标记）
    created_at: datetime
    mastered_at: Optional[datetime]

# 存储: SQLite (结构化查询) + ChromaDB (语义检索)
# Record 时机: QuizEvaluatorTool 判断为错误时
# Retrieve 时机:
#   1. ReviewSummaryTool 生成复习时 → 优先覆盖错题知识点
#   2. QuizGeneratorTool 出题时 → 优先出薄弱知识点的题
#   3. 对话开始时 → 注入 "你有3道未掌握的错题" 提醒
```

#### 7.4.3 知识掌握图谱 (KnowledgeMapMemory)

```python
class KnowledgeNode(BaseModel):
    concept: str               # "B+树"
    chapter: str               # "第6章"
    mastery_level: float       # 0.0 ~ 1.0
    quiz_count: int            # 该知识点总答题数
    correct_count: int         # 正确数
    last_reviewed: datetime
    review_interval_days: int  # 艾宾浩斯间隔（1, 2, 4, 7, 15, 30）

# 更新策略:
# - 答对 → mastery_level += 0.1 (max 1.0)
# - 答错 → mastery_level -= 0.15 (min 0.0)
# - 超过 review_interval 未复习 → mastery_level 自然衰减

# Retrieve 用途:
# - 生成"需要复习的知识点"列表（按 mastery_level 升序 + 到期需复习）
# - 可视化学习雷达图（前端展示）
```

#### 7.4.4 工具经验记忆 (SkillMemory)

参考 Vanna 的 `SaveQuestionToolArgsTool`，记录成功的"问题→工具调用链"映射：

```python
class ToolUsageRecord(BaseModel):
    question_pattern: str      # "总结第X章考点" (泛化后的模式)
    tool_chain: list[str]      # ["KnowledgeQueryTool", "ReviewSummaryTool"]
    tool_args: list[dict]      # 每个工具的参数
    quality_score: float       # 用户反馈或自评
    embedding: list[float]     # 问题向量（用于相似检索）

# Retrieve 用途:
# - LlmContextEnhancer 检索相似成功案例
# - 注入 system prompt: "对于类似问题，推荐的工具调用链是..."
# - 减少 LLM 的决策不确定性，提高工具选择准确率
```

### 7.5 Memory 与 Agent 的集成点

```
                    Agent 请求生命周期中的 Memory 集成
                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ① 对话开始
  │
  ├─ [LlmContextEnhancer.enhance_system_prompt()]
  │   → Retrieve: StudentProfile → 注入学生画像
  │   → Retrieve: ErrorMemory → 注入 "你有N道未掌握错题"
  │   → Retrieve: KnowledgeMap → 注入 "需要复习: B+树, 范式"
  │   → Retrieve: SkillMemory → 注入类似问题的成功工具链
  │
  ② 工具循环中
  │
  ├─ [QuizEvaluatorTool.execute()]
  │   → Record: 错误答案 → ErrorMemory
  │   → Update: KnowledgeMap 掌握度
  │
  ├─ [QuizGeneratorTool.execute()]
  │   → Retrieve: ErrorMemory → 优先出错题相关知识点
  │   → Retrieve: KnowledgeMap → 根据掌握度调整难度
  │
  ├─ [ReviewSummaryTool.execute()]
  │   → Retrieve: KnowledgeMap → 标注已掌握/未掌握的考点
  │   → Retrieve: ErrorMemory → 补充典型错题案例
  │
  ③ 对话结束
  │
  ├─ [LifecycleHook.after_message()]
  │   → Record: 从对话中提取 → 更新 StudentProfile
  │   → Record: 成功的工具链 → SkillMemory
  │
  ④ 异步任务
  │
  └─ [ScheduledTask]
      → 艾宾浩斯衰减: 定期降低未复习知识点的 mastery_level
      → 记忆清理: 删除已掌握且超过 30 天的错题记录
```

### 7.6 面试讲解要点

> **面试官问**：你的 Agent 和普通 RAG 有什么区别？
>
> **回答**：核心区别在于 **Memory 系统**。普通 RAG 是无状态的"检索-生成"管道，每次对话都是一张白纸。我们的 Agent 有两层记忆：
> - **短期记忆**通过上下文工程（滑动窗口 + 工具结果卸载 + 历史摘要）解决 LLM 上下文窗口限制；
> - **长期记忆**包含四类：学生画像（千人千面）、错题本（艾宾浩斯遗忘曲线）、知识掌握图谱（自适应出题）、工具经验（提高工具调用准确率）。
>
> 长期记忆通过 Record/Retrieve 双向交互与 Agent 集成：对话结束时 LLM 提取有效信息写入（Record），对话开始时检索相关记忆注入上下文（Retrieve）。这让 Agent 能够"记住"学生，提供个性化的学习体验。

---

## 8. Skill 技能系统设计

> 参考：[2026年AI应用技术栈：Agent Skill 渐进式披露架构](https://cloud.tencent.com.cn/developer/article/2614419)、[Agent Skills 深度解析](https://atbug.com/agent-skills-reusable-ecosystem-for-ai-agents/)

### 8.1 为什么需要 Skill — 从"工具"到"技能"的升级

当前设计中的 5 个 Tool 是底层"原子操作"，但教育场景中的任务往往是**多工具组合**：

| 用户意图 | 需要的工具组合 | 问题 |
|---------|--------------|------|
| "帮我准备期末考试" | 检索全部章节 → 逐章生成考点 → 综合出题 → 模拟考试 | LLM 需要 5-10 轮工具调用，耗时长、不稳定 |
| "回顾我的错题" | 查询错题记忆 → 重新出相似题 → 评判 → 更新掌握度 | 每次都要 LLM 重新"发明"这个流程 |
| "讲解第 3 章" | 检索 → 按知识点分组 → 逐点讲解 → 出练习题 | 复杂编排，LLM 可能遗漏步骤 |

**Skill = 预定义的工具编排 + 专业 Prompt + 最佳实践**，让 Agent 对复杂任务有"肌肉记忆"。

### 8.2 渐进式披露架构 (Progressive Disclosure)

核心思想：**不在启动时加载所有技能的完整信息**，而是分三层按需加载，大幅降低 Token 消耗。

```
┌─────────────────────────────────────────────────────────────────┐
│              Skill Progressive Disclosure                         │
│                                                                 │
│  Level 1: 元数据层 (启动时加载)                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  每个 Skill 仅加载 name + description (50-100 bytes)    │    │
│  │                                                         │    │
│  │  exam_prep:    "期末考试复习准备，涵盖全部章节"            │    │
│  │  error_review: "错题回顾与强化练习"                       │    │
│  │  chapter_deep: "单章节深度讲解与练习"                     │    │
│  │  quiz_drill:   "专项题型训练（选择/SQL/简答）"             │    │
│  │  knowledge_check: "知识掌握度自测"                        │    │
│  │                                                         │    │
│  │  Token 成本: ~500 bytes (即使 10 个 Skill 也 < 1KB)     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                         │                                        │
│                         ▼ 用户意图匹配后                          │
│                                                                 │
│  Level 2: 指令层 (匹配后加载)                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  完整的执行 SOP + 专业 Prompt                            │    │
│  │                                                         │    │
│  │  exam_prep Skill:                                       │    │
│  │  ├── 步骤1: 获取课程章节列表                              │    │
│  │  ├── 步骤2: 逐章调用 KnowledgeQueryTool 检索             │    │
│  │  ├── 步骤3: 对每章调用 ReviewSummaryTool 生成考点         │    │
│  │  ├── 步骤4: 查询 ErrorMemory 获取薄弱知识点               │    │
│  │  ├── 步骤5: 调用 QuizGeneratorTool 重点覆盖薄弱点         │    │
│  │  ├── 步骤6: 生成复习计划时间表                            │    │
│  │  ├── 输出格式: Markdown + 题目 JSON                      │    │
│  │  └── 质量自检: 确保每章至少覆盖 3 个核心考点               │    │
│  │                                                         │    │
│  │  Token 成本: ~2KB per skill (仅加载匹配的 1 个)          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                         │                                        │
│                         ▼ 执行过程中按需                          │
│                                                                 │
│  Level 3: 资源层 (执行时加载)                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  外部知识、模板、代码片段                                  │    │
│  │                                                         │    │
│  │  - RAG 检索结果（调用 KnowledgeQueryTool 时获取）         │    │
│  │  - 题目模板库（选择题/SQL题的标准格式）                    │    │
│  │  - 评分标准（评判答案时的评分 rubric）                     │    │
│  │  - 领域知识（数据库课程大纲 JSON）                        │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### 8.3 Skill 定义格式

每个 Skill 采用 **Markdown + YAML Frontmatter** 的标准化格式：

```
skills/
├── exam_prep/
│   ├── SKILL.md              # 技能定义（元数据 + 指令）
│   ├── templates/
│   │   ├── review_outline.md  # 复习大纲模板
│   │   └── quiz_format.json   # 题目输出格式
│   └── prompts/
│       ├── summary_prompt.txt # ReviewSummaryTool 的专用 prompt
│       └── quiz_prompt.txt    # QuizGeneratorTool 的专用 prompt
│
├── error_review/
│   ├── SKILL.md
│   └── prompts/
│       └── spaced_repetition.txt  # 间隔重复策略 prompt
│
├── chapter_deep_dive/
│   ├── SKILL.md
│   └── templates/
│       └── chapter_structure.md
│
├── quiz_drill/
│   ├── SKILL.md
│   └── prompts/
│       ├── sql_quiz.txt
│       └── concept_quiz.txt
│
└── knowledge_check/
    ├── SKILL.md
    └── templates/
        └── radar_chart_data.json
```

SKILL.md 文件示例：

```yaml
---
name: exam_prep
description: "期末考试复习准备，涵盖全部章节考点总结与强化练习"
trigger_patterns:
  - "准备考试"
  - "期末复习"
  - "全面复习"
  - "考前冲刺"
tools_required:
  - KnowledgeQueryTool
  - ReviewSummaryTool
  - QuizGeneratorTool
memory_required:
  - ErrorMemory
  - KnowledgeMapMemory
estimated_tokens: 5000
estimated_tool_calls: 8-15
difficulty: advanced
---

# 期末考试复习 Skill

## 执行步骤

### Step 1: 课程结构分析
获取已入库的所有章节列表，构建课程知识地图。

### Step 2: 逐章考点检索
对每个章节调用 KnowledgeQueryTool (top_k=8)，收集核心内容。

### Step 3: 考点提炼
对每章的检索结果调用 ReviewSummaryTool，生成：
- 核心概念（3-5 个）
- 重要定理/公式
- 易错点
- 与其他章节的关联

### Step 4: 薄弱点强化
从 ErrorMemory 和 KnowledgeMapMemory 中获取学生薄弱知识点，
在考点总结中对这些知识点添加 ⚠️ 标记和额外解释。

### Step 5: 综合练习生成
调用 QuizGeneratorTool，按以下分配出题：
- 薄弱知识点: 40% 题量
- 中等掌握: 35% 题量
- 已掌握(巩固): 25% 题量

### Step 6: 输出复习计划
按时间线生成复习建议（根据距考试天数调整节奏）。

## 输出格式
使用 templates/review_outline.md 模板格式化输出。

## 质量自检
- [ ] 每章至少覆盖 3 个核心考点
- [ ] 薄弱知识点有额外解释
- [ ] 生成的题目涵盖多种题型
- [ ] 输出包含明确的复习时间建议
```

### 8.4 Skill 加载与执行机制

```python
class SkillRegistry:
    """管理所有可用 Skill，支持渐进式加载"""

    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)
        self.metadata_cache: dict[str, SkillMetadata] = {}  # Level 1: 始终缓存
        self.instruction_cache: dict[str, SkillInstruction] = {}  # Level 2: LRU
        self._load_all_metadata()  # 启动时仅加载元数据

    def _load_all_metadata(self):
        """Level 1: 启动时加载所有 Skill 的 YAML frontmatter"""
        for skill_dir in self.skills_dir.iterdir():
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                meta = self._parse_frontmatter(skill_file)
                self.metadata_cache[meta.name] = meta

    def get_skill_descriptions_for_prompt(self) -> str:
        """返回所有 Skill 的简短描述，注入 system prompt"""
        lines = ["可用的学习技能:"]
        for name, meta in self.metadata_cache.items():
            lines.append(f"  - {name}: {meta.description}")
        return "\n".join(lines)

    def match_skill(self, user_message: str) -> Optional[str]:
        """根据用户消息匹配最适合的 Skill"""
        for name, meta in self.metadata_cache.items():
            for pattern in meta.trigger_patterns:
                if pattern in user_message:
                    return name
        return None  # 无匹配时由 LLM 自由决策

    def load_instruction(self, skill_name: str) -> SkillInstruction:
        """Level 2: 匹配后加载完整指令"""
        if skill_name not in self.instruction_cache:
            skill_file = self.skills_dir / skill_name / "SKILL.md"
            self.instruction_cache[skill_name] = self._parse_body(skill_file)
        return self.instruction_cache[skill_name]

    def load_resource(self, skill_name: str, resource_path: str) -> str:
        """Level 3: 执行时按需加载资源文件"""
        full_path = self.skills_dir / skill_name / resource_path
        return full_path.read_text(encoding="utf-8")
```

### 8.5 Skill 与 Agent 的集成

Skill 通过 `WorkflowHandler` 和 `LlmContextEnhancer` 两个扩展点与 Agent 集成：

```
用户消息 "帮我准备期末考试"
    │
    ▼
[WorkflowHandler.try_handle()]
    ├─ SkillRegistry.match_skill("帮我准备期末考试")
    │  → 匹配到 "exam_prep"
    │
    ├─ 方式A: 注入式（推荐）
    │  → should_skip_llm = False
    │  → 将 Skill 指令注入 system prompt 的 "Active Skill" 区
    │  → LLM 按照 Skill SOP 逐步执行工具调用
    │  → 优点: LLM 仍有灵活性，可根据上下文调整
    │
    └─ 方式B: 编排式（确定性高的任务）
       → should_skip_llm = True (部分步骤)
       → 直接按 SOP 编排工具调用，绕过 LLM 决策
       → 仅在需要生成文本时调用 LLM
       → 优点: 确定性高、速度快、成本低
```

注入式集成的 system prompt 效果：

```
[System Prompt]

你是一位数据库课程学习助手...

[Active Skill: exam_prep]
用户希望进行期末考试复习。请按以下步骤执行：
1. 调用 KnowledgeQueryTool 获取各章节内容
2. 对每章调用 ReviewSummaryTool 生成考点
3. 查询学生的薄弱知识点，在考点中重点标注
4. 调用 QuizGeneratorTool 生成强化练习（薄弱点 40%，中等 35%，巩固 25%）
5. 生成复习时间计划

注意：
- 学生薄弱知识点: B+树, 事务隔离级别
- 学生偏好题型: 选择题
- 请确保每章至少覆盖 3 个核心考点

[Available Tools]
...
```

### 8.6 预定义 Skill 列表

| Skill | 触发模式 | 工具链 | 估计调用次数 |
|-------|---------|--------|------------|
| **exam_prep** | "准备考试/期末复习/考前冲刺" | Query → Review × N → Quiz | 8-15 |
| **error_review** | "回顾错题/错题本/纠错" | ErrorMemory → Quiz(similar) → Evaluate | 3-6 |
| **chapter_deep_dive** | "讲解第X章/深入学习" | Query → 结构化讲解 → Quiz | 4-8 |
| **quiz_drill** | "出题/练习/做题" | Query → Quiz → Evaluate × N | 3-10 |
| **knowledge_check** | "测试掌握度/知识自测" | KnowledgeMap → Quiz(全覆盖) → 雷达图 | 5-12 |
| **quick_qa** | (默认，无特定触发) | Query → 直接回答 | 1-2 |

### 8.7 面试讲解要点

> **面试官问**：你的工具系统和 LangChain 的 Tool 有什么区别？
>
> **回答**：我们在 Tool 之上增加了 **Skill 技能层**。Tool 是原子操作（检索、出题、评判），Skill 是多个 Tool 的编排组合，代表一个完整的教学场景（如期末复习、错题回顾）。
>
> Skill 采用**渐进式披露**（Progressive Disclosure）架构：启动时仅加载元数据（< 1KB），匹配后加载完整 SOP，执行时按需加载资源。这比把所有 Skill 的指令全部塞进 system prompt 节省 60-80% 的 Token 成本。
>
> Skill 还与 Memory 系统深度集成：出题技能会参考错题记忆和知识掌握图谱，自适应调整难度和覆盖范围。这让 Agent 不仅是"会调工具"，而是"会教学"。

---

## 附录 A: 技术选型决策记录

| 决策点 | 选择 | 备选 | 理由 |
|--------|------|------|------|
| Agent 框架 | 自建（参考 Vanna） | LangChain, CrewAI | 面试需展示底层理解，自建更有说服力 |
| LLM 调用 | 直接 SDK (openai) | LiteLLM, LangChain | 减少依赖，OpenAI SDK 兼容多家 API |
| 向量存储 | ChromaDB | Qdrant, Milvus | 已有基础设施，零运维，适合 demo |
| Web 框架 | FastAPI | Flask, Starlette | 原生 async，SSE 支持好，类型提示 |
| 前端 | 原生 HTML/JS | React, Vue | 零构建依赖，面试演示简单直接 |
| PPT 解析 | python-pptx | LibreOffice, Unstructured | 纯 Python，无系统依赖 |
| 会话存储 | 文件系统 JSON | SQLite, Redis | 简单可靠，面试规模足够 |
| 长期记忆 | 自建（参考 Mem0 模式） | Mem0, Zep | 面试展示底层理解 + 教育场景定制 |
| Skill 系统 | 自建渐进式披露 | LangChain Tool, CrewAI Task | 教育场景需要多工具编排，Skill 更贴合 |

## 附录 B: 与 Vanna.ai v2.0 及业界实践的对照

| Vanna / 业界设计 | 我们的采用 | 我们的创新/调整 |
|-----------------|-----------|---------------|
| Agent + ToolRegistry + LlmService | 完整采用 | - |
| Tool[T] 泛型 + Pydantic Schema | 完整采用 | - |
| LifecycleHook (before/after) | 完整采用 | 增加限流 Hook + Memory Record Hook |
| LlmMiddleware | 完整采用 | 增加重试/熔断 Middleware |
| UserResolver + RequestContext | 简化采用 | 简化为 Cookie-based 认证 |
| ConversationStore | 采用 FileSystem 实现 | 增加上下文工程（压缩/卸载/摘要） |
| WorkflowHandler | **增强** | 集成 Skill 匹配，按需注入 SOP |
| ObservabilityProvider + Span | 采用 | 简化为文件输出 |
| AuditLogger | 完整采用 | JSONL 文件实现 |
| ErrorRecoveryStrategy | **增强** | Vanna 未在主循环调用，我们实际接入 |
| Vanna AgentMemory | **大幅增强** | 4 类长期记忆（画像/错题/知识图谱/工具经验） |
| Mem0 Record/Retrieve 模式 | 采用核心理念 | 自建轻量实现，针对教育场景定制 |
| AgentScope 上下文工程 | 采用渐进式压缩 | 4 级压缩策略（窗口→卸载→摘要→预算） |
| Agent Skill 渐进式披露 | 采用三层架构 | 3 级加载（元数据→指令→资源），节省 60%+ Token |
| RunSqlTool / VisualizeDataTool | **替换** | 替换为教育专用工具 (5 个) + Skill 编排 (5 个) |
| `<vanna-chat>` Web Component | **简化** | 纯 HTML/JS 实现，增加习题交互模式 |
| SSE + WebSocket + Poll | SSE only | 面试场景 SSE 足够 |

## 附录 C: 面试关键问答准备

| 面试问题 | 关键回答要点 |
|---------|------------|
| "这是什么类型的 Agent？" | ReAct 范式 + 现代 Tool Calling API，不是传统文本解析 |
| "和普通 RAG 有什么区别？" | 两层 Memory（短期上下文工程 + 长期四类记忆）让 Agent 有状态、可个性化 |
| "为什么不用 LangChain？" | 面试展示底层理解；自建 Agent Core 约 300 行，Tool 注册约 150 行，完全可控 |
| "Skill 和 Tool 的区别？" | Tool 是原子操作，Skill 是多 Tool 编排 + 专业 Prompt + 最佳实践 |
| "渐进式披露有什么好处？" | 10 个 Skill 元数据 < 1KB，比全量加载节省 60-80% Token，参考 Agent Skill 社区标准 |
| "长期记忆怎么实现的？" | 参考 Mem0 的 Record/Retrieve 模式，LLM 提取事实 → ChromaDB 向量存储 → 语义检索注入 |
| "错题本怎么用来出题？" | ErrorMemory + KnowledgeMap → 识别薄弱点 → QuizGeneratorTool 按比例分配难度 |
| "稳定性怎么保证？" | 四层防护：限流(LifecycleHook) → 超时(分层) → 重试(LlmMiddleware) → 熔断/降级 |
| "上下文太长怎么办？" | 渐进式压缩：滑动窗口 → 工具结果卸载 → 历史摘要 → Token 预算分配 |
| "数据怎么隔离？" | 会话 per-user，记忆 per-user，知识库 per-collection，工具 access_groups 权限 |
