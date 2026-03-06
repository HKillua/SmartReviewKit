# Phase O: Agent 代码质量深度修复 — 复习文档

## 1. 概述

Phase O 对 Agent 层代码进行了全面的面试深度审查和修复，涵盖 **3 个 Bug 修复、功能接线补全、5 项安全加固、6 项架构改进**，新增 34 个测试用例，全部通过。

---

## 2. Phase 1: Bug 修复

### 2.1 AuditHook 计时 key 错误

**问题**: `before_tool` 用 `context.request_id` 存入 `_start_times`，`after_tool` 用空字符串 `""` 取，导致 `duration_ms` 恒为 ~0。

**修复**:
- 扩展 `LifecycleHook.after_tool` 签名加入 `context: ToolContext | None = None`
- `AuditHook.after_tool` 用 `context.request_id` 取回起始时间
- `agent.py` 的调用点传入 `context=tool_ctx`

**面试要点**: 审计日志的计时精确性对性能分析至关重要，这类 key 不一致 bug 是典型的代码审查考点。

### 2.2 FileConversationStore fd 清理

**问题**: 原子写的异常处理使用 `os.get_inheritable(fd)` 判断是否需要关闭 fd，与实际需求无关。

**修复**: 改为 `fd_closed` 布尔标记 + `try/finally` 模式，确保 fd 和临时文件在任何异常路径下都正确清理。

**面试要点**: 资源泄漏防护、原子文件写入（temp + rename）。

### 2.3 assistant 消息丢弃 content

**问题**: LLM 返回 `tool_calls` 时，`response.content`（推理文本）被丢弃。

**修复**: `Message(role="assistant", content=response.content, tool_calls=response.tool_calls)`

### 2.4 动态 import 反模式

**问题**: `__import__("src.agent.types", fromlist=["ToolResult"])` 和 `__import__("fastapi.responses", fromlist=["HTMLResponse"])`

**修复**: 替换为标准 `from ... import ...`

### 2.5 Message.role 类型约束

**问题**: `role: str` 无约束。

**修复**: `role: Literal["user", "assistant", "system", "tool"]`（通过 `RoleType` 类型别名）

---

## 3. Phase 2: 接线 + 功能补全

### 3.1 RateLimitHook 接入

**之前**: 令牌桶限流器实现完整但从未注册。

**修复**: 在 `app.py` 中注册 `RateLimitHook`，`AgentConfig` 增加 `rate_limit_rpm` 配置。同时增加桶数量上限（`_MAX_BUCKETS = 1024`）防止内存泄漏。

### 3.2 RetryWithBackoffMiddleware

**之前**: 名为 "retry middleware" 但只有 pass-through 基类。

**新实现**: `RetryWithBackoffMiddleware(LlmMiddleware)`
- 可重试错误匹配: timeout / rate_limit / 429 / 502-504 / overloaded
- 指数退避: `delay = min(base_delay * 2^attempt, max_delay)`
- 集成 `CircuitBreaker`: 连续失败 5 次 → OPEN → 冷却 30s → HALF_OPEN → 试探
- 在 `app.py` 中注册为 agent middleware

**面试要点**: 能完整说出"令牌桶限流 + 指数退避重试 + 三态熔断器"的组合策略及各自适用场景。

### 3.3 Token Budget (Level 4)

**之前**: `max_context_tokens` 定义了但未使用。

**修复**: `ContextEngineeringFilter` 新增 `_estimate_tokens` 和 `_apply_token_budget` 方法，在 Level 1 滑窗之后裁剪超出 token 预算的消息。

---

## 4. Phase 3: 安全加固

### 4.1 路径穿越防护（3 处）

| 位置 | 修复方式 |
|------|---------|
| `DocumentIngestTool` | 新增 `allowed_dirs` 白名单 + `_is_path_allowed` resolve 校验 |
| `routes.py` upload | `Path(filename).name` 提取纯文件名 + resolve 前缀校验 |
| `SkillRegistry.load_resource` | resolve 后检查路径前缀是否在 `_skills_dir` 下 |

### 4.2 SSE 断连检测

`chat_endpoint` 接收 `http_request: Request`，传入 `EventSourceResponse`，循环内检查 `await http_request.is_disconnected()`。客户端关闭浏览器后 agent 停止推理。

### 4.3 Prompt Injection 防护

新建 `src/agent/utils/sanitizer.py`:
- `sanitize_user_input(text, max_length)`: 正则匹配常见注入模式（中/英），替换为 `[FILTERED]`
- `validate_path_within(path, base)`: 路径穿越验证工具

在 `quiz_generator.py`、`review_summary.py`、`quiz_evaluator.py` 的所有用户输入插入 prompt 处调用。

---

## 5. Phase 4: 架构改进

### 5.1 aiosqlite 迁移

**之前**: 所有 memory store 在 `async def` 中使用同步 `sqlite3.connect()`，阻塞 asyncio 事件循环。

**修复**: 5 个 store 全部迁移到 `aiosqlite`:
- `student_profile.py`: `async with aiosqlite.connect()` 模式
- `error_memory.py`: 同上
- `knowledge_map.py`: 同上（含 `apply_decay` 事务化）
- `session_memory.py`: 同上
- `feedback_store.py`: 保留同步接口（routes 层使用），增加 `close()` 方法

建表仍用同步 `sqlite3`（`__init__` 中执行，确保服务启动前表就绪）。

### 5.2 Graceful Shutdown

`app.py` 增加 `@app.on_event("shutdown")`，按序关闭 memory store 和 `FeedbackStore`。

### 5.3 Async Compaction 修复

`ContextEngineeringFilter.filter_messages`: 在 async 上下文中不再 `pass`，改为明确记录 debug 日志并告知调用方应使用 `filter_messages_async`。agent.py 已优先调用 async 版本。

### 5.4 Conversation Schema 版本化

`Conversation` 增加 `schema_version: int = 1`。`FileConversationStore.get()` 调用 `_migrate_schema(data)` 对旧数据做向前兼容迁移。

### 5.5 缓存 TTL/LRU 淘汰

新建 `src/agent/utils/ttl_cache.py` — `TTLCache(OrderedDict)`:
- `max_size`: 超出时淘汰最旧条目
- `ttl_seconds`: 过期条目在 `get_value` / `__contains__` / `put` 时清除
- 应用: `ContextEngineeringFilter._cached_compactions` 使用 `TTLCache(64, 3600s)`
- `RateLimitHook._buckets` 增加 `_MAX_BUCKETS = 1024` 上限

---

## 6. 测试覆盖

| 测试类 | 测试数 | 覆盖内容 |
|--------|--------|---------|
| TestAuditHookKeyFix | 2 | duration > 0, fallback without context |
| TestFdCleanup | 2 | 原子写成功, 无残留临时文件 |
| TestContentPreservation | 1 | content + tool_calls 共存 |
| TestRoleLiteral | 2 | 合法/非法 role 校验 |
| TestSchemaVersion | 2 | 默认版本, 旧数据迁移 |
| TestRateLimitHook | 3 | 正常请求, 超限拦截, 桶淘汰 |
| TestRetryMiddleware | 3 | timeout 重试, 非重试错误, 正常通过 |
| TestCircuitBreaker | 2 | 阈值打开, HALF_OPEN 转换 |
| TestTokenBudget | 2 | 超限裁剪, 预算内不裁剪 |
| TestSanitizer | 4 | 中文注入, 英文注入, 正常保留, 长度限制 |
| TestPathTraversal | 3 | 合法路径, 穿越拦截, document_ingest 拦截 |
| TestSkillRegistryPathTraversal | 1 | `../` 拦截 |
| TestAiosqliteMemoryStores | 4 | 四个 store 的 roundtrip |
| TestTTLCache | 3 | max_size 淘汰, TTL 过期, contains 检查 |
| **总计** | **34** | |

---

## 7. 面试亮点总结

1. **稳定性三件套**: 令牌桶限流 → 指数退避重试 → 三态熔断器，完整实现且接入调用链
2. **安全纵深防御**: 路径穿越（3 处）、prompt injection（正则 + 长度限制）、SSE 断连检测
3. **async 正确性**: aiosqlite 替代 sqlite3、filter_messages_async 优先调用
4. **代码质量**: Literal 类型约束、原子写（fd_closed flag）、schema 版本化、TTL 缓存淘汰
5. **可观测性**: AuditHook duration 修复、CircuitBreaker 状态日志、shutdown handler 生命周期管理
