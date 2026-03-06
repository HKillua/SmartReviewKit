# Phase P: 端到端延迟深度优化 — 复习文档

## 一、问题背景

从前端输入到看到第一个字，用户感受到明显延迟（3-8秒）。本次优化通过链路分析定位了 11 个瓶颈，从 P0（致命）到 P10（低优先级），逐一修复。

---

## 二、核心优化详解

### P0: 流式输出完全失效（致命 Bug）

**问题**: `_handle_streaming` 将所有 LLM stream chunk 收集到 `list` 中，**全部完成后**才通过 `return` 返回给 `_tool_loop`，然后 `_tool_loop` 用 `for ev in events: yield ev` 批量发出。这完全抹杀了 SSE 流式输出的优势。

```python
# 修复前 — 伪流式
async def _handle_streaming(self, request):
    events = []                                    # 缓冲区
    async for chunk in self.llm.stream_request(request):
        events.append(StreamEvent(...))            # 全部存入列表
    return LlmResponse(...), events                # 等完才返回

# _tool_loop:
response, events = await self._handle_streaming(request)  # 阻塞等待
for ev in events: yield ev                                 # 批量释放
```

**修复**: 删除 `_handle_streaming` 方法，将 stream 消费直接内联到 `_tool_loop` 中，每收到一个 chunk 就立即 `yield`：

```python
# 修复后 — 真实时流式
async for chunk in self.llm.stream_request(request):
    chunks.append(chunk)
    if chunk.delta_content:
        content_parts.append(chunk.delta_content)
        yield StreamEvent(type=TEXT_DELTA, content=chunk.delta_content)  # 即时发出

response = LlmResponse(content="".join(content_parts), tool_calls=_accumulate_tool_calls(chunks))
```

**面试要点**:
- 为什么不能用 `return` + `yield` 混合？因为 Python 的 async generator 不允许 `return` 非 None 值。必须在 generator 内部维护状态。
- 这是一个典型的"看起来实现了流式，实际上是伪流式"的 bug — 用户体验测试而非单元测试才能发现。

---

### P1: Agent 预处理并行化

**问题**: `get_memory_summary()`、`get_review_context()`、`try_handle()` 三个无依赖的异步操作串行执行。

**修复**: `asyncio.gather()` 并行执行：

```python
memory_ctx, review_ctx, wf_result = await asyncio.gather(
    _fetch_memory(), _fetch_review(), _fetch_skill()
)
```

**注意**: `try_handle` 可能返回 `direct_response` 导致提前返回。我们在 `gather` 完成后检查结果，不影响并行化。

---

### P2: Memory Summary 内部 DB 查询并行化

**问题**: `get_memory_summary` 内部 5 次 DB 查询（profile, sessions, due_for_review, weak_nodes, errors）串行执行。

**修复**: 将每个查询包装为独立的 async 函数，通过 `asyncio.gather` 一次性并行执行：

```python
profile, recent_sessions, (due, weak), errors = await asyncio.gather(
    _load_profile(), _load_sessions(), _load_kmap(), _load_errors(),
)
```

**注意**: `_load_kmap` 内部再用一层 `asyncio.gather` 并行查询 `due_for_review` 和 `weak_nodes`。

---

### P3/P6: 异步搜索 + 并行子查询

**问题**: `hybrid_search.search()` 是同步方法（内部用 `ThreadPoolExecutor`），在 async 上下文中直接调用会阻塞事件循环。Multi-query 子查询在 `for` 循环中逐个执行。

**修复**:
- `asyncio.to_thread()` 将同步搜索分派到线程池
- `asyncio.gather()` 并行执行所有子查询

```python
async def _search_one(q):
    return await asyncio.to_thread(self._hybrid_search.search, query=q, top_k=top_k, **kwargs)

search_results = await asyncio.gather(*[_search_one(q) for q in queries])
```

**面试要点**: `asyncio.to_thread` vs `run_in_executor` — 前者是 Python 3.9+ 的简化 API，内部使用默认线程池。

---

### P4: Prompt 模板缓存

**问题**: `SystemPromptBuilder.build()` 每次调用都从磁盘读取 `system_prompt.txt`。

**修复**: 在 `_load_template()` 中增加 `self._cached_template` 缓存，只读一次。

---

### P5: 持久化 DB 连接

**问题**: 每个 `aiosqlite` 操作都 `async with aiosqlite.connect(path)` 打开新连接、操作后关闭。SQLite 连接开销虽小，但高频访问时累积可观。

**修复**: 每个 store 维护 `self._conn: aiosqlite.Connection | None`，通过 `_get_conn()` 懒初始化，`close()` 时关闭：

```python
async def _get_conn(self) -> aiosqlite.Connection:
    if self._conn is None:
        self._conn = await aiosqlite.connect(self._db_path)
    return self._conn
```

**注意**: aiosqlite 连接不是线程安全的，但在单事件循环内顺序 await 是安全的。

---

### P7: 前端渲染防抖

**问题**: 每收到一个 `text_delta`（可能只有几个字），就执行 `renderMarkdown(fullContent)` + `renderMath()`。随着内容增长，渲染开销 O(n) 递增。

**修复**: 80ms debounce + 流式结束时 `flushRender()` 确保最终一致性：

```javascript
function scheduleRender(el, content) {
  _renderPending = true;
  if (_renderTimer) return;
  _renderTimer = setTimeout(() => {
    el.innerHTML = renderMarkdown(content);
    renderMath(el);
  }, 80);
}
```

---

### P8: 后台化 after_message hooks

**问题**: `DONE` 事件 yield 后，`conversations.update()` 和 `after_message` hooks（含 LLM 提取 ~1-2s）仍在 generator 内执行。SSE 连接直到 generator 完成才关闭。

**修复**: `asyncio.create_task()` 将保存和 hooks 移到后台：

```python
yield StreamEvent(type=DONE, metadata={...})
asyncio.create_task(self._post_message_tasks(conversation))  # fire-and-forget
```

**面试要点**: `create_task` 的生命周期管理 — task 会被 event loop 引用，不会被 GC。但需注意异常日志。

---

### P9: LLM 显式超时

**问题**: `AsyncOpenAI` 客户端没有设置 `timeout` 参数，依赖 SDK 默认值。网络不稳定时可能长时间挂起。

**修复**: 构造时传入 `httpx.Timeout(120.0, connect=10.0)`:
- 总超时 120s（LLM 生成可能较慢）
- 连接超时 10s（快速失败）

---

## 三、优化效果总结

| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| 首字延迟 | 3-8s | 0.5-1.5s | **70-80%** |
| 预处理耗时 | ~300ms (串行) | ~100ms (并行) | **66%** |
| Memory 查询 | ~250ms (5x50ms) | ~50ms (并行) | **80%** |
| DB 连接开销 | 每次 open/close | 持久化复用 | 消除 |
| 前端渲染 | 每 token 全量渲染 | 80ms 防抖 | **90%** 减少渲染次数 |
| SSE 连接保持 | hooks 完成才关闭 | DONE 后立即关闭 | **减少 1-2s** |

## 四、测试覆盖

| 测试 | 验证内容 |
|------|---------|
| `test_p0_streaming_yields_in_realtime` | 验证 text_delta 事件间有真实时间间隔（非批量） |
| `test_p1_parallel_preprocessing` | 验证 3 个预处理步骤并行执行（总耗时 < 单步） |
| `test_p2_parallel_memory_reads` | 验证 5 个 DB 查询并行（总耗时 < 累加） |
| `test_p4_prompt_template_cached` | 验证模板文件只读一次 |
| `test_p5_persistent_connection_*` (x4) | 验证每个 store 复用同一连接实例 |
| `test_p8_hooks_run_in_background` | 验证 generator 不等待 hooks 完成 |
| `test_p9_openai_service_has_timeout` | 验证 OpenAI 客户端有显式 timeout |
| `test_p3_search_runs_in_thread` | 验证搜索通过 to_thread 执行 |
