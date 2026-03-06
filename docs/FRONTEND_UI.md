# 前端 UI 优化 — ChatGPT 风格重构

> 阶段 L 实现文档 | 面试复习参考

---

## 1. 设计目标

将简陋的单页聊天页面升级为 **ChatGPT 风格** 的现代 UI：

- 左侧深色侧边栏（会话历史 + 新建对话）
- 右侧白色聊天区（消息气泡 + 头像）
- 支持 light/dark 双主题
- 移动端响应式（侧边栏折叠）

---

## 2. 架构总览

```
┌──────────────┬──────────────────────────────────┐
│              │  Header (title + hamburger menu)  │
│   Sidebar    ├──────────────────────────────────┤
│              │                                   │
│  + 新建对话   │        Chat Messages Area         │
│              │    (avatar + msg body rows)        │
│  会话历史     │                                   │
│  - Conv 1    │                                   │
│  - Conv 2    ├──────────────────────────────────┤
│  - Conv 3    │     Input Area (capsule input)    │
│              │                                   │
│  ──────────  │                                   │
│  上传课件     │                                   │
│  主题切换     │                                   │
└──────────────┴──────────────────────────────────┘
```

---

## 3. 前后端变更清单

### 后端新增/修改

| 文件 | 变更 | 说明 |
|---|---|---|
| `src/agent/types.py` | `Conversation.title: str` | 会话标题字段 |
| `src/agent/conversation.py` | `ConversationStore.delete()` | 抽象方法 + 两个实现类 |
| `src/server/routes.py` | `GET /api/conversations` | 返回会话列表 |
| `src/server/routes.py` | `DELETE /api/conversations/{id}` | 删除会话 |
| `src/agent/agent.py` | 自动生成标题 | 首条消息截取前30字 |
| `src/agent/agent.py` | `done` metadata | 包含 `title` 字段 |

### 前端重写

| 文件 | 说明 |
|---|---|
| `src/web/index.html` | ChatGPT 两栏布局 + 欢迎页 + 快捷按钮 |
| `src/web/style.css` | CSS 变量系统 + 侧边栏 + 响应式 |
| `src/web/app.js` | 会话 CRUD + 侧边栏 + 移动端适配 |

---

## 4. API 规格

### GET /api/conversations

```
GET /api/conversations?user_id=default_user&limit=50

Response: [
  {
    "id": "abc123",
    "title": "TCP 三次握手相关问题...",
    "updated_at": "2026-03-04T15:30:00",
    "message_count": 12
  }
]
```

### DELETE /api/conversations/{id}

```
DELETE /api/conversations/abc123?user_id=default_user

Response: {"success": true}
```

### done 事件 metadata 扩展

```json
{
  "type": "done",
  "metadata": {
    "conversation_id": "abc123",
    "title": "TCP 三次握手相关问题..."
  }
}
```

---

## 5. CSS 设计系统

### 颜色变量

```css
:root {
  /* 侧边栏 — 始终深色 */
  --sidebar-bg: #171717;
  --sidebar-hover: #2a2a2a;
  --sidebar-text: #ececec;

  /* 聊天区 — 跟随主题 */
  --bg: #f7f7f8;          /* dark: #212121 */
  --surface: #ffffff;      /* dark: #2f2f2f */
  --text: #1a1a1a;         /* dark: #ececec */

  /* 品牌色 */
  --primary: #10a37f;      /* ChatGPT 绿 */
  --avatar-user: #5436DA;  /* 紫色用户头像 */
  --avatar-ai: #10a37f;    /* 绿色 AI 头像 */
}
```

### 布局关键点

- 侧边栏固定 `260px` 宽，`flex-shrink: 0`
- 主面板 `flex: 1`，消息区 `max-width: 768px` 居中
- 输入框圆角胶囊形 `border-radius: 16px`
- 移动端侧边栏 `position: fixed` + `transform: translateX(-100%)`

---

## 6. JavaScript 核心逻辑

### 会话管理流程

```
页面加载
  ├── initTheme()           → 恢复主题偏好
  ├── loadConversationList() → GET /api/conversations
  └── updateSendBtn()       → 根据输入状态禁用/启用

用户点击 "新建对话"
  ├── conversationId = null
  ├── clearMessages()       → 清空消息区
  ├── showWelcome()         → 显示欢迎页面
  └── loadConversationList()

用户发送消息
  ├── addMessage('user', ...)
  ├── addThinkingIndicator()
  ├── fetch('/api/chat', SSE)
  │   ├── text_delta → 追加 AI 回复
  │   ├── tool_start → 显示工具调用指示器
  │   ├── tool_result → 移除工具指示器
  │   └── done → 保存 conversationId + title, 刷新侧边栏
  └── removeThinking()

用户点击侧边栏会话
  ├── switchConversation(id)
  ├── GET /api/conversations/{id}
  ├── clearMessages()
  └── 遍历 messages 渲染

用户删除会话
  ├── DELETE /api/conversations/{id}
  ├── 如果删的是当前会话 → startNewChat()
  └── loadConversationList()
```

### 移动端侧边栏

```
汉堡菜单按钮 (menu-btn)
  ├── 点击 → sidebar.classList.add('open')
  │         + overlay.classList.add('active')
  └── 点击 overlay 或选中会话 → closeSidebar()
```

---

## 7. 响应式断点

| 断点 | 行为 |
|---|---|
| `> 768px` | 侧边栏始终显示，两栏布局 |
| `<= 768px` | 侧边栏隐藏为抽屉，显示汉堡菜单，快捷建议单列 |
| `<= 480px` | 欢迎标题缩小，消息字体缩小 |

---

## 8. 面试亮点叙事

### 问：前端是怎么设计的？

> 参考 ChatGPT 的交互范式设计了两栏布局：
>
> 左侧深色侧边栏管理多会话（列表、新建、删除），
> 右侧白色区域是聊天主体，采用 SSE 实时流式渲染。
>
> 技术上是纯 HTML/CSS/JS，没用 React/Vue 等框架，
> 因为项目重点在后端 RAG + Agent，前端追求轻量。
> 但设计规范没有妥协——用了 CSS 变量系统支持 light/dark 双主题，
> 移动端用 `transform + overlay` 实现侧边栏折叠。

### 问：会话管理怎么做的？

> 后端 `ConversationStore` 是抽象接口，目前用
> `FileConversationStore`（JSON 文件，按用户目录隔离）。
>
> 新建对话只需前端把 `conversationId` 置空，
> Agent 的 `chat()` 方法检测到 `conversation_id=None` 时自动
> 调用 `create()` 创建新会话。
>
> 标题自动从用户第一条消息截取前 30 字符，
> 通过 SSE 的 `done` 事件 metadata 返回给前端更新侧边栏。
>
> 删除走 `DELETE /api/conversations/{id}`，后端直接删文件。

### 问：SSE 流式渲染怎么实现的？

> 后端用 FastAPI + `sse-starlette`，Agent 的 `chat()` 是
> `AsyncGenerator[StreamEvent]`，每产生一个 event 就 yield。
>
> 前端用 `fetch + ReadableStream` 读 SSE 流，
> 逐行解析 `data: {...}` JSON，按 event type 分发：
> - `text_delta` → 追加到 assistant 消息内容，重新 render Markdown + KaTeX
> - `tool_start` → 显示工具调用 spinner
> - `done` → 保存 conversationId 和 title
>
> 这样用户看到的是打字机效果的逐字输出。

---

## 9. 会话标题生成

```python
# Agent.chat() 中
if not conversation.title:
    conversation.title = effective_message[:30].strip()
    if len(effective_message) > 30:
        conversation.title += "..."
```

通过 `done` 事件传递给前端：

```python
yield StreamEvent(
    type=StreamEventType.DONE,
    metadata={"conversation_id": conversation.id, "title": conversation.title},
)
```

前端 JS 接收并更新：

```javascript
case 'done':
    conversationId = event.metadata.conversation_id;
    chatTitle.textContent = event.metadata.title;
    loadConversationList();  // 刷新侧边栏
    break;
```

---

## 10. 欢迎页面快捷操作

新对话时显示欢迎页面，包含 4 个快捷按钮：

| 按钮文本 | 发送消息 |
|---|---|
| TCP 三次握手考点 | `帮我总结 TCP 三次握手的考点` |
| IP 协议习题练习 | `出几道关于 IP 协议的选择题` |
| 复习应用层协议 | `帮我复习应用层协议` |
| 查看所有技能 | `/help` |

点击后直接调用 `sendMessage(msg)`，用户零输入即可开始学习。
