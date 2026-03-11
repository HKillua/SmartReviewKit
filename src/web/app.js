/** SmartReviewKit — ChatGPT-style Frontend */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const messagesEl = $('#messages');
const inputEl = $('#messageInput');
const sendBtn = $('#sendBtn');
const themeToggle = $('#themeToggle');
const themeLabel = $('#themeLabel');
const uploadBtn = $('#uploadBtn');
const uploadModal = $('#uploadModal');
const closeModal = $('#closeModal');
const dropZone = $('#dropZone');
const fileInput = $('#fileInput');
const uploadProgress = $('#uploadProgress');
const uploadStatus = $('#uploadStatus');
const progressFill = $('.progress-fill');
const sidebar = $('#sidebar');
const sidebarOverlay = $('#sidebarOverlay');
const menuBtn = $('#menuBtn');
const newChatBtn = $('#newChatBtn');
const convListEl = $('#conversationList');
const chatTitle = $('#chatTitle');
const welcomeScreen = $('#welcomeScreen');

let conversationId = null;
let isStreaming = false;
const USER_ID = 'default_user';

/* P7: debounced rendering — avoid full markdown+math re-render on every tiny chunk */
let _renderTimer = null;
let _renderPending = false;
function scheduleRender(el, content) {
  _renderPending = true;
  if (_renderTimer) return;
  _renderTimer = setTimeout(() => {
    _renderTimer = null;
    if (_renderPending) {
      _renderPending = false;
      el.innerHTML = renderMarkdown(content);
      renderMath(el);
      scrollToBottom();
    }
  }, 80);
}
function flushRender(el, content) {
  if (_renderTimer) { clearTimeout(_renderTimer); _renderTimer = null; }
  _renderPending = false;
  el.innerHTML = renderMarkdown(content);
  renderMath(el);
  scrollToBottom();
}

// ================================================================
// THEME
// ================================================================

function initTheme() {
  const saved = localStorage.getItem('theme') || 'light';
  applyTheme(saved);
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  themeLabel.textContent = theme === 'dark' ? '浅色模式' : '深色模式';
  localStorage.setItem('theme', theme);
}

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

// ================================================================
// MARKDOWN + KATEX
// ================================================================

marked.setOptions({
  highlight: (code, lang) => {
    if (lang && hljs.getLanguage(lang)) {
      return hljs.highlight(code, { language: lang }).value;
    }
    return hljs.highlightAuto(code).value;
  },
  breaks: true,
});

function renderMarkdown(text) {
  return marked.parse(text);
}

function renderMath(el) {
  if (!window.renderMathInElement) return;
  try {
    window.renderMathInElement(el, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
      ],
      throwOnError: false,
    });
  } catch (_) {}
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

// ================================================================
// MESSAGES
// ================================================================

function hideWelcome() {
  if (welcomeScreen) welcomeScreen.style.display = 'none';
}

function showWelcome() {
  if (welcomeScreen) welcomeScreen.style.display = '';
}

function addMessage(role, content) {
  hideWelcome();
  const div = document.createElement('div');
  div.className = 'message';

  const avatarClass = role === 'user' ? 'user' : 'ai';
  const avatarLetter = role === 'user' ? 'U' : 'AI';

  const avatarHtml = `<div class="msg-avatar ${avatarClass}">${avatarLetter}</div>`;
  const bodyHtml = role === 'user'
    ? `<div class="msg-body"><p>${escapeHtml(content)}</p></div>`
    : `<div class="msg-body">${renderMarkdown(content)}</div>`;

  div.innerHTML = avatarHtml + bodyHtml;

  const bodyEl = div.querySelector('.msg-body');
  if (role !== 'user') renderMath(bodyEl);

  messagesEl.appendChild(div);
  scrollToBottom();
  return bodyEl;
}

function addThinkingIndicator() {
  hideWelcome();
  const div = document.createElement('div');
  div.className = 'message';
  div.id = 'thinkingMsg';
  div.innerHTML = `
    <div class="msg-avatar ai">AI</div>
    <div class="msg-body">
      <div class="thinking-dots"><span></span><span></span><span></span></div>
    </div>`;
  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

function removeThinking() {
  const el = $('#thinkingMsg');
  if (el) el.remove();
}

function addToolIndicator(toolName) {
  hideWelcome();
  const div = document.createElement('div');
  div.className = 'message';
  div.innerHTML = `
    <div class="msg-avatar ai">AI</div>
    <div class="msg-body">
      <div class="tool-indicator"><span class="spinner"></span> 正在调用: ${escapeHtml(toolName)}</div>
    </div>`;
  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

function renderAssistantMeta(bodyEl, metadata) {
  if (!bodyEl) return;

  const existing = bodyEl.querySelector('.assistant-meta');
  if (existing) existing.remove();

  const citations = Array.isArray(metadata?.citations) ? metadata.citations : [];
  const groundingAction = metadata?.grounding_policy_action || '';
  const generationMode = metadata?.generation_mode || '';
  const needsWarning = groundingAction && groundingAction !== 'normal';

  if (!citations.length && !needsWarning && generationMode !== 'insufficient_evidence') {
    return;
  }

  const wrapper = document.createElement('div');
  wrapper.className = 'assistant-meta';

  if (needsWarning) {
    const warning = document.createElement('div');
    warning.className = 'assistant-warning';
    warning.textContent = groundingAction === 'conservative_rewrite'
      ? '证据偏弱，回答已按可验证范围保守收敛。'
      : '当前回答的课程证据不足，请结合课件再次确认。';
    wrapper.appendChild(warning);
  }

  if (generationMode === 'insufficient_evidence') {
    const note = document.createElement('div');
    note.className = 'assistant-warning';
    note.textContent = '未生成正式题目，请缩小范围、指定章节或补充相关资料。';
    wrapper.appendChild(note);
  }

  if (citations.length) {
    const details = document.createElement('details');
    details.className = 'assistant-sources';
    details.open = true;

    const summary = document.createElement('summary');
    summary.textContent = `Sources (${citations.length})`;
    details.appendChild(summary);

    const list = document.createElement('div');
    list.className = 'assistant-sources-list';
    list.innerHTML = citations.map((citation) => {
      const title = citation?.metadata?.title ? ` · ${escapeHtml(citation.metadata.title)}` : '';
      const page = citation?.page ? ` · p.${escapeHtml(String(citation.page))}` : '';
      const snippet = citation?.text_snippet ? `<div class="assistant-source-snippet">${escapeHtml(citation.text_snippet)}</div>` : '';
      return `
        <div class="assistant-source-item">
          <div class="assistant-source-title">[${escapeHtml(String(citation.index || '?'))}] ${escapeHtml(citation.source || 'unknown')}${page}${title}</div>
          ${snippet}
        </div>
      `;
    }).join('');
    details.appendChild(list);
    wrapper.appendChild(details);
  }

  bodyEl.appendChild(wrapper);
}

function scrollToBottom() {
  const container = $('#chatContainer');
  requestAnimationFrame(() => {
    container.scrollTop = container.scrollHeight;
  });
}

// ================================================================
// CHAT
// ================================================================

async function sendMessage(overrideMsg) {
  const message = overrideMsg || inputEl.value.trim();
  if (!message || isStreaming) return;

  addMessage('user', message);
  inputEl.value = '';
  inputEl.style.height = 'auto';
  updateSendBtn();
  isStreaming = true;
  sendBtn.disabled = true;

  let assistantEl = null;
  let fullContent = '';
  let currentToolEl = null;
  const thinkingEl = addThinkingIndicator();

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
        user_id: USER_ID,
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        try {
          const event = JSON.parse(jsonStr);
          switch (event.type) {
            case 'text_delta':
              removeThinking();
              if (currentToolEl) { currentToolEl.remove(); currentToolEl = null; }
              if (!assistantEl) { assistantEl = addMessage('assistant', ''); fullContent = ''; }
              fullContent += event.content || '';
              scheduleRender(assistantEl, fullContent);
              break;
            case 'tool_start':
              removeThinking();
              currentToolEl = addToolIndicator(event.tool_name || 'unknown');
              break;
            case 'tool_result':
              if (currentToolEl) { currentToolEl.remove(); currentToolEl = null; }
              break;
            case 'done':
              removeThinking();
              if (assistantEl && fullContent) flushRender(assistantEl, fullContent);
              if (assistantEl) renderAssistantMeta(assistantEl, event.metadata || {});
              if (event.metadata?.conversation_id) {
                conversationId = event.metadata.conversation_id;
              }
              if (event.metadata?.title) {
                chatTitle.textContent = event.metadata.title;
              }
              setTimeout(() => loadConversationList(), 300);
              break;
            case 'error':
              removeThinking();
              if (!assistantEl) { assistantEl = addMessage('assistant', ''); fullContent = ''; }
              fullContent += `\n\n**Error:** ${event.content || '发生错误'}`;
              assistantEl.innerHTML = renderMarkdown(fullContent);
              break;
          }
        } catch (_) {}
      }
    }
  } catch (err) {
    removeThinking();
    addMessage('assistant', `**网络错误:** ${err.message}`);
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    updateSendBtn();
    inputEl.focus();
  }
}

// ================================================================
// INPUT HANDLING
// ================================================================

function updateSendBtn() {
  sendBtn.disabled = !inputEl.value.trim() || isStreaming;
}

sendBtn.addEventListener('click', () => sendMessage());

inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + 'px';
  updateSendBtn();
});

// Suggestion buttons
document.addEventListener('click', (e) => {
  if (e.target.classList.contains('suggestion-btn')) {
    const msg = e.target.dataset.msg;
    if (msg) sendMessage(msg);
  }
});

// ================================================================
// SIDEBAR — CONVERSATION LIST
// ================================================================

async function loadConversationList() {
  try {
    const resp = await fetch(`/api/conversations?user_id=${USER_ID}&limit=50`);
    if (!resp.ok) return;
    const convs = await resp.json();
    renderConversationList(convs);
  } catch (_) {}
}

function renderConversationList(convs) {
  convListEl.innerHTML = '';
  for (const c of convs) {
    const item = document.createElement('div');
    item.className = 'conv-item' + (c.id === conversationId ? ' active' : '');
    item.innerHTML = `
      <svg class="conv-item-icon" width="16" height="16" viewBox="0 0 16 16" fill="none"><path d="M2 3h12v8a1 1 0 01-1 1H5l-3 3V3z" stroke="currentColor" stroke-width="1.2"/></svg>
      <span class="conv-item-title">${escapeHtml(c.title || '新对话')}</span>
      <button class="conv-item-delete" data-id="${c.id}" title="删除">
        <svg width="14" height="14" viewBox="0 0 14 14" fill="none"><path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
      </button>`;

    item.addEventListener('click', (e) => {
      if (e.target.closest('.conv-item-delete')) return;
      switchConversation(c.id);
    });

    const deleteBtn = item.querySelector('.conv-item-delete');
    deleteBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteConversation(c.id);
    });

    convListEl.appendChild(item);
  }
}

async function switchConversation(id) {
  if (id === conversationId) return;
  conversationId = id;

  try {
    const resp = await fetch(`/api/conversations/${id}?user_id=${USER_ID}`);
    if (!resp.ok) return;
    const conv = await resp.json();

    clearMessages();
    chatTitle.textContent = conv.title || 'SmartReviewKit';

    for (const msg of conv.messages) {
      if (msg.role === 'user') {
        addMessage('user', msg.content || '');
      } else if (msg.role === 'assistant' && msg.content) {
        addMessage('assistant', msg.content);
      }
    }

    loadConversationList();
    closeSidebar();
  } catch (_) {}
}

async function deleteConversation(id) {
  try {
    await fetch(`/api/conversations/${id}?user_id=${USER_ID}`, { method: 'DELETE' });
    if (id === conversationId) {
      startNewChat();
    }
    loadConversationList();
  } catch (_) {}
}

// ================================================================
// NEW CHAT
// ================================================================

function startNewChat() {
  conversationId = null;
  clearMessages();
  showWelcome();
  chatTitle.textContent = 'SmartReviewKit';
  loadConversationList();
  closeSidebar();
  inputEl.focus();
}

function clearMessages() {
  messagesEl.innerHTML = '';
  if (welcomeScreen) {
    messagesEl.appendChild(welcomeScreen);
  }
}

newChatBtn.addEventListener('click', startNewChat);

// ================================================================
// SIDEBAR TOGGLE (mobile)
// ================================================================

function openSidebar() {
  sidebar.classList.add('open');
  sidebarOverlay.classList.add('active');
}

function closeSidebar() {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('active');
}

menuBtn.addEventListener('click', () => {
  sidebar.classList.contains('open') ? closeSidebar() : openSidebar();
});

sidebarOverlay.addEventListener('click', closeSidebar);

// ================================================================
// FILE UPLOAD
// ================================================================

uploadBtn.addEventListener('click', () => uploadModal.classList.remove('hidden'));
closeModal.addEventListener('click', () => uploadModal.classList.add('hidden'));
$('.modal-backdrop')?.addEventListener('click', () => uploadModal.classList.add('hidden'));

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) uploadFile(fileInput.files[0]); });

async function uploadFile(file) {
  uploadProgress.classList.remove('hidden');
  uploadStatus.textContent = `正在上传: ${file.name}...`;
  progressFill.style.width = '30%';

  const formData = new FormData();
  formData.append('file', file);

  try {
    progressFill.style.width = '60%';
    const resp = await fetch(`/api/upload?user_id=${USER_ID}`, { method: 'POST', body: formData });
    const result = await resp.json();
    progressFill.style.width = '100%';

    if (result.success) {
      uploadStatus.textContent = `${file.name} 上传成功 (${result.chunk_count} 个分块)`;
      addMessage('assistant', `课件 **${file.name}** 已成功导入知识库（${result.chunk_count} 个分块）。`);
      setTimeout(() => uploadModal.classList.add('hidden'), 1500);
    } else {
      uploadStatus.textContent = `上传失败: ${result.error}`;
    }
  } catch (err) {
    uploadStatus.textContent = `上传错误: ${err.message}`;
    progressFill.style.width = '0%';
  }
}

// ================================================================
// INIT
// ================================================================

initTheme();
loadConversationList();
updateSendBtn();
inputEl.focus();
