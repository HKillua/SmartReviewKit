/** Database Course Agent — Frontend Application */

const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const themeToggle = document.getElementById('themeToggle');
const uploadBtn = document.getElementById('uploadBtn');
const helpBtn = document.getElementById('helpBtn');
const uploadModal = document.getElementById('uploadModal');
const closeModal = document.getElementById('closeModal');
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const uploadProgress = document.getElementById('uploadProgress');
const uploadStatus = document.getElementById('uploadStatus');
const progressFill = document.querySelector('.progress-fill');

let conversationId = null;
let isStreaming = false;

// --- Theme ---
function initTheme() {
  const saved = localStorage.getItem('theme') || 'light';
  document.documentElement.setAttribute('data-theme', saved);
  themeToggle.textContent = saved === 'dark' ? '☀️' : '🌙';
}

themeToggle.addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  themeToggle.textContent = next === 'dark' ? '☀️' : '🌙';
});

// --- Markdown rendering ---
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

// --- Messages ---
function addMessage(role, content) {
  const div = document.createElement('div');
  div.className = `message ${role}`;
  const inner = document.createElement('div');
  inner.className = 'message-content';
  inner.innerHTML = role === 'user' ? `<p>${escapeHtml(content)}</p>` : renderMarkdown(content);
  div.appendChild(inner);
  messagesEl.appendChild(div);
  scrollToBottom();
  return inner;
}

function addToolIndicator(toolName) {
  const div = document.createElement('div');
  div.className = 'message assistant';
  div.innerHTML = `<div class="tool-indicator"><span class="spinner"></span> 正在调用工具: ${escapeHtml(toolName)}</div>`;
  messagesEl.appendChild(div);
  scrollToBottom();
  return div;
}

function scrollToBottom() {
  const container = document.getElementById('chatContainer');
  container.scrollTop = container.scrollHeight;
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

// --- Chat ---
async function sendMessage() {
  const message = inputEl.value.trim();
  if (!message || isStreaming) return;

  addMessage('user', message);
  inputEl.value = '';
  inputEl.style.height = 'auto';
  isStreaming = true;
  sendBtn.disabled = true;

  let assistantEl = null;
  let fullContent = '';
  let currentToolEl = null;

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        conversation_id: conversationId,
        user_id: 'default_user',
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
              if (currentToolEl) { currentToolEl.remove(); currentToolEl = null; }
              if (!assistantEl) { assistantEl = addMessage('assistant', ''); fullContent = ''; }
              fullContent += event.content || '';
              assistantEl.innerHTML = renderMarkdown(fullContent);
              scrollToBottom();
              break;
            case 'tool_start':
              currentToolEl = addToolIndicator(event.tool_name || 'unknown');
              break;
            case 'tool_result':
              if (currentToolEl) { currentToolEl.remove(); currentToolEl = null; }
              break;
            case 'done':
              if (event.metadata?.conversation_id) {
                conversationId = event.metadata.conversation_id;
              }
              break;
            case 'error':
              if (!assistantEl) { assistantEl = addMessage('assistant', ''); fullContent = ''; }
              fullContent += `\n\n⚠️ ${event.content || '发生错误'}`;
              assistantEl.innerHTML = renderMarkdown(fullContent);
              break;
          }
        } catch (e) { /* skip malformed SSE */ }
      }
    }
  } catch (err) {
    addMessage('assistant', `⚠️ 网络错误: ${err.message}`);
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 150) + 'px';
});

// --- Help ---
helpBtn.addEventListener('click', () => {
  inputEl.value = '/help';
  sendMessage();
});

// --- Upload ---
uploadBtn.addEventListener('click', () => uploadModal.classList.remove('hidden'));
closeModal.addEventListener('click', () => uploadModal.classList.add('hidden'));
uploadModal.addEventListener('click', (e) => { if (e.target === uploadModal) uploadModal.classList.add('hidden'); });

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
    const resp = await fetch('/api/upload?user_id=default_user', { method: 'POST', body: formData });
    const result = await resp.json();
    progressFill.style.width = '100%';

    if (result.success) {
      uploadStatus.textContent = `✅ ${file.name} 上传成功！分块数: ${result.chunk_count}`;
      addMessage('assistant', `📎 课件 **${file.name}** 已成功导入知识库（${result.chunk_count} 个分块）。`);
    } else {
      uploadStatus.textContent = `❌ 上传失败: ${result.error}`;
    }
  } catch (err) {
    uploadStatus.textContent = `❌ 上传错误: ${err.message}`;
    progressFill.style.width = '0%';
  }
}

// --- Init ---
initTheme();
inputEl.focus();
