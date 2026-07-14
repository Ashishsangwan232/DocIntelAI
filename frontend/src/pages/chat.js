/**
 * pages/chat.js
 * =============
 * Full chat page — feature parity with `src/ui/chat.py`: streaming
 * replies, citations, regenerate, per-conversation document scoping,
 * and export (PDF/Markdown/copy-to-clipboard).
 *
 * Session persistence: unlike Streamlit's `st.session_state` (server-
 * side, lives only as long as that browser tab's connection),
 * `localStorage` is used to remember the active session ID across
 * page reloads — a small, deliberate UX improvement, since losing
 * your conversation on every refresh would be a regression, not
 * parity. If the stored session ID no longer exists server-side
 * (e.g. deleted from another tab), a fresh one is created.
 */

import { chat as chatApi } from "../api/resources.js";
import { createDocumentScopeSelect } from "../components/documentScopeSelect.js";
import { openModal } from "../components/modal.js";
import { showToast } from "../components/toast.js";
import { escapeHtml, renderMatchDial } from "../utils/format.js";

const SESSION_STORAGE_KEY = "docintel_chat_session_id";

export function renderChatPage(container) {
  let sessionId = null;
  let messages = [];
  let documentScope = []; // empty = search all documents
  let streamController = null;
  let disposed = false;

  container.innerHTML = `
    <div class="page-content stack">
      <div class="page-header">
        <h1>💬 Chat</h1>
        <p class="text-muted">Ask questions across your documents, with citations.</p>
      </div>

      <div class="chat-toolbar">
        <button class="btn" id="new-chat-btn">New Chat</button>
        <button class="btn" id="clear-chat-btn">Clear Conversation</button>
        <button class="btn" id="export-btn" hidden>📤 Export</button>
        <div class="doc-scope-select" id="doc-scope-container"></div>
      </div>

      <div id="chat-messages" class="chat-messages"></div>

      <form id="chat-form" class="chat-input-bar">
        <textarea
          id="chat-input" class="textarea" rows="1"
          placeholder="Ask a question about your documents..."
        ></textarea>
        <button type="submit" class="btn btn-primary" id="send-btn">Send</button>
      </form>
    </div>
  `;

  const newChatBtn = container.querySelector("#new-chat-btn");
  const clearChatBtn = container.querySelector("#clear-chat-btn");
  const exportBtn = container.querySelector("#export-btn");
  const docScopeContainer = container.querySelector("#doc-scope-container");
  const docScopeSelect = createDocumentScopeSelect({
    onChange: (ids) => {
      documentScope = ids;
    },
  });
  docScopeContainer.appendChild(docScopeSelect);
  const messagesEl = container.querySelector("#chat-messages");
  const form = container.querySelector("#chat-form");
  const input = container.querySelector("#chat-input");
  const sendBtn = container.querySelector("#send-btn");

  // --- Input UX: grow with content, Enter to send, Shift+Enter for newline ---
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  });
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const query = input.value.trim();
    if (!query || streamController) return;
    input.value = "";
    input.style.height = "auto";
    sendMessage(query);
  });

  // --- Toolbar wiring --------------------------------------------------
  newChatBtn.addEventListener("click", async () => {
    newChatBtn.disabled = true;
    try {
      const session = await chatApi.createSession("New Conversation");
      sessionId = session.id;
      localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
      documentScope = [];
      docScopeSelect.clearSelection();
      messages = [];
      renderMessages();
    } catch (err) {
      showToast(`Couldn't start a new chat: ${err.message}`, { type: "error" });
    } finally {
      newChatBtn.disabled = false;
    }
  });

  clearChatBtn.addEventListener("click", async () => {
    clearChatBtn.disabled = true;
    try {
      await chatApi.clearConversation(sessionId);
      messages = [];
      renderMessages();
    } catch (err) {
      showToast(`Couldn't clear the conversation: ${err.message}`, { type: "error" });
    } finally {
      clearChatBtn.disabled = false;
    }
  });

  exportBtn.addEventListener("click", openExportModal);

  // --- Session bootstrap -----------------------------------------------
  async function ensureActiveSession() {
    const storedId = localStorage.getItem(SESSION_STORAGE_KEY);
    if (storedId) {
      try {
        messages = await chatApi.getMessages(storedId);
        sessionId = storedId;
        return;
      } catch {
        // Session no longer exists server-side — fall through and create one.
      }
    }
    const session = await chatApi.createSession("New Conversation");
    sessionId = session.id;
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    messages = [];
  }

  // --- Message list rendering --------------------------------------------
  function renderMessages() {
    if (disposed) return;
    exportBtn.hidden = messages.length === 0;

    if (messages.length === 0) {
      messagesEl.innerHTML = `
        <div class="empty-state">
          Ask a question about your uploaded documents to get started — answers are
          grounded in your documents, with citations.
        </div>
      `;
      return;
    }

    messagesEl.innerHTML = "";
    messages.forEach((message, index) => {
      const isLastAssistant =
        message.role === "assistant" && index === messages.length - 1;
      messagesEl.appendChild(renderMessageBubble(message, isLastAssistant));
    });
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function renderMessageBubble(message, isLastAssistant) {
    const bubble = document.createElement("div");
    bubble.className = `chat-message chat-message--${message.role}`;

    const contentEl = document.createElement("p");
    contentEl.style.whiteSpace = "pre-wrap";
    contentEl.style.margin = "0";
    contentEl.textContent = message.content;
    bubble.appendChild(contentEl);

    if (message.role === "assistant" && !message._streaming) {
      if (message.citations?.length) {
        bubble.appendChild(renderCitationsPanel(message.citations));
      }
      bubble.appendChild(renderMessageActions(message, isLastAssistant));
    }

    return bubble;
  }

  function renderCitationsPanel(citations) {
    const details = document.createElement("details");
    details.className = "citations-panel";
    const count = citations.length;
    details.innerHTML = `
      <summary>📎 ${count} source${count !== 1 ? "s" : ""}</summary>
      ${citations
        .map((citation) => {
          const pageInfo = citation.page_number ? `, page ${citation.page_number}` : "";
          return `
            <div class="citation-item">
              <p class="citation-title">
                <strong>${escapeHtml(citation.filename)}${pageInfo}</strong>
                ${renderMatchDial(citation.similarity_score)}
              </p>
              <p class="citation-excerpt">${escapeHtml(citation.excerpt)}</p>
            </div>
          `;
        })
        .join("")}
    `;
    return details;
  }

  function renderMessageActions(message, isLastAssistant) {
    const actions = document.createElement("div");
    actions.className = "message-actions";

    const copyBtn = document.createElement("button");
    copyBtn.className = "btn btn-icon";
    copyBtn.title = "Copy answer";
    copyBtn.setAttribute("aria-label", "Copy answer");
    copyBtn.textContent = "📋";
    copyBtn.addEventListener("click", async () => {
      await navigator.clipboard.writeText(message.content);
      copyBtn.textContent = "✅";
      setTimeout(() => {
        copyBtn.textContent = "📋";
      }, 1500);
    });
    actions.appendChild(copyBtn);

    if (isLastAssistant) {
      const regenerateBtn = document.createElement("button");
      regenerateBtn.className = "btn btn-icon";
      regenerateBtn.title = "Regenerate response";
      regenerateBtn.setAttribute("aria-label", "Regenerate response");
      regenerateBtn.textContent = "🔄";
      regenerateBtn.addEventListener("click", () => regenerate(regenerateBtn));
      actions.appendChild(regenerateBtn);
    }

    return actions;
  }

  // --- Sending + streaming -------------------------------------------------
  function sendMessage(query) {
    messages.push({ role: "user", content: query, citations: [] });
    const assistantDraft = { role: "assistant", content: "", citations: [], _streaming: true };
    messages.push(assistantDraft);
    renderMessages();

    input.disabled = true;
    sendBtn.disabled = true;

    const bubble = messagesEl.lastElementChild;
    const contentEl = bubble.querySelector("p");
    contentEl.classList.add("typing-cursor");

    let capturedCitations = [];

    streamController = chatApi.streamMessage(
      sessionId,
      { query, document_ids: documentScope.length ? documentScope : null },
      {
        citations: (data) => {
          try {
            capturedCitations = JSON.parse(data);
          } catch {
            capturedCitations = [];
          }
        },
        token: (data) => {
          assistantDraft.content += data;
          contentEl.textContent = assistantDraft.content;
          messagesEl.scrollTop = messagesEl.scrollHeight;
        },
        error: (message) => {
          contentEl.classList.remove("typing-cursor");
          bubble.insertAdjacentHTML(
            "beforeend",
            `<div class="alert alert-error">Something went wrong answering that question: ${escapeHtml(message)}</div>`
          );
        },
        done: async () => {
          streamController = null;
          input.disabled = false;
          sendBtn.disabled = false;
          input.focus();
          contentEl.classList.remove("typing-cursor");
          // Refetch from the server so IDs/timestamps/citations are
          // all the persisted, canonical values rather than this
          // client's in-progress draft.
          try {
            messages = await chatApi.getMessages(sessionId);
          } catch {
            assistantDraft.citations = capturedCitations;
          }
          renderMessages();
        },
      }
    );
  }

  async function regenerate(button) {
    button.disabled = true;
    const bubble = button.closest(".chat-message");
    const contentEl = bubble.querySelector("p");
    const previousContent = contentEl.textContent;
    contentEl.textContent = "Regenerating...";
    contentEl.classList.add("text-muted");

    try {
      await chatApi.regenerate(sessionId, {
        document_ids: documentScope.length ? documentScope : null,
      });
      messages = await chatApi.getMessages(sessionId);
      renderMessages();
    } catch (err) {
      contentEl.textContent = previousContent;
      contentEl.classList.remove("text-muted");
      button.disabled = false;
      bubble.insertAdjacentHTML(
        "beforeend",
        `<div class="alert alert-error">Couldn't regenerate the response: ${escapeHtml(err.message)}</div>`
      );
    }
  }

  // --- Export modal --------------------------------------------------------
  function openExportModal() {
    const { body } = openModal({ title: "Export Conversation" });
    body.innerHTML = `
      <div class="export-actions">
        <button class="btn" data-action="pdf">⬇️ Download as PDF</button>
        <button class="btn" data-action="markdown">⬇️ Download as Markdown</button>
        <button class="btn" data-action="copy">📋 Copy conversation</button>
        <p id="export-feedback" class="text-muted" style="margin: 0"></p>
      </div>
    `;
    const feedback = body.querySelector("#export-feedback");

    body.querySelector('[data-action="pdf"]').addEventListener("click", async () => {
      try {
        await chatApi.exportConversation(sessionId, "pdf");
      } catch (err) {
        feedback.textContent = `Couldn't export: ${err.message}`;
      }
    });
    body.querySelector('[data-action="markdown"]').addEventListener("click", async () => {
      try {
        await chatApi.exportConversation(sessionId, "markdown");
      } catch (err) {
        feedback.textContent = `Couldn't export: ${err.message}`;
      }
    });
    body.querySelector('[data-action="copy"]').addEventListener("click", async () => {
      try {
        const text = await chatApi.getConversationText(sessionId);
        await navigator.clipboard.writeText(text);
        feedback.textContent = "Copied to clipboard.";
      } catch (err) {
        feedback.textContent = `Couldn't copy: ${err.message}`;
      }
    });
  }

  // --- Boot ------------------------------------------------------------
  (async () => {
    messagesEl.innerHTML = `<div class="row"><span class="spinner"></span><span>Loading conversation...</span></div>`;
    try {
      await ensureActiveSession();
      renderMessages();
    } catch (err) {
      messagesEl.innerHTML = `<div class="alert alert-error">Couldn't load this conversation: ${escapeHtml(err.message)}</div>`;
    }
  })();

  return function dispose() {
    disposed = true;
    streamController?.abort();
  };
}
