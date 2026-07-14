/**
 * api/resources.js
 * ================
 * One named function per backend endpoint (Phases A-D). Pages import
 * from here, never from `client.js` directly — this is the one file
 * that needs to change if a route's path or shape ever changes.
 */

import { api } from "./client.js";

export const documents = {
  upload: (files, { chunkSize, chunkOverlap } = {}) => {
    const formData = new FormData();
    for (const file of files) formData.append("files", file);
    const params = new URLSearchParams();
    if (chunkSize != null) params.set("chunk_size", chunkSize);
    if (chunkOverlap != null) params.set("chunk_overlap", chunkOverlap);
    const query = params.toString() ? `?${params}` : "";
    return api.upload(`/documents/upload${query}`, formData);
  },
  list: ({ status, collectionId } = {}) => {
    const params = new URLSearchParams();
    if (status) params.set("status", status);
    if (collectionId) params.set("collection_id", collectionId);
    const query = params.toString() ? `?${params}` : "";
    return api.get(`/documents${query}`);
  },
  get: (documentId) => api.get(`/documents/${documentId}`),
  preview: (documentId, maxChars = 2000) =>
    api.get(`/documents/${documentId}/preview?max_chars=${maxChars}`),
  remove: (documentId) => api.delete(`/documents/${documentId}`),
  summarize: (documentId) => api.post(`/documents/${documentId}/summary`),
};

export const chat = {
  createSession: (title, collectionId) => api.post("/chat/sessions", { title, collection_id: collectionId }),
  listSessions: () => api.get("/chat/sessions"),
  getMessages: (sessionId) => api.get(`/chat/sessions/${sessionId}/messages`),
  clearConversation: (sessionId) => api.delete(`/chat/sessions/${sessionId}/messages`),
  deleteSession: (sessionId) => api.delete(`/chat/sessions/${sessionId}`),
  sendMessage: (sessionId, body) => api.post(`/chat/sessions/${sessionId}/messages`, body),
  regenerate: (sessionId, body) => api.post(`/chat/sessions/${sessionId}/regenerate`, body),
  /** Returns an AbortController; see api.streamSSE for the handlers contract. */
  streamMessage: (sessionId, body, handlers) =>
    api.streamSSE(`/chat/sessions/${sessionId}/messages/stream`, body, handlers),
  exportConversation: (sessionId, format = "markdown") =>
    api.download(`/chat/sessions/${sessionId}/export?format=${format}`),
  /** Fetches the plain-text export as a string (for "copy conversation"), rather than triggering a file download. */
  getConversationText: (sessionId) =>
    api.get(`/chat/sessions/${sessionId}/export?format=text`).then((response) => response.text()),
};

export const search = {
  run: (query, { topK, documentIds } = {}) =>
    api.post("/search", { query, top_k: topK, document_ids: documentIds }),
};

export const analytics = {
  getSummary: ({ mostQueriedLimit, activityDays } = {}) => {
    const params = new URLSearchParams();
    if (mostQueriedLimit != null) params.set("most_queried_limit", mostQueriedLimit);
    if (activityDays != null) params.set("activity_days", activityDays);
    const query = params.toString() ? `?${params}` : "";
    return api.get(`/analytics/summary${query}`);
  },
};

export const settings = {
  get: () => api.get("/settings"),
  update: (partial) => api.put("/settings", partial),
  reset: () => api.post("/settings/reset"),
  listModels: () => api.get("/settings/models"),
};

export const health = {
  check: () => api.get("/health"),
  checkConfig: () => api.get("/health/config"),
};
