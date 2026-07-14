/**
 * api/client.js
 * ==============
 * Every later phase (F-J) talks to the backend exclusively through
 * this module — no page should call `fetch` directly. That keeps the
 * error-envelope parsing, base-URL resolution, and SSE framing logic
 * in exactly one place.
 *
 * Base URL: defaults to the relative path "/api/v1", which works
 * unchanged in both environments:
 *   - dev: Vite's proxy (vite.config.js) forwards /api/* to FastAPI,
 *     so the browser never makes a cross-origin request at all.
 *   - prod: FastAPI serves the built frontend and the API from the
 *     same origin (Phase K), so "/api/v1" is already correct.
 * Override with VITE_API_BASE_URL only for the unusual case of
 * pointing a local frontend at a separately-deployed backend.
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api/v1";

/**
 * Thrown for every non-2xx response. Mirrors the `{"error": {"type",
 * "message"}}` envelope every route in api/errors.py returns, so a
 * single `catch` block anywhere in the app can branch on `.type`
 * (e.g. show a "session not found" empty state for
 * `RecordNotFoundError` vs. a generic toast for anything else).
 */
export class ApiError extends Error {
  constructor(status, type, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.type = type;
  }
}

async function parseErrorBody(response) {
  try {
    const body = await response.json();
    if (body?.error?.type && body?.error?.message) {
      return new ApiError(response.status, body.error.type, body.error.message);
    }
  } catch {
    /* response wasn't JSON (e.g. a proxy 502) — fall through */
  }
  return new ApiError(response.status, "UnknownError", `Request failed with status ${response.status}`);
}

async function request(path, { method = "GET", json, body, headers = {}, signal } = {}) {
  const init = { method, headers: { ...headers }, signal };

  if (json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(json);
  } else if (body !== undefined) {
    init.body = body; // FormData — let the browser set the multipart boundary itself
  }

  const response = await fetch(`${BASE_URL}${path}`, init);

  if (!response.ok) {
    throw await parseErrorBody(response);
  }

  if (response.status === 204) {
    return null;
  }

  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  return response; // caller handles blobs/text (e.g. export downloads)
}

export const api = {
  get: (path, options) => request(path, { ...options, method: "GET" }),
  post: (path, json, options) => request(path, { ...options, method: "POST", json }),
  put: (path, json, options) => request(path, { ...options, method: "PUT", json }),
  delete: (path, options) => request(path, { ...options, method: "DELETE" }),

  /** Multipart upload — pass a FormData with one or more "files" entries. */
  upload: (path, formData, options) => request(path, { ...options, method: "POST", body: formData }),

  /**
   * Downloads a non-JSON response (export endpoints) as a Blob and
   * triggers a browser save-as, reading the filename FastAPI already
   * set in Content-Disposition rather than guessing one client-side.
   */
  async download(path, options) {
    const response = await request(path, { ...options, method: "GET" });
    const blob = await response.blob();
    const disposition = response.headers.get("content-disposition") || "";
    const match = disposition.match(/filename="?([^"]+)"?/);
    const filename = match ? match[1] : "download";

    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  },

  /**
   * Consumes a Server-Sent Events endpoint (the streaming chat route)
   * via `fetch` + a manual `ReadableStream` reader rather than the
   * built-in `EventSource`, because `EventSource` can only send GET
   * requests with no body — the chat stream needs a POST with a JSON
   * question. Frames look like:
   *
   *   event: citations
   *   data: [...]
   *
   *   event: token
   *   data: Hello
   *
   * `handlers` maps event name -> callback(dataString). Returns an
   * `AbortController` the caller can use to cancel mid-stream (e.g.
   * the user navigates away or clicks "stop generating").
   */
  streamSSE(path, json, handlers) {
    const controller = new AbortController();

    (async () => {
      let response;
      try {
        response = await fetch(`${BASE_URL}${path}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(json),
          signal: controller.signal,
        });
      } catch (err) {
        if (err.name !== "AbortError") handlers.error?.(err.message);
        return;
      }

      if (!response.ok) {
        const apiError = await parseErrorBody(response);
        handlers.error?.(apiError.message);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      try {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          // SSE frames are separated by a blank line.
          let boundary;
          while ((boundary = buffer.indexOf("\n\n")) !== -1) {
            const frame = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            _dispatchSSEFrame(frame, handlers);
          }
        }
      } catch (err) {
        if (err.name !== "AbortError") handlers.error?.(err.message);
      }
    })();

    return controller;
  },
};

function _dispatchSSEFrame(frame, handlers) {
  let eventName = "message";
  const dataLines = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice("event:".length).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice("data:".length).trim());
  }
  handlers[eventName]?.(dataLines.join("\n"));
}
