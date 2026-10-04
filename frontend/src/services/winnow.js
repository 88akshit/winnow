async function requestJson(path, options) {
  const response = await fetch(path, options);
  let result = {};
  try {
    result = (await response.json()) || {};
  } catch {
    result.detail = `The server returned an unreadable response (${response.status}).`;
  }
  if (!response.ok) {
    const error = new Error(result.detail || `Request failed (${response.status}).`);
    error.status = response.status;
    error.retryAfter = response.headers.get("Retry-After");
    throw error;
  }
  return result;
}

export function uploadDataset(file) {
  const body = new FormData();
  body.append("file", file);
  return requestJson("/api/upload", { method: "POST", body });
}

export function chatWithWinnow(message, threadId) {
  return requestJson("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, thread_id: threadId }),
  });
}

export function resumeChat(answer, threadId) {
  return requestJson("/api/chat/resume", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answer, thread_id: threadId }),
  });
}
