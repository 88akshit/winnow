async function requestJson(path, options) {
  const response = await fetch(path, options);
  let result = {};
  try {
    result = (await response.json()) || {};
  } catch {
    result.detail = `The server returned an unreadable response (${response.status}).`;
  }
  if (!response.ok) {
    throw new Error(result.detail || `Request failed (${response.status}).`);
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
