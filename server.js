
import http from "node:http";
import { Agent, setGlobalDispatcher } from "undici";

const PORT = process.env.PORT || 3000;

setGlobalDispatcher(new Agent({
  headersTimeout: 90000,
  bodyTimeout: 90000,
  connectTimeout: 15000
}));

const server = http.createServer(async (req, res) => {
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    return res.end();
  }

  const pathname = new URL(req.url, "http://localhost").pathname;

  if (req.method === "GET" && pathname === "/") {
    res.writeHead(200);
    return res.end(JSON.stringify({
      status: "healthy",
      service: "BM AI",
      provider: "Groq",
      version: "1.0.0"
    }));
  }

  if (req.method !== "POST" || pathname !== "/api/chat") {
    res.writeHead(404);
    return res.end(JSON.stringify({
      error: "Not found"
    }));
  }

  const apiKey = process.env.GROQ_API_KEY;

  if (!apiKey) {
    res.writeHead(500);
    return res.end(JSON.stringify({
      error: "Groq API key is not configured"
    }));
  }

  try {
    let body = "";

    for await (const chunk of req) {
      body += chunk;

      if (body.length > 100000) {
        res.writeHead(413);
        return res.end(JSON.stringify({
          error: "Request too large"
        }));
      }
    }

    let parsed;

    try {
      parsed = JSON.parse(body || "{}");
    } catch {
      res.writeHead(400);
      return res.end(JSON.stringify({
        error: "Invalid JSON"
      }));
    }

    const message = parsed.message;

    if (typeof message !== "string" || !message.trim()) {
      res.writeHead(400);
      return res.end(JSON.stringify({
        error: "Please provide a message"
      }));
    }

    const response = await fetch(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey}`
        },
        body: JSON.stringify({
          model: "openai/gpt-oss-20b",
          messages: [
            {
              role: "system",
              content: "You are BM AI, a helpful, accurate and friendly AI assistant. Give clear and useful answers."
            },
            {
              role: "user",
              content: message.trim()
            }
          ],
          max_completion_tokens: 1024
        }),
        signal: AbortSignal.timeout(80000)
      }
    );

    const responseText = await response.text();

    let data;

    try {
      data = JSON.parse(responseText);
    } catch {
      throw new Error("Groq returned an invalid response.");
    }

    if (!response.ok) {
      console.error("Groq API error:", data.error?.message);

      const status = response.status === 429
        ? 429
        : response.status >= 500
          ? 503
          : 502;

      res.writeHead(status);
      return res.end(JSON.stringify({
        error: data.error?.message || "Groq request failed"
      }));
    }

    const answer =
      data.choices?.[0]?.message?.content || "No response received.";

    res.writeHead(200);
    return res.end(JSON.stringify({
      answer: answer
    }));

  } catch (error) {
    console.error("BM AI error:", error);

    const timedOut =
      error.name === "TimeoutError" ||
      error.name === "AbortError" ||
      error.code === "UND_ERR_HEADERS_TIMEOUT" ||
      error.code === "UND_ERR_BODY_TIMEOUT";

    res.writeHead(timedOut ? 504 : 500);

    return res.end(JSON.stringify({
      error: timedOut
        ? "Groq is taking too long to respond. Please try again."
        : "Unable to process your request. Please try again."
    }));
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`BM AI server running on port ${PORT}`);
});
  
