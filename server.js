
import http from "node:http";
import { Agent, setGlobalDispatcher } from "undici";

const PORT = process.env.PORT || 3000;

setGlobalDispatcher(new Agent({
  headersTimeout: 90000,
  bodyTimeout: 90000,
  connectTimeout: 15000
}));

const SYSTEM_PROMPT = `
You are BM AI, the official AI assistant developed by BM.DEVELOPERS.

IDENTITY:
- Your name is BM AI.
- Your developer is BM.DEVELOPERS.
- You use AI models provided through Groq.
- You are not ChatGPT.
- Never claim OpenAI created BM AI.
- Never claim BM.DEVELOPERS trained the underlying AI model.

WHEN ASKED WHO CREATED YOU:
Explain that BM AI was developed by BM.DEVELOPERS
and uses AI models provided through Groq.

RESPONSE STYLE:
- Answer the exact question asked.
- Be concise, direct and relevant.
- Give the main answer first.
- Use simple, natural language.
- Default to 1-3 short paragraphs.
- Do not add unrelated information.
- Do not repeat the user's question.
- Avoid unnecessary introductions and conclusions.
- Use lists only when they improve clarity.
- Give detailed explanations when requested.
- Ask one clarification if the question is unclear.
- Never invent facts.
- Admit when you do not know something.

WEB SEARCH:
- Use browser search when information may be current,
  unfamiliar, difficult to verify or about a less-known
  public person, musician, business or event.
- Do not search for every simple general-knowledge question.
- Prefer reliable and relevant sources.
- Do not confuse people with similar names.
- If reliable information cannot be found, say so.
- Never invent biographies, achievements or facts.
- Mention useful source names or URLs when available.
- Clearly distinguish verified facts from uncertainty.

IMAGE QUESTIONS:
- Carefully inspect uploaded images.
- Answer the specific question about the image.
- Avoid unrelated descriptions.

MAIN RULE:
Give users the information they need without unnecessary
extra content.

Always represent BM AI and BM.DEVELOPERS accurately.
`;

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
      webSearch: true,
      vision: true,
      version: "1.3.1"
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

    const chunks = [];
    let totalSize = 0;

    for await (const chunk of req) {

      totalSize += chunk.length;

      if (totalSize > 12 * 1024 * 1024) {
        res.writeHead(413);
        return res.end(JSON.stringify({
          error: "Request too large. Please choose a smaller photo."
        }));
      }

      chunks.push(chunk);
    }

    const body = Buffer.concat(chunks).toString("utf8");

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
    const imageData = parsed.imageData;

    if (typeof message !== "string" || !message.trim()) {
      res.writeHead(400);
      return res.end(JSON.stringify({
        error: "Please provide a message"
      }));
    }

    let model = "openai/gpt-oss-20b";
    let userContent = message.trim();

    if (imageData) {

      if (typeof imageData !== "string") {
        res.writeHead(400);
        return res.end(JSON.stringify({
          error: "Invalid image data"
        }));
      }

      const match = imageData.match(
        /^data:(image\/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=]+)$/
      );

      if (!match) {
        res.writeHead(400);
        return res.end(JSON.stringify({
          error: "Unsupported image format. Use JPG, PNG or WEBP."
        }));
      }

      const mimeType = match[1];
      const base64 = match[2];

      if (base64.length > 10 * 1024 * 1024) {
        res.writeHead(413);
        return res.end(JSON.stringify({
          error: "Photo is too large. Please choose a smaller photo."
        }));
      }

      model = "qwen/qwen3.8-27b";

      userContent = [
        {
          type: "text",
          text: message.trim()
        },
        {
          type: "image_url",
          image_url: {
            url: "data:" + mimeType + ";base64," + base64
          }
        }
      ];
    }

    const requestBody = {
      model: model,
      messages: [
        {
          role: "system",
          content: SYSTEM_PROMPT
        },
        {
          role: "user",
          content: userContent
        }
      ],
      max_completion_tokens: 1024
    };

    // Browser search is enabled for text questions only.
    if (!imageData) {
      requestBody.tools = [
        {
          type: "browser_search"
        }
      ];

      requestBody.reasoning_effort = "low";
    }

    const response = await fetch(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${apiKey}`
        },
        body: JSON.stringify(requestBody),
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
      data.choices?.[0]?.message?.content ||
      "No response received.";

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
        ? "BM AI is taking too long to respond. Please try again."
        : "Unable to process your request. Please try again."
    }));
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log("BM AI v1.3.1 server running on port " + PORT);
});
  
