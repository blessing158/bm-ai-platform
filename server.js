
import http from "node:http";
import { Agent, setGlobalDispatcher } from "undici";

const PORT = process.env.PORT || 3000;

// BM AI's own model configuration
const BM_MODEL_URL = process.env.BM_AI_MODEL_URL;
const BM_MODEL_KEY = process.env.BM_AI_MODEL_API_KEY;
const BM_MODEL_NAME =
  process.env.BM_AI_MODEL_NAME || "bm-ai-1m";

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
- Use BM.DEVELOPERS' own model infrastructure only.
- Never claim another company created BM AI.
- Never claim the model has been trained beyond what is verified.

CONVERSATION:
- Use the supplied conversation history.
- Understand follow-up questions.
- Do not pretend to remember conversations that were not supplied.
- Do not ask users to repeat information already provided.

RESPONSE STYLE:
- Answer the exact question.
- Give the main answer first.
- Be concise, natural and helpful.
- Use simple language.
- Give detailed explanations when requested.
- Never invent facts.
- Admit when you do not know.

ACCURACY:
- Never invent sources, URLs, citations or achievements.
- Clearly distinguish verified facts from uncertainty.
- Be honest about your capabilities and limitations.

MAIN RULE:
Give users accurate and useful information.
Always represent BM AI and BM.DEVELOPERS accurately.
`;

function send(res, status, data) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8"
  });

  res.end(JSON.stringify(data));
}

function extractOwnModelText(data) {
  const answer = data.choices?.[0]?.message?.content;

  if (typeof answer === "string") {
    return answer.trim();
  }

  if (Array.isArray(answer)) {
    return answer
      .filter(part => typeof part.text === "string")
      .map(part => part.text)
      .join("\n")
      .trim();
  }

  return "";
}

// Connect to BM AI's Python model.
async function callOwnModel(message, history) {
  if (!BM_MODEL_URL) {
    throw new Error("BM AI model URL is not configured.");
  }

  const messages = [
    {
      role: "system",
      content: SYSTEM_PROMPT
    },
    ...history,
    {
      role: "user",
      content: message
    }
  ];

  const headers = {
    "Content-Type": "application/json"
  };

  if (BM_MODEL_KEY) {
    headers.Authorization = "Bearer " + BM_MODEL_KEY;
  }

  const endpoint =
    BM_MODEL_URL.replace(/\/+$/, "") +
    "/v1/chat/completions";

  const response = await fetch(endpoint, {
    method: "POST",
    headers: headers,
    body: JSON.stringify({
      model: BM_MODEL_NAME,
      messages: messages,
      max_tokens: 2048,
      temperature: 0.7
    }),
    signal: AbortSignal.timeout(80000)
  });

  let data;

  try {
    data = await response.json();
  } catch {
    throw new Error("BM AI model returned an invalid response.");
  }

  if (!response.ok) {
    console.error(
      "BM AI model API error:",
      data.error?.message || response.status
    );

    throw new Error("BM AI model request failed.");
  }

  const answer = extractOwnModelText(data);

  if (!answer) {
    throw new Error("BM AI model returned an empty response.");
  }

  return answer;
}

// All requests use BM AI's own model.
// There is NO Gemini or third-party AI fallback.
async function generateAnswer(message, history) {
  const answer = await callOwnModel(message, history);

  return {
    answer: answer,
    provider: "bm-ai",
    fallbackUsed: false
  };
}

const server = http.createServer(async (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader(
    "Access-Control-Allow-Methods",
    "GET, POST, OPTIONS"
  );
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type"
  );

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    return res.end();
  }

  const pathname = new URL(
    req.url,
    "http://localhost"
  ).pathname;

  // Health check
  if (req.method === "GET" && pathname === "/") {
    return send(res, 200, {
      status: "healthy",
      service: "BM AI",
      providers: {
        bmAi: !!BM_MODEL_URL
      },
      automaticRouting: false,
      fallbackEnabled: false,
      vision: false,
      webSearch: false,
      conversationMemory: true,
      version: "4.0.0"
    });
  }

  if (req.method !== "POST" || pathname !== "/api/chat") {
    return send(res, 404, {
      error: "Not found"
    });
  }

  try {
    const chunks = [];
    let totalSize = 0;

    for await (const chunk of req) {
      totalSize += chunk.length;

      if (totalSize > 12 * 1024 * 1024) {
        return send(res, 413, {
          error: "Request too large."
        });
      }

      chunks.push(chunk);
    }

    let parsed;

    try {
      parsed = JSON.parse(
        Buffer.concat(chunks).toString("utf8") || "{}"
      );
    } catch {
      return send(res, 400, {
        error: "Invalid JSON."
      });
    }

    const message = parsed.message;
    const history = parsed.history || [];

    // Images are not supported by the current model.
    if (parsed.imageData) {
      return send(res, 400, {
        error: "Image input is not supported by BM AI yet."
      });
    }

    if (
      typeof message !== "string" ||
      !message.trim()
    ) {
      return send(res, 400, {
        error: "Please provide a message."
      });
    }

    if (!Array.isArray(history)) {
      return send(res, 400, {
        error: "Invalid conversation history."
      });
    }

    const recentHistory = history.slice(-20);
    const conversationMessages = [];

    for (const item of recentHistory) {
      if (
        !item ||
        (item.role !== "user" && item.role !== "assistant") ||
        typeof item.content !== "string" ||
        !item.content.trim() ||
        item.content.length > 4000
      ) {
        return send(res, 400, {
          error: "Invalid conversation history."
        });
      }

      conversationMessages.push({
        role: item.role,
        content: item.content
      });
    }

    const historyLength = conversationMessages.reduce(
      (total, item) => total + item.content.length,
      0
    );

    if (historyLength > 30000) {
      return send(res, 400, {
        error: "Conversation too long. Start a new chat."
      });
    }

    const result = await generateAnswer(
      message.trim(),
      conversationMessages
    );

    return send(res, 200, {
      answer: result.answer,
      provider: result.provider,
      automaticRouting: false,
      fallbackUsed: false
    });

  } catch (error) {
    console.error("BM AI error:", error.message);

    const timedOut =
      error.name === "TimeoutError" ||
      error.name === "AbortError" ||
      error.code === "UND_ERR_HEADERS_TIMEOUT" ||
      error.code === "UND_ERR_BODY_TIMEOUT";

    return send(res, timedOut ? 504 : 502, {
      error: timedOut
        ? "BM AI is taking too long. Please try again."
        : "BM AI's own model could not process your request."
    });
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log("BM AI v4.0.0 running on port " + PORT);
  console.log("BM AI own model configured: " + !!BM_MODEL_URL);
  console.log("Third-party AI providers: disabled");
  console.log("Gemini integration: removed");
  console.log("Fallback: disabled");
});
      
