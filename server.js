
import http from "node:http";
import { Agent, setGlobalDispatcher } from "undici";

const PORT = process.env.PORT || 3000;

// Google Gemini
const GEMINI_KEY = process.env.GEMINI_API_KEY;
const GEMINI_MODEL =
  process.env.GEMINI_MODEL || "gemini-3.5-flash-lite";

// Future independent BM AI model.
// This must point to an OpenAI-compatible model server.
const BM_MODEL_URL = process.env.BM_AI_MODEL_URL;
const BM_MODEL_KEY = process.env.BM_AI_MODEL_API_KEY;
const BM_MODEL_NAME =
  process.env.BM_AI_MODEL_NAME || "qwen2.5-3b-instruct";

// Search is disabled unless explicitly enabled.
const WEB_SEARCH_ENABLED =
  process.env.BM_AI_WEB_SEARCH_ENABLED === "true";

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
- You may use BM.DEVELOPERS' own model infrastructure and Google Gemini.
- Never claim OpenAI created BM AI.
- Never claim BM.DEVELOPERS trained an underlying model unless verified.

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

WEB RESEARCH:
- Use search tools when available and useful.
- Prefer reliable sources.
- Never invent URLs, citations, biographies or achievements.
- Clearly distinguish verified facts from uncertainty.
- When you cannot verify something, say so.

IMAGE QUESTIONS:
- Carefully inspect uploaded images.
- Answer the user's specific question about the image.

MAIN RULE:
Give users accurate and useful information.

Always represent BM AI and BM.DEVELOPERS accurately.
`;

function send(res, status, data) {
  res.writeHead(status);
  res.end(JSON.stringify(data));
}

function isValidImage(imageData) {
  return typeof imageData === "string" &&
    /^data:image\/(jpeg|png|webp);base64,[A-Za-z0-9+/=]+$/.test(imageData);
}

function extractGeminiText(data) {
  const parts =
    data.candidates?.[0]?.content?.parts || [];

  return parts
    .filter(part => typeof part.text === "string")
    .map(part => part.text)
    .join("\n")
    .trim();
}

function extractGeminiSources(data) {
  const chunks =
    data.candidates?.[0]?.groundingMetadata?.groundingChunks || [];

  return chunks
    .filter(item => item.web?.uri)
    .map(item => ({
      title: item.web.title || item.web.uri,
      url: item.web.uri
    }));
}

function appendSources(answer, sources) {
  const unique = [];
  const seen = new Set();

  for (const source of sources) {
    if (!source.url || seen.has(source.url)) {
      continue;
    }

    seen.add(source.url);
    unique.push(source);
  }

  if (unique.length === 0) {
    return answer;
  }

  return answer + "\n\nSources:\n" +
    unique.slice(0, 6).map(source =>
      "- " + source.title + ": " + source.url
    ).join("\n");
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

// BM AI's future independent model.
async function callOwnModel(message, history) {
  if (!BM_MODEL_URL) {
    throw new Error("BM AI model is not configured.");
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

  const data = await response.json();

  if (!response.ok) {
    console.error(
      "BM model API error:",
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

// Google Gemini.
async function callGemini(message, history, imageData) {
  if (!GEMINI_KEY) {
    throw new Error("Gemini API key is not configured.");
  }

  const contents = [];

  for (const item of history) {
    contents.push({
      role: item.role === "assistant" ? "model" : "user",
      parts: [{ text: item.content }]
    });
  }

  const currentParts = [
    { text: message }
  ];

  if (imageData) {
    if (!isValidImage(imageData)) {
      throw new Error("Unsupported image format.");
    }

    const match = imageData.match(
      /^data:(image\/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=]+)$/
    );

    currentParts.push({
      inlineData: {
        mimeType: match[1],
        data: match[2]
      }
    });
  }

  contents.push({
    role: "user",
    parts: currentParts
  });

  const requestBody = {
    systemInstruction: {
      parts: [{ text: SYSTEM_PROMPT }]
    },
    contents: contents,
    generationConfig: {
      maxOutputTokens: 2048
    }
  };

  if (WEB_SEARCH_ENABLED && !imageData) {
    requestBody.tools = [
      { google_search: {} }
    ];
  }

  const response = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/models/" +
      GEMINI_MODEL + ":generateContent",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_KEY
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(80000)
    }
  );

  const data = await response.json();

  if (!response.ok) {
    console.error(
      "Gemini API error:",
      data.error?.message || response.status
    );

    throw new Error("Gemini request failed.");
  }

  const answer = extractGeminiText(data);

  if (!answer) {
    throw new Error("Gemini returned an empty response.");
  }

  return appendSources(
    answer,
    extractGeminiSources(data)
  );
}

async function generateAnswer(message, history, imageData) {

  // Images currently go directly to Gemini.
  if (imageData) {
    return {
      answer: await callGemini(
        message,
        history,
        imageData
      ),
      provider: "gemini",
      fallbackUsed: false
    };
  }

  // If our own model is configured, try it first.
  if (BM_MODEL_URL) {
    try {
      const answer = await callOwnModel(
        message,
        history
      );

      return {
        answer: answer,
        provider: "bm-ai",
        fallbackUsed: false
      };

    } catch (error) {
      console.error(
        "BM AI model failed:",
        error.message
      );

      // Gemini becomes the fallback.
      const answer = await callGemini(
        message,
        history,
        null
      );

      return {
        answer: answer,
        provider: "gemini",
        fallbackUsed: true
      };
    }
  }

  // Until our model is hosted, use Gemini.
  return {
    answer: await callGemini(
      message,
      history,
      null
    ),
    provider: "gemini",
    fallbackUsed: false
  };
}

const server = http.createServer(async (req, res) => {

  res.setHeader(
    "Content-Type",
    "application/json; charset=utf-8"
  );

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

  // Health check.
  if (req.method === "GET" && pathname === "/") {
    return send(res, 200, {
      status: "healthy",
      service: "BM AI",
      providers: {
        bmAi: !!BM_MODEL_URL,
        gemini: !!GEMINI_KEY
      },
      automaticRouting: true,
      webSearch: WEB_SEARCH_ENABLED,
      vision: true,
      conversationMemory: true,
      version: "3.0.0"
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
          error: "Request too large. Choose a smaller photo."
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
    const imageData = parsed.imageData;
    const history = parsed.history || [];

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

    if (imageData && !isValidImage(imageData)) {
      return send(res, 400, {
        error: "Unsupported image format. Use JPG, PNG or WEBP."
      });
    }

    if (imageData && imageData.length > 10 * 1024 * 1024) {
      return send(res, 413, {
        error: "Photo is too large."
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
      conversationMessages,
      imageData
    );

    return send(res, 200, {
      answer: result.answer,
      provider: result.provider,
      automaticRouting: true,
      fallbackUsed: result.fallbackUsed
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
        : "Unable to process your request. Check your AI provider configuration."
    });
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(
    "BM AI v3.0.0 running on port " + PORT
  );

  console.log(
    "Gemini configured: " + !!GEMINI_KEY
  );

  console.log(
    "BM AI own model configured: " + !!BM_MODEL_URL
  );

  console.log(
    "Grok integration: removed"
  );

  console.log(
    "Automatic routing enabled."
  );
});
  
