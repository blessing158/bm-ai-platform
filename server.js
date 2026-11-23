
import http from "node:http";
import { Agent, setGlobalDispatcher } from "undici";

const PORT = process.env.PORT || 3000;

const GEMINI_KEY = process.env.GEMINI_API_KEY;
const XAI_KEY = process.env.XAI_API_KEY;

const GEMINI_MODEL =
  process.env.GEMINI_MODEL || "gemini-3.5-flash-lite";

const GROK_MODEL =
  process.env.XAI_MODEL || "grok-4.7";

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
- You use AI models from Google Gemini and xAI Grok.
- Never claim OpenAI created BM AI.
- Never claim BM.DEVELOPERS trained the underlying models.

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

// Automatic provider selection.
function selectProvider(message, imageData) {
  if (imageData) {
    return "gemini";
  }

  const text = message.toLowerCase();

  // Questions that benefit from current information.
  const researchPatterns = [
    /\blatest\b/,
    /\btoday\b/,
    /\bcurrent\b/,
    /\brecent\b/,
    /\bnews\b/,
    /\bsearch\b/,
    /\blook up\b/,
    /\bfind information\b/,
    /\bsource\b/,
    /\bwebsite\b/,
    /\blink\b/,
    /\bprice\b/,
    /\bweather\b/,
    /\bwho is\b/,
    /\bwho was\b/,
    /\bmusician\b/,
    /\bsinger\b/,
    /\bartist\b/,
    /\bsong\b/,
    /\bmusic\b/,
    /\balbum\b/,
    /\brelease date\b/,
    /\bbiography\b/,
    /\bdiscography\b/,
    /\bwhere can i\b/,
    /\bwhat happened\b/
  ];

  for (const pattern of researchPatterns) {
    if (pattern.test(text)) {
      return "grok";
    }
  }

  // Name-like questions often need research.
  const words = text.trim().split(/\s+/);

  if (
    words.length >= 2 &&
    words.length <= 4 &&
    text.length <= 65 &&
    !/\b(the|and|but|because|please|thanks|explain)\b/.test(text)
  ) {
    return "grok";
  }

  // General knowledge, coding and reasoning default to Gemini.
  return "gemini";
}

function getHistory(history) {
  return history.map(item => ({
    role: item.role,
    content: item.content
  }));
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

function extractGrokText(data) {
  if (typeof data.output_text === "string") {
    return data.output_text.trim();
  }

  const output = [];

  for (const item of data.output || []) {
    if (item.type === "message") {
      for (const part of item.content || []) {
        if (
          (part.type === "output_text" ||
           part.type === "text") &&
          typeof part.text === "string"
        ) {
          output.push(part.text);
        }
      }
    }
  }

  return output.join("\n").trim();
}

function extractGrokSources(data) {
  const sources = [];

  for (const item of data.output || []) {
    for (const part of item.content || []) {
      for (const annotation of part.annotations || []) {
        const citation = annotation.url_citation;

        if (citation?.url) {
          sources.push({
            title: citation.title || citation.url,
            url: citation.url
          });
        }
      }
    }
  }

  return sources;
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

    throw new Error(
      data.error?.message || "Gemini request failed."
    );
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

async function callGrok(message, history, imageData) {
  if (!XAI_KEY) {
    throw new Error("Grok API key is not configured.");
  }

  const messages = [
    {
      role: "system",
      content: SYSTEM_PROMPT
    },
    ...getHistory(history)
  ];

  if (imageData) {
    if (!isValidImage(imageData)) {
      throw new Error("Unsupported image format.");
    }

    const match = imageData.match(
      /^data:(image\/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=]+)$/
    );

    if (match[1] === "image/webp") {
      throw new Error(
        "Grok image requests currently require JPG or PNG."
      );
    }

    messages.push({
      role: "user",
      content: [
        {
          type: "text",
          text: message
        },
        {
          type: "image_url",
          image_url: {
            url: imageData,
            detail: "high"
          }
        }
      ]
    });

    const response = await fetch(
      "https://api.x.ai/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": "Bearer " + XAI_KEY
        },
        body: JSON.stringify({
          model: GROK_MODEL,
          messages: messages,
          max_tokens: 2048
        }),
        signal: AbortSignal.timeout(80000)
      }
    );

    const data = await response.json();

    if (!response.ok) {
      console.error(
        "Grok image error:",
        data.error?.message || response.status
      );

      throw new Error(
        data.error?.message || "Grok image request failed."
      );
    }

    const answer = data.choices?.[0]?.message?.content;

    if (typeof answer !== "string" || !answer.trim()) {
      throw new Error("Grok returned an empty response.");
    }

    return answer.trim();
  }

  const input = [
    {
      role: "system",
      content: SYSTEM_PROMPT
    },
    ...getHistory(history),
    {
      role: "user",
      content: message
    }
  ];

  const requestBody = {
    model: GROK_MODEL,
    input: input,
    max_output_tokens: 2048
  };

  if (WEB_SEARCH_ENABLED) {
    requestBody.tools = [
      { type: "web_search" }
    ];
  }

  const response = await fetch(
    "https://api.x.ai/v1/responses",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + XAI_KEY
      },
      body: JSON.stringify(requestBody),
      signal: AbortSignal.timeout(80000)
    }
  );

  const data = await response.json();

  if (!response.ok) {
    console.error(
      "Grok API error:",
      data.error?.message || response.status
    );

    throw new Error(
      data.error?.message || "Grok request failed."
    );
  }

  const answer = extractGrokText(data);

  if (!answer) {
    throw new Error("Grok returned an empty response.");
  }

  return appendSources(
    answer,
    extractGrokSources(data)
  );
}

async function generateAnswer(provider, message, history, imageData) {
  if (provider === "grok") {
    return await callGrok(message, history, imageData);
  }

  return await callGemini(message, history, imageData);
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
        gemini: !!GEMINI_KEY,
        grok: !!XAI_KEY
      },
      automaticRouting: true,
      webSearch: WEB_SEARCH_ENABLED,
      vision: true,
      conversationMemory: true,
      version: "2.0.0"
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

    // Choose provider automatically.
    const selectedProvider = selectProvider(
      message.trim(),
      imageData
    );

    // Try the selected provider first.
    const otherProvider =
      selectedProvider === "gemini" ? "grok" : "gemini";

    let answer;
    let usedProvider = selectedProvider;
    let firstError = null;

    try {
      answer = await generateAnswer(
        selectedProvider,
        message.trim(),
        conversationMessages,
        imageData
      );
    } catch (error) {
      firstError = error;

      console.error(
        selectedProvider + " failed:",
        error.message
      );

      // Automatically try the other configured provider.
      const otherKey =
        otherProvider === "gemini" ? GEMINI_KEY : XAI_KEY;

      if (!otherKey) {
        throw error;
      }

      usedProvider = otherProvider;

      answer = await generateAnswer(
        otherProvider,
        message.trim(),
        conversationMessages,
        imageData
      );
    }

    return send(res, 200, {
      answer: answer,
      provider: usedProvider,
      automaticRouting: true,
      fallbackUsed: usedProvider !== selectedProvider
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
    "BM AI v2.0.0 running on port " + PORT
  );

  console.log(
    "Gemini configured: " + !!GEMINI_KEY
  );

  console.log(
    "Grok configured: " + !!XAI_KEY
  );

  console.log(
    "Automatic routing enabled."
  );
});
  
