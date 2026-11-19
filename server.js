import http from "node:http";

const PORT = process.env.PORT || 3000;

const server = http.createServer(async (req, res) => {
  res.setHeader("Content-Type", "application/json; charset=utf-8");

  if (req.method === "GET" && req.url === "/") {
    res.writeHead(200);
    return res.end(JSON.stringify({
      status: "healthy",
      service: "BM AI"
    }));
  }

  if (req.method === "GET" && new URL(req.url, "http://localhost").pathname === "/") {
    res.writeHead(404);
    return res.end(JSON.stringify({ error: "Not found" }));
  }

  const apiKey = process.env.GEMINI_API_KEY;

  if (!apiKey) {
    res.writeHead(500);
    return res.end(JSON.stringify({
      error: "Gemini API key is not configured"
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
      "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": apiKey
        },
        body: JSON.stringify({
          contents: [
            {
              parts: [{ text: message }]
            }
          ]
        })
      }
    );

    const data = await response.json();

    if (!response.ok) {
      res.writeHead(response.status);
      return res.end(JSON.stringify({
        error: data.error?.message || "Gemini request failed"
      }));
    }

    const answer = data.candidates?.[0]?.content?.parts
      ?.map(part => part.text || "")
      .join("") || "No response received.";

    res.writeHead(200);
    return res.end(JSON.stringify({ answer }));

  } catch (error) {
    console.error("BM AI error:", error);

    res.writeHead(500);
    return res.end(JSON.stringify({
      error: "Unable to process your request"
    }));
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`BM AI server running on port ${PORT}`);
});
