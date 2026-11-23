
import json
import os
import re

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from bm_model import BM_AI_1M

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cpu")

app = FastAPI(title="BM AI Model API")

model = None
vocab = None
reverse_vocab = None


def encode(text):
    words = re.findall(r"\w+|[^\w\s]", text.lower())
    return [vocab.get(word, vocab["<UNK>"]) for word in words]


def decode(token_ids):
    return " ".join(
        reverse_vocab.get(int(token), "<UNK>")
        for token in token_ids
    )


@app.on_event("startup")
def load_model():
    global model, vocab, reverse_vocab

    with open(
        os.path.join(BASE_DIR, "bm_ai_tokenizer.json"),
        "r",
        encoding="utf-8"
    ) as file:
        vocab = json.load(file)

    checkpoint = torch.load(
        os.path.join(BASE_DIR, "bm_ai_1m.pt"),
        map_location=DEVICE,
        weights_only=True
    )

    if checkpoint["vocab_size"] != len(vocab):
        raise RuntimeError("Tokenizer and model do not match")

    model = BM_AI_1M(len(vocab)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    reverse_vocab = {
        int(value): key
        for key, value in vocab.items()
    }


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "bm-ai-1m"
    messages: list[ChatMessage]
    max_tokens: int = 20
    temperature: float = 1.0


@app.get("/")
def health():
    return {
        "status": "healthy",
        "service": "BM AI Model API",
        "model_loaded": model is not None,
        "model": "BM AI-1M"
    }


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    if not request.messages:
        raise HTTPException(
            status_code=400,
            detail="No messages provided"
        )

    prompt = " ".join(
        message.content
        for message in request.messages[-6:]
    )

    token_ids = [vocab["<BOS>"]] + encode(prompt)
    token_ids = token_ids[-255:]

    generated = []
    current = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=DEVICE
    )

    temperature = max(
        0.2,
        min(request.temperature, 2.0)
    )

    with torch.inference_mode():
        for _ in range(max(1, min(request.max_tokens, 40))):
            logits = model(current[:, -256:])[:, -1, :]
            probabilities = torch.softmax(
                logits / temperature,
                dim=-1
            )

            next_id = torch.multinomial(
                probabilities,
                num_samples=1
            )

            token = int(next_id.item())

            if token == vocab["<EOS>"]:
                break

            generated.append(token)
            current = torch.cat(
                [current, next_id],
                dim=1
            )

    answer = decode(generated)

    return {
        "model": "bm-ai-1m",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": answer
            },
            "finish_reason": "length"
        }],
        "note": "Experimental untrained model."
}
