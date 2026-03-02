import random
import time
from fastapi import FastAPI

from app.models.api_models import ChatRequest, ChatResponse


app = FastAPI(title="Local Mock Chatbot", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    # Simulate LLM latency
    latency_ms = random.uniform(20, 120)
    time.sleep(latency_ms / 1000.0)

    msg = req.message.lower()

    if any(k in msg for k in ["refund", "cancel", "chargeback"]):
        intent = "refund_request"
        confidence = round(random.uniform(0.85, 0.98), 2)
        response = random.choice([
            "I can help with a refund. Please share your order ID.",
            "Sure — refunds are possible. What’s your order number?"
        ])

    elif any(k in msg for k in ["price", "cost", "how much"]):
        intent = "pricing_question"
        confidence = round(random.uniform(0.75, 0.95), 2)
        response = random.choice([
            "Pricing depends on the plan. Are you interested in Basic or Pro?",
            "Costs vary by plan—Basic or Pro?"
        ])

    elif any(k in msg for k in ["hi", "hello", "hey"]):
        intent = "greeting"
        confidence = round(random.uniform(0.90, 0.99), 2)
        response = random.choice([
            "Hi! How can I help you today?",
            "Hello! What can I do for you?"
        ])

    else:
        intent = "unknown"
        confidence = round(random.uniform(0.45, 0.75), 2)
        response = random.choice([
            "Could you clarify what you mean?",
            "I’m not sure I understood—can you rephrase?"
        ])

    return ChatResponse(
        response=response,
        intent=intent,
        confidence=float(confidence),
        latency_ms=float(latency_ms),
    )