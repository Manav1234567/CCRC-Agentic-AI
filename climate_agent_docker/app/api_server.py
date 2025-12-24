import json
import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.rag_engine import ClimateRAG

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = ClimateRAG()

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "climate-agent", "object": "model", "created": int(time.time()), "owned_by": "custom"}]
    }

@app.post("/v1/chat/completions")
async def chat_proxy(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    user_query = messages[-1]["content"] if messages else ""

    def generate_response():
        # Get the full answer from your agent
        full_answer = agent.ask(user_query)
        
        # Split into small chunks to simulate streaming for the UI
        # (This prevents the 'waiting forever' hang)
        words = full_answer.split(" ")
        for i, word in enumerate(words):
            chunk = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "climate-agent",
                "choices": [{
                    "index": 0,
                    "delta": {"content": word + (" " if i < len(words) - 1 else "")},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # Final stop signal
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")