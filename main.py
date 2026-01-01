import time
import uuid
import uvicorn
import httpx

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from typing import List, Optional

import constants

from rag import answer_query

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PROXY_BACKEND = f"{constants.LLM_ENDPOINT}/v1"        # Forward unknown routes here

APP_PORT = 8001

# ------------------------------------------------------------------
# FASTAPI APP
# ------------------------------------------------------------------
app = FastAPI(title="Climate RAG Gateway (OpenAI Compatible)")

# ------------------------------------------------------------------
# OPENAI-SHAPED MODELS
# ------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Message]
    temperature: Optional[float] = 0.1
    stream: Optional[bool] = False

class QueryRequest(BaseModel):
    question: str

# ------------------------------------------------------------------
# SIMPLE /chat (for direct testing)
# ------------------------------------------------------------------
@app.post("/chat")
def chat_endpoint(query: QueryRequest):
    try:
        print(f"💬 /chat → {query.question}")
        return {"answer": answer_query(query.question)}
    except Exception as e:
        print("❌ /chat error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------
# OPENAI-COMPATIBLE CHAT COMPLETIONS (Open WebUI entrypoint)
# ------------------------------------------------------------------
# main.py

@app.post("/v1/chat/completions")
def openai_chat_completions(req: ChatCompletionRequest):
    try:
        user_messages = [m.content for m in req.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message provided")

        user_question = user_messages[-1]
        
        result = answer_query(user_question)

        answer_text = result.content if hasattr(result, "content") else str(result)
        
        # Open WebUI looks for "reasoning_content" specifically
        reasoning_text = result.reasoning_content if hasattr(result, "reasoning_content") else None

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(result, "response_metadata"):
            token_info = result.response_metadata.get("token_usage", {})
            usage = {
                "prompt_tokens": token_info.get("prompt_tokens", 0),
                "completion_tokens": token_info.get("completion_tokens", 0),
                "total_tokens": token_info.get("total_tokens", 0)
            }

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or "local-climate-rag",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer_text,
                        "reasoning_content": reasoning_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": usage
        }

    except Exception as e:
        print("❌ /v1/chat/completions error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------------------------------------------------
# MINIMAL /v1/models (required by Open WebUI)
# ------------------------------------------------------------------
@app.get("/v1/models")
def list_models():
    return {
        "data": [
            {
                "id": "local-climate-rag",
                "object": "model",
                "owned_by": "local"
            }
        ]
    }

# ------------------------------------------------------------------
# PROXY ALL OTHER /v1/* REQUESTS TO LM STUDIO
# ------------------------------------------------------------------
@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
)
async def proxy_to_lmstudio(request: Request, path: str):
    backend_url = f"{PROXY_BACKEND}/{path}"
    print(f"🔁 Proxy {request.method} /v1/{path} → {backend_url}")

    async with httpx.AsyncClient() as client:
        headers = dict(request.headers)
        headers.pop("host", None)

        try:
            response = await client.request(
                method=request.method,
                url=backend_url,
                headers=headers,
                content=await request.body(),
                timeout=60.0
            )
        except Exception as e:
            print("❌ Proxy error:", e)
            raise HTTPException(status_code=502, detail=str(e))

    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers)
    )

# ------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(f"🚀 Climate RAG Gateway running on http://{constants.RAG_HOST}:{constants.RAG_PORT}")
    uvicorn.run(app, host=constants.RAG_HOST, port=constants.RAG_PORT)
