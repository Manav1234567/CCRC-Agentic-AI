import os
import time
import uuid
import uvicorn
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from typing import List, Optional

from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1) CONFIG
LLM_HOST = "http://127.0.0.1:1234"  # LM Studio base (no /v1)
EMBEDDING_HOST = "http://127.0.0.1:1234"

PROXY_BACKEND = f"{LLM_HOST}/v1"    # requests not handled by us will be forwarded here

EMBEDDING_MODEL = "text-embedding-qwen3-embedding-4b"
# LLM_MODEL = "mistralai/ministral-3-14b-reasoning"
LLM_MODEL = "nvidia/nemotron-3-nano"

# 2) FASTAPI APP
app = FastAPI(title="Climate News RAG API (OpenAI-compatible)")

# 3) Pydantic models for OpenAI-style request
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

# 4) Initialize RAG components
print("⏳ Initializing RAG components...")

embeddings = OpenAIEmbeddings(
    base_url=f"{EMBEDDING_HOST}/v1",
    api_key="lm-studio",
    model=EMBEDDING_MODEL,
    check_embedding_ctx_length=False
)

llm = ChatOpenAI(
    base_url=f"{LLM_HOST}/v1",
    api_key="lm-studio",
    model=LLM_MODEL,
    temperature=0.1
)

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "./climate_news.db"},
    collection_name="climate_articles",
    text_field="text",
    auto_id=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Template: always include context (even if empty). Let the model decide whether to use it.
template = """You are a specialized Climate News Assistant.
You are given retrieved context (which may be empty). Use the context when it helps to answer.
If the answer is not present in the context, first explicitly state:
"I could not find this information in the local climate news database."
Then you may answer from your general knowledge.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    if not docs:
        return ""
    return "\n\n".join(doc.page_content for doc in docs)

# --- IMPORTANT: build the runnable RAG chain so retrieval happens correctly ---
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ RAG chain initialized.")

# Helper: run the RAG-first pipeline (returns string) by invoking the chain
def answer_with_rag(question: str) -> str:
    # Use the runnable chain which performs retrieval -> formatting -> prompt -> llm
    return rag_chain.invoke(question)


# Existing convenience endpoint (backwards compat)
@app.post("/chat")
def chat_endpoint(query: QueryRequest):
    try:
        print(f"💬 /chat received: {query.question}")
        answer = answer_with_rag(query.question)
        return {"answer": answer}
    except Exception as e:
        print("❌ /chat error:", e)
        raise HTTPException(status_code=500, detail=str(e))


# OpenAI-compatible endpoint — this is what Open WebUI will call as a model
@app.post("/v1/chat/completions")
def openai_chat_completions(req: ChatCompletionRequest):
    try:
        user_msgs = [m.content for m in req.messages if m.role == "user"]
        if not user_msgs:
            raise HTTPException(status_code=400, detail="No user message provided")
        user_question = user_msgs[-1]

        print(f"💬 /v1/chat/completions called. question: {user_question}")

        answer_text = answer_with_rag(user_question)

        out = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or "climate-news-rag",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": answer_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        return out

    except Exception as e:
        print("❌ /v1/chat/completions error:", e)
        raise HTTPException(status_code=500, detail=str(e))


# Minimal /v1/models so Open WebUI sees at least one model
@app.get("/v1/models")
def list_models():
    return {
        "data": [
            {
                "id": "climate-news-rag",
                "object": "model",
                "owned_by": "local"
            }
        ]
    }


# Proxy: forward any other /v1/* request to LM Studio so nothing else breaks.
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy_to_backend(request: Request, path: str):
    backend_url = f"{PROXY_BACKEND}/{path}"
    print(f"Proxying {request.method} /v1/{path} -> {backend_url}")

    async with httpx.AsyncClient() as client:
        headers = dict(request.headers)
        headers.pop("host", None)
        try:
            resp = await client.request(
                method=request.method,
                url=backend_url,
                headers=headers,
                content=await request.body(),
                timeout=60.0
            )
        except Exception as e:
            print("Proxy request error:", e)
            raise HTTPException(status_code=502, detail=str(e))

    return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
