import json

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_milvus import Milvus

import constants



# SETUP MODELS
embeddings = OpenAIEmbeddings(
    base_url = f"{constants.EMBEDDING_ENDPOINT}/v1",
    api_key = constants.EMBEDDING_API_KEY,
    model = constants.EMBEDDING_MODEL,
    check_embedding_ctx_length = False
)

llm = ChatOpenAI(
    base_url = f"{constants.LLM_ENDPOINT}/v1",
    api_key = constants.LLM_API_KEY,
    model = constants.LLM_MODEL,
    temperature = constants.LLM_TEMPERATURE
)

# ------------------------------------------------------------------
# 1. LOAD COLLECTION REGISTRY (Step 1)
# ------------------------------------------------------------------
with open("db_description.json", "r") as f:
    COLLECTION_REGISTRY = json.load(f)

def format_registry():
    """Convert collection registry JSON into readable text for the LLM."""
    blocks = []
    for col in COLLECTION_REGISTRY["collections"]:
        blocks.append(
            f"""
Collection: {col['name']}
Type: {col['type']}
Domain: {col['domain']}
Granularity: {col['granularity']}
Strengths: {", ".join(col['strengths'])}
Weaknesses: {", ".join(col['weaknesses'])}
"""
        )
    return "\n".join(blocks)


# ------------------------------------------------------------------
# 2. MULTI-COLLECTION RETRIEVER (Step 2)
# ------------------------------------------------------------------
# One Milvus vector store per collection
research_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "./climate_news.db"},
    collection_name="research_papers",
    text_field="text",
)

news_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "./climate_news.db"},
    collection_name="climate_articles",
    text_field="text",
)

research_retriever = research_store.as_retriever(search_kwargs={"k": 3})
news_retriever = news_store.as_retriever(search_kwargs={"k": 3})


def retrieve_docs(query: str):
    """
    Retrieve from BOTH collections.
    Later we’ll make this LLM-controlled.
    """
    docs = []
    docs.extend(research_retriever.invoke(query))
    docs.extend(news_retriever.invoke(query))
    return docs


def format_docs(docs):
    blocks = []
    for d in docs:
        source = d.metadata.get("source", "unknown")
        blocks.append(f"[SOURCE: {source}]\n{d.page_content}")
    return "\n\n".join(blocks)


# Wrap functions as Runnables ✅
retrieve_docs_runnable = RunnableLambda(retrieve_docs)
format_docs_runnable = RunnableLambda(format_docs)
registry_runnable = RunnableLambda(lambda _: format_registry())

# ------------------------------------------------------------------
# COLLECTION ROUTER PROMPT
# ------------------------------------------------------------------
router_template = """You are a routing assistant.

You are given a list of available data collections.
Your job is to decide which collections (if any) should be queried
to answer the user question.

Rules:
- You can select multiple collections.
- Only select collections that are clearly relevant.
- If none are relevant, return "none".
- Return ONLY valid JSON.
- Do NOT explain your reasoning.

Available collections:
{registry}

User question:
{question}

Respond in this exact JSON format:
{{
  "collections": ["collection_name_1", "collection_name_2"]
}}

OR

{{
  "collections": "none"
}}
"""

router_prompt = PromptTemplate.from_template(router_template)

router_chain = (
    {
        "registry": registry_runnable,
        "question": RunnablePassthrough()
    }
    | router_prompt
    | llm
    | StrOutputParser()
)


def select_collections(query: str):
    """Stage 1: Decide which collections to use."""
    raw = router_chain.invoke(query)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Hard fail-safe
        return {"collections": "none"}

COLLECTION_VECTORSTORES = {
    "research_papers": research_store,
    "climate_articles": news_store,
}

def retrieve_from_collections(query: str, collections):
    """Retrieve documents from selected collections using direct vector search."""
    docs = []

    # Embed the query once using the real embedding model
    query_vector = embeddings.embed_query(query)

    for name in collections:
        store = COLLECTION_VECTORSTORES.get(name)
        if store:
            # Use similarity_search_by_vector instead of similarity_search
            # to avoid re-embedding with FakeEmbeddings
            docs.extend(store.similarity_search_by_vector(embedding=query_vector, k=3))

    return docs


final_template = """You are a careful climate expert.

Use ONLY the provided context to answer the question.
If the context does not contain the answer, say so clearly.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


final_prompt = PromptTemplate.from_template(final_template)

final_chain = (
    {
        "context": RunnablePassthrough(),
        "question": RunnablePassthrough()
    }
    | final_prompt
    | llm
    | StrOutputParser()
)

def answer_query(query: str):
    """
    Full pipeline:
    1. Decide collections
    2. Retrieve context if needed
    3. Answer with or without RAG
    """

    routing = select_collections(query)
    collections = routing.get("collections")

    # --------------------------------------------------
    # CASE 1: No retrieval needed
    # --------------------------------------------------
    if collections == "none":
        return llm.invoke(
            f"You are a climate expert.\n"
            f"Answer from general knowledge and say so if needed\n\n"
            f"Question: {query}"
        )

    if isinstance(collections, str):
        collections = [collections]

    # --------------------------------------------------
    # CASE 2: Targeted RAG
    # --------------------------------------------------
    docs = retrieve_from_collections(query, collections)
    context = format_docs(docs)

    if not context.strip():
        return "The selected datasets do not contain relevant information."

    result = final_chain.invoke(
        {
            "context": context,
            "question": query
        }
    )

    return result
