import os
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI
# 1. IMPORT YOUR NEW CLASS
from robust_embedding import RobustLMStudioEmbeddings 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class ClimateRAG:
    def __init__(self, db_path=None):
        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://host.docker.internal:1234/v1")
        
        milvus_uri = os.getenv("MILVUS_URI")
        if not milvus_uri and db_path:
            milvus_uri = db_path
        elif not milvus_uri:
            milvus_uri = "./climate_news.db"

        print(f"🔌 Connecting to Milvus at: {milvus_uri}")
        
        # 2. USE THE ROBUST CLASS
        # We use the exact model ID you confirmed via cURL
        self.embeddings = RobustLMStudioEmbeddings(
            base_url=self.llm_base_url,
            model="text-embedding-qwen3-embedding-4b" 
        )
        
        # 3. LLM Setup (Chat model)
        self.llm = ChatOpenAI(
            base_url=self.llm_base_url,
            api_key="lm-studio",
            model="mistralai/ministral-3-14b-reasoning",
            temperature=0.1
        )
        
        # The rest remains exactly the same...
        self.vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": milvus_uri},
            collection_name="climate_articles",
            text_field="text",         # Matches your 'entry' dict in db_manager
            vector_field="vector",     # Matches your schema
            primary_field="id",        # 🔴 This must match your schema.add_field name
            auto_id=True
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        template = """You are a Climate Expert. Use the context below to answer.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:"""
        prompt = PromptTemplate.from_template(template)
        
        self.chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def ask(self, query):
        return self.chain.invoke(query)