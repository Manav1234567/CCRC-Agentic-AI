from pydantic import BaseModel, Field
import requests
from neo4j import GraphDatabase
import os


class Tools:
    class Valves(BaseModel):
        # UI Settings - Change these in Open WebUI > Tool Settings
        NEO4J_URI: str = Field(
            default="neo4j://127.0.0.1:7687",
            description="Neo4j URI (Use host.docker.internal if in Docker)",
        )
        NEO4J_USER: str = Field(default="neo4j", description="Database User")
        NEO4J_PASSWORD: str = Field(
            default="manav@1234", description="Database Password"
        )
        EMBED_URL: str = Field(
            default="http://127.0.0.1:1234/v1/embeddings",
            description="LM Studio Embedding Endpoint",
        )
        EMBED_MODEL: str = Field(
            default="text-embedding-qwen3-embedding-4b",
            description="Model ID in LM Studio",
        )

    def __init__(self):
        self.valves = self.Valves()

    def _get_embedding(self, text):
        """Internal helper to get vectors from LM Studio"""
        try:
            payload = {"model": self.valves.EMBED_MODEL, "input": text}
            # We use the valve URL
            response = requests.post(self.valves.EMBED_URL, json=payload, timeout=5)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            return f"Error generating embedding: {e}"

    def search_climate_papers(self, query: str) -> str:
        """
        Search the Neo4j Knowledge Graph for research papers, authors, and climate data.
        Use this tool whenever the user asks about droughts, climate change, specific authors (like Falster), or research findings.

        :param query: The specific search topic or question (e.g. "Australian drought causes" or "Falster 2024 findings")
        :return: A formatted string of citations and text chunks.
        """
        print(f"🔍 Graph Tool Triggered: {query}")

        # 1. Connect to DB
        try:
            driver = GraphDatabase.driver(
                self.valves.NEO4J_URI,
                auth=(self.valves.NEO4J_USER, self.valves.NEO4J_PASSWORD),
            )
        except Exception as e:
            return f"Connection Failed: {e}"

        # 2. Get Vector
        vector = self._get_embedding(query)
        if isinstance(vector, str):
            return vector  # Return error if string

        results = []
        try:
            with driver.session() as session:
                # --- A. VECTOR SEARCH ---
                # Note: Ensure index name 'chunk_vector_index' matches your DB setup
                vec_query = """
                CALL db.index.vector.queryNodes('chunk_vector_index', 5, $vec) 
                YIELD node AS chunk, score
                MATCH (doc:Document)-[:HAS_CHUNK]->(chunk)
                RETURN doc, chunk, score
                """
                vec_res = session.run(vec_query, vec=vector).data()

                # --- B. KEYWORD SEARCH (For Authors) ---
                kw_query = """
                MATCH (doc:Document)-[:HAS_CHUNK]->(chunk)
                WHERE any(a IN doc.author_list WHERE toLower(a) CONTAINS toLower($q)) 
                   OR toLower(doc.title) CONTAINS toLower($q)
                RETURN doc, chunk, 1.0 as score
                LIMIT 3
                """
                kw_res = session.run(kw_query, q=query).data()

                # --- C. MERGE RESULTS ---
                # Simple deduplication by chunk ID
                seen_ids = set()
                combined = []
                for r in vec_res + kw_res:
                    if r["chunk"]["id"] not in seen_ids:
                        seen_ids.add(r["chunk"]["id"])
                        combined.append(r)

                # Sort and Format
                combined = sorted(combined, key=lambda x: x["score"], reverse=True)[:6]

                for item in combined:
                    doc = item["doc"]
                    chunk = item["chunk"]

                    # Create Friendly Citation
                    first_author = doc.get("author_list", ["Unknown"])[0].split()[-1]
                    year = doc.get("year", 0)
                    cite_ref = f"{first_author} et al. ({year})"

                    results.append(
                        f"SOURCE: {doc.get('title')}\n"
                        f"CITATION_REF: {cite_ref}\n"
                        f"CONTENT: {chunk.get('text')}\n"
                        f"---"
                    )

        except Exception as e:
            return f"Graph Search Error: {e}"
        finally:
            driver.close()

        if not results:
            return "No relevant documents found in the database."

        return "\n".join(results)
