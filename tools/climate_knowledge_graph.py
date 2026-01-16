from pydantic import BaseModel, Field
import requests
from neo4j import GraphDatabase
import os
import json


class Tools:
    class Valves(BaseModel):
        # UI Settings
        NEO4J_URI: str = Field(
            default="neo4j://127.0.0.1:7687",
            description="Neo4j URI (Use host.docker.internal if in Docker)",
        )
        NEO4J_USER: str = Field(default="neo4j", description="Database User")
        NEO4J_PASSWORD: str = Field(
            default="manav@1234", description="Database Password"
        )
        EMBED_URL: str = Field(
            default="http://127.0.0.1:8001/v1/embeddings",
            description="Gadi Endpoint",
        )
        EMBED_MODEL: str = Field(
            default="jinaai/jina-embeddings-v3", description="Model ID in Gadi"
        )

    def __init__(self):
        self.valves = self.Valves()

    def _get_embedding(self, text):
        """Internal helper to get vectors from LM Studio"""
        try:
            payload = {"model": self.valves.EMBED_MODEL, "input": text}
            response = requests.post(self.valves.EMBED_URL, json=payload, timeout=5)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except Exception as e:
            return f"Error generating embedding: {e}"

    def search_climate_papers(self, query: str) -> str:
        """
        Search the Neo4j Knowledge Graph for research papers, authors, and climate data.
        Returns a formatted text block with grouped sources and chunk-level IDs (e.g. 1a, 1b).
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
            return vector

        try:
            with driver.session() as session:
                # --- A. VECTOR SEARCH ---
                vec_query = """
                CALL db.index.vector.queryNodes('chunk_vector_index', 6, $vec) 
                YIELD node AS chunk, score
                MATCH (doc:Document)-[:HAS_CHUNK]->(chunk)
                RETURN doc, chunk, score
                """
                vec_res = session.run(vec_query, vec=vector).data()

                # --- B. KEYWORD SEARCH ---
                kw_query = """
                MATCH (doc:Document)-[:HAS_CHUNK]->(chunk)
                WHERE 
                    any(a IN doc.author_list WHERE toLower(a) CONTAINS toLower($q)) OR 
                    any(p IN doc.publisher_list WHERE toLower(p) CONTAINS toLower($q)) OR
                    toLower(doc.title) CONTAINS toLower($q)
                RETURN doc, chunk, 1.0 as score
                LIMIT 6
                """
                kw_res = session.run(kw_query, q=query).data()

                # --- C. DEDUPLICATE & SORT ---
                seen_chunk_ids = set()
                combined = []
                for r in vec_res + kw_res:
                    # chunk may be a dict-like object
                    chunk = r.get("chunk", {})
                    # ensure stable chunk id (fallback to hash of text)
                    chunk_id = chunk.get("id") or chunk.get("chunk_id") or str(abs(hash(chunk.get("text", "") )) )
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        combined.append(r)

                # Sort by score and limit total chunks
                combined = sorted(combined, key=lambda x: x.get("score", 0), reverse=True)[:12]

                if not combined:
                    return "No relevant documents found."

                # --- D. GROUP BY DOCUMENT ---
                docs = {}  # key -> {"doc": doc_obj, "chunks": [ {chunk, score, chunk_id} ] }
                order_of_docs = []  # preserve order encountered
                for r in combined:
                    doc = r["doc"]
                    chunk = r["chunk"]
                    score = r.get("score", 0)

                    # stable doc key - prefer doc.id if present, else title hash
                    doc_key = doc.get("id") or doc.get("doc_id") or doc.get("title", "")[:80] + f"_{abs(hash(doc.get('title','')))}"
                    if doc_key not in docs:
                        docs[doc_key] = {"doc": doc, "chunks": []}
                        order_of_docs.append(doc_key)

                    chunk_id = chunk.get("id") or chunk.get("chunk_id") or str(abs(hash(chunk.get("text", ""))))
                    docs[doc_key]["chunks"].append({"chunk": chunk, "score": score, "chunk_id": chunk_id})

                # --- E. BUILD FORMATTED OUTPUT (INLINE SUMMARY + EXPANDABLE SOURCES) ---
                out_lines = []
                out_lines.append(f"Query: {query}")
                out_lines.append("")
                out_lines.append("Top matched documents and chunk previews:")
                out_lines.append("")

                # top summary list with doc index and chunk-letter previews
                for d_idx, doc_key in enumerate(order_of_docs, start=1):
                    doc = docs[doc_key]["doc"]
                    chunks = docs[doc_key]["chunks"]

                    # Citation ref (author/publisher/title)
                    title = doc.get("title", "Unknown Title")
                    authors = doc.get("author_list", []) or []
                    publishers = doc.get("publisher_list", []) or []
                    year = doc.get("year")
                    cite_year = f" ({year})" if year else ""
                    if authors:
                        primary = authors[0].split()[-1]
                        cite_name = f"{primary} et al." if len(authors) > 1 else primary
                        citation_ref = f"{cite_name}{cite_year}"
                    elif publishers:
                        citation_ref = f"{publishers[0]}{cite_year}"
                    else:
                        citation_ref = f"{title[:30]}...{cite_year}"

                    out_lines.append(f"[{d_idx}] {title} — {citation_ref}")

                    # list chunk previews
                    for c_i, ch in enumerate(chunks):
                        letter = chr(ord("a") + c_i)
                        text = (ch["chunk"].get("text") or "").replace("\n", " ").strip()
                        preview = (text[:120] + "...") if len(text) > 120 else text
                        out_lines.append(f"  - [{d_idx}{letter}] {preview}")
                    out_lines.append("")

                # --- F. DETAILED SOURCES (click-to-expand using HTML <details>) ---
                out_lines.append("\n-----\nSOURCES (Exact Text):\n")
                for d_idx, doc_key in enumerate(order_of_docs, start=1):
                    doc = docs[doc_key]["doc"]
                    chunks = docs[doc_key]["chunks"]

                    title = doc.get("title", "Unknown Title")
                    authors = doc.get("author_list", []) or []
                    publishers = doc.get("publisher_list", []) or []
                    year = doc.get("year")
                    full_date = doc.get("full_date")
                    cite_year = f" ({year})" if year else ""
                    if authors:
                        primary = authors[0].split()[-1]
                        cite_name = f"{primary} et al." if len(authors) > 1 else primary
                        citation_ref = f"{cite_name}{cite_year}"
                    elif publishers:
                        citation_ref = f"{publishers[0]}{cite_year}"
                    else:
                        citation_ref = f"{title[:30]}...{cite_year}"

                    # Document header
                    out_lines.append(f"### [{d_idx}] {title}")
                    out_lines.append(f"CITATION_REF: {citation_ref}")
                    if full_date:
                        out_lines.append(f"DATE: {full_date}")
                    elif year:
                        out_lines.append(f"YEAR: {year}")

                    if authors:
                        out_lines.append(f"AUTHORS: {', '.join(authors)}")
                    if publishers:
                        out_lines.append(f"PUBLISHERS: {', '.join(publishers)}")
                    out_lines.append("")

                    # chunk details using <details> tags so UI can expand
                    for c_i, ch in enumerate(chunks):
                        letter = chr(ord("a") + c_i)
                        chunk_text = (ch["chunk"].get("text") or "").strip()
                        chunk_preview = (chunk_text.replace("\n", " ")[:140] + "...") if len(chunk_text) > 140 else chunk_text
                        # Use an HTML details block (many markdown renderers support this)
                        out_lines.append(f"\n\n-----\n[{d_idx}{letter}]\n-----\n{chunk_text}\n\n")

                    out_lines.append("\n")  # space between docs

                return "\n".join(out_lines)

        except Exception as e:
            return f"Graph Search Error: {e}"
        finally:
            try:
                driver.close()
            except:
                pass
