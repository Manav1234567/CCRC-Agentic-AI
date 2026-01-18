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

    def _format_doc_meta(self, doc, index):
        """Return a small metadata string for a document (used in final output)."""
        title = doc.get("title", "Unknown Title")
        authors = doc.get("author_list") or doc.get("authors") or []
        publishers = doc.get("publisher_list") or doc.get("publishers") or []
        year = doc.get("year")
        full_date = doc.get("full_date")
        parts = [f"[{index}] {title}"]
        if authors:
            parts.append(f"AUTHORS: {', '.join(authors)}")
        if publishers:
            parts.append(f"PUBLISHERS: {', '.join(publishers)}")
        if full_date:
            parts.append(f"DATE: {full_date}")
        elif year:
            parts.append(f"YEAR: {year}")
        if doc.get("source_url"):
            parts.append(f"SOURCE_URL: {doc.get('source_url')}")
        return "\n".join(parts)

    def search_documents(self, query: str, top_k: int = 5) -> str:
        """
        Document-level retrieval ONLY (document_vector_index).
        Returns either:
        - A short machine-friendly list of docs (one line per doc) with a stable doc_id property, or
        - The exact string 'NONE_FOUND' if nothing reliable was matched.
        LLM usage pattern: call this first when the user mentions a title/author/year/publisher specifically.
        """
        try:
            driver = GraphDatabase.driver(
                self.valves.NEO4J_URI,
                auth=(self.valves.NEO4J_USER, self.valves.NEO4J_PASSWORD),
            )
        except Exception as e:
            return f"ERROR: Connection Failed: {e}"

        vec = self._get_embedding(query)
        if isinstance(vec, str):
            return vec  # error string

        try:
            with driver.session() as session:
                q = """
                CALL db.index.vector.queryNodes('document_vector_index', $k, $vec)
                YIELD node AS doc, score
                RETURN doc, score
                """
                rows = session.run(q, k=top_k, vec=vec).data()

                if not rows:
                    return "NONE_FOUND"

                # If top score is extremely low consider NONE_FOUND -- but keep simple: return rows
                out_lines = []
                for r in rows:
                    doc = r["doc"]
                    score = r.get("score", 0)
                    # Ensure a stable id for later scoping; prefer doc.id/doc_id/title fallback
                    stable_id = doc.get("id") or doc.get("doc_id") or None
                    # Compose one-line record: STABLE_ID || SCORE || TITLE || AUTHORS || YEAR
                    title = doc.get("title", "").replace("\n", " ").strip()
                    authors = doc.get("author_list") or doc.get("authors") or []
                    year = doc.get("year") or ""
                    out_lines.append(
                        json.dumps(
                            {
                                "stable_id": stable_id,
                                "score": score,
                                "title": title,
                                "authors": authors,
                                "year": year,
                            },
                            ensure_ascii=False,
                        )
                    )
                return "\n".join(out_lines)

        except Exception as e:
            return f"ERROR: Doc search failed: {e}"
        finally:
            try:
                driver.close()
            except:
                pass

    def get_chunks_for_docs(
        self,
        doc_ids: list,
        query: str = None,
        max_chunks: int = 5,
        max_chars: int = 1200,
    ) -> str:
        """
        Given a list of document stable_ids (strings), retrieve up to max_chunks from those documents.
        If a doc_id is None or no doc-level property exists, we will attempt matching by title if query provides one.
        Output format (final): a single text block that the LLM can append to its answer.
        Format:
        ---
        [Doc metadata blocks]
        1a) <chunk text truncated>
        1b) <chunk text truncated>
        2a) ...
        Notes: each chunk is truncated to max_chars to respect token budget.
        """
        if not doc_ids:
            return "NO_DOC_IDS_PROVIDED"

        try:
            driver = GraphDatabase.driver(
                self.valves.NEO4J_URI,
                auth=(self.valves.NEO4J_USER, self.valves.NEO4J_PASSWORD),
            )
        except Exception as e:
            return f"ERROR: Connection Failed: {e}"

        # sanitise doc_ids (list)
        doc_ids = [d for d in doc_ids if d]

        try:
            with driver.session() as session:
                # Build a simple Cypher that matches documents by id/doc_id or title in provided list
                # We will collect chunks from those docs, limit total to max_chunks (across all docs).
                # Prefer an index backed lookup; this query uses a MATCH rather than vector.queryNodes because docs are already known.
                q = """
                UNWIND $doc_ids AS did
                MATCH (doc:Document)
                WHERE doc.id = did OR doc.doc_id = did OR doc.title = did
                WITH doc
                MATCH (doc)-[:HAS_CHUNK]->(chunk)
                RETURN doc, chunk
                LIMIT $limit
                """
                rows = session.run(q, doc_ids=doc_ids, limit=max_chunks).data()

                if not rows:
                    return "NO_CHUNKS_FOUND"

                # group by doc
                docs = {}
                for r in rows:
                    doc = r["doc"]
                    chunk = r["chunk"]
                    doc_key = (
                        doc.get("id")
                        or doc.get("doc_id")
                        or doc.get("title")
                        or str(abs(hash(doc.get("title", ""))))
                    )
                    if doc_key not in docs:
                        docs[doc_key] = {"doc": doc, "chunks": []}
                    docs[doc_key]["chunks"].append(chunk)

                # produce final compact output
                out = []
                out.append("---")
                # Document metadata blocks (numbered)
                for d_index, (doc_key, payload) in enumerate(docs.items(), start=1):
                    out.append(self._format_doc_meta(payload["doc"], d_index))
                    # list up to max_chunks_per_doc but overall we'll cap below
                # Now produce chunk numbering and texts; iterate docs in same order
                chunk_count = 0
                for d_index, (doc_key, payload) in enumerate(docs.items(), start=1):
                    for c_i, ch in enumerate(payload["chunks"]):
                        if chunk_count >= max_chunks:
                            break
                        chunk_count += 1
                        letter = chr(ord("a") + (c_i % 26))
                        label = f"{d_index}{letter}"
                        text = (ch.get("text") or ch.get("content") or "").strip()
                        # Truncate to max_chars (not tokens) safely
                        if len(text) > max_chars:
                            text = text[: max_chars - 3].rsplit(" ", 1)[0] + "..."
                        out.append(f"{label}) {text}")
                    if chunk_count >= max_chunks:
                        break

                return "\n".join(out)

        except Exception as e:
            return f"ERROR: Chunk retrieval failed: {e}"
        finally:
            try:
                driver.close()
            except:
                pass

    def search_climate_chunks(
        self, query: str, max_chunks: int = 5, max_chars: int = 1200
    ) -> str:
        """
        Direct chunk-level nearest neighbor retrieval (no doc pass).
        Returns final block formatted like get_chunks_for_docs.
        Use this when the user query is generic and doesn't request a specific title/author/publisher.
        """
        try:
            driver = GraphDatabase.driver(
                self.valves.NEO4J_URI,
                auth=(self.valves.NEO4J_USER, self.valves.NEO4J_PASSWORD),
            )
        except Exception as e:
            return f"ERROR: Connection Failed: {e}"

        vec = self._get_embedding(query)
        if isinstance(vec, str):
            return vec

        try:
            with driver.session() as session:
                q = """
                CALL db.index.vector.queryNodes('chunk_vector_index', $k, $vec)
                YIELD node AS chunk, score
                MATCH (doc:Document)-[:HAS_CHUNK]->(chunk)
                RETURN doc, chunk, score
                LIMIT $limit
                """
                rows = session.run(q, k=max_chunks, vec=vec, limit=max_chunks).data()
                if not rows:
                    return "NO_CHUNKS_FOUND"

                # group by doc
                docs = {}
                for r in rows:
                    doc = r["doc"]
                    chunk = r["chunk"]
                    doc_key = (
                        doc.get("id")
                        or doc.get("doc_id")
                        or doc.get("title")
                        or str(abs(hash(doc.get("title", ""))))
                    )
                    if doc_key not in docs:
                        docs[doc_key] = {"doc": doc, "chunks": []}
                    docs[doc_key]["chunks"].append(chunk)

                # format the same as get_chunks_for_docs
                out = []
                out.append("---")
                for d_index, (doc_key, payload) in enumerate(docs.items(), start=1):
                    out.append(self._format_doc_meta(payload["doc"], d_index))

                chunk_count = 0
                for d_index, (doc_key, payload) in enumerate(docs.items(), start=1):
                    for c_i, ch in enumerate(payload["chunks"]):
                        if chunk_count >= max_chunks:
                            break
                        chunk_count += 1
                        letter = chr(ord("a") + (c_i % 26))
                        label = f"{d_index}{letter}"
                        text = (ch.get("text") or ch.get("content") or "").strip()
                        if len(text) > max_chars:
                            text = text[: max_chars - 3].rsplit(" ", 1)[0] + "..."
                        out.append(f"{label}) {text}")
                    if chunk_count >= max_chunks:
                        break

                return "\n".join(out)

        except Exception as e:
            return f"ERROR: Chunk vector search failed: {e}"
        finally:
            try:
                driver.close()
            except:
                pass
