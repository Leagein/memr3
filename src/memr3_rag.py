import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
import tiktoken
from sentence_transformers import CrossEncoder

from src.memr3_base import BaseMemR3Manager, MemoryState


class MaskedRetriever:
    """RAG retriever that masks previously used chunks."""

    def __init__(
        self,
        client,
        embedding_model: str,
        candidate_k: int = 20,
        rerank_model: Optional[str] = None,
    ):
        self.client = client
        self.embedding_model = embedding_model
        self.candidate_k = candidate_k
        self.rerank_model = rerank_model
        self.cross_encoder: Optional[Any] = CrossEncoder(rerank_model, device="cuda") if rerank_model else None
        if self.cross_encoder:
            print(f"[Retriever] Using cross-encoder rerank model: {rerank_model}")

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def retrieve(
        self, query: str, chunks: List[str], embeddings: List[List[float]], mask: List[int], top_k: int
    ) -> List[tuple[int, str, float]]:
        if not chunks:
            return []

        query_embedding = self.embed(query)
        chunk_embeddings = np.array(embeddings)
        similarities = np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        mask_set = set(mask or [])
        ranked_indices = np.argsort(similarities)[::-1]
        candidates: List[tuple[int, str, float]] = []
        candidate_limit = min(len(ranked_indices), max(top_k, self.candidate_k))

        for idx in ranked_indices:
            if int(idx) in mask_set:
                continue
            candidates.append((int(idx), chunks[int(idx)], float(similarities[int(idx)])))
            if len(candidates) >= candidate_limit:
                break

        if not candidates:
            return candidates

        cross_encoder = self.cross_encoder
        if not cross_encoder:
            return candidates[:top_k]

        pairs = [[query, chunk] for _, chunk, _ in candidates]
        scores = cross_encoder.predict(pairs)
        reranked = sorted(
            zip(candidates, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [
            (idx, chunk, float(score))
            for (idx, chunk, _sim), score in reranked[:top_k]
        ]


class MemR3RAGManager(BaseMemR3Manager):
    """MemR3 memory agent backed by direct RAG retrieval."""

    def __init__(
        self,
        data_path: str = "dataset/locomo10_rag.json",
        chunk_size: int = 500,
        top_k: int = 1,
        max_iterations: int = 5,
        candidate_k: int = 20,
        rerank_model: Optional[str] = None,
    ):
        self.chunk_size = chunk_size
        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        if not os.getenv("MODEL"):
            raise ValueError("MODEL environment variable must be set.")
        if not self.embedding_model:
            raise ValueError("EMBEDDING_MODEL environment variable must be set.")

        cache_dir_env = os.getenv("MEMR3_RAG_CACHE_DIR")
        default_cache_dir = Path(__file__).resolve().parent.parent / "rag_cache"
        self.rag_cache_dir = Path(cache_dir_env).expanduser() if cache_dir_env else default_cache_dir
        self.rag_cache_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(data_path=data_path, top_k=top_k, max_iterations=max_iterations)

        try:
            self.encoding = tiktoken.encoding_for_model(self.embedding_model)
        except Exception:
            self.encoding = tiktoken.get_encoding(self.embedding_model)

        rerank_env_model = os.getenv("MEMR3_RERANK_MODEL")
        if rerank_model is None:
            rerank_model_name = rerank_env_model or "cross-encoder/ms-marco-MiniLM-L-12-v2"
        else:
            rerank_model_name = rerank_model
        if rerank_model_name == "":
            rerank_model_name = None
        self.retriever = MaskedRetriever(
            self.client,
            self.embedding_model,
            candidate_k=candidate_k,
            rerank_model=rerank_model_name,
        )

    def _has_retrieval_opportunity(self, state: MemoryState) -> bool:
        chunks = state.get("chunks", [])
        masked = state.get("masked_indices", [])
        return bool(chunks) and len(masked) < len(chunks)

    def _retrieve(self, state: MemoryState) -> None:
        query = state["question"]
        if state.get("retrieval_query"):
            query = f"{query}. {state['retrieval_query']}" # new query ablation
        chunks = state.get("chunks", [])
        embeddings = state.get("embeddings", [])
        start = time.time()
        retrieved = self.retriever.retrieve(query, chunks, embeddings, state.get("masked_indices", []), self.top_k)
        duration = time.time() - start
        # print(f"[Retrieval Query] {query}")

        if not retrieved:
            print("[Retrieval] No chunks retrieved.")
            state.setdefault("routing_trace", []).append("Retrieve node found no new chunks; reflecting.")
            return

        state.setdefault("raw_snippets", [])
        state.setdefault("masked_indices", [])
        state.setdefault("retrieval_history", [])
        for idx, chunk, score in retrieved:
            snippet = chunk.replace("\n", " ").strip()
            if len(snippet) > 200:
                snippet = f"{snippet[:200]}..."
            # print(f"[Retrieval Result] chunk_{idx} score={score:.4f} content={snippet}")
            state["raw_snippets"].append(chunk)
            state["masked_indices"].append(idx) # mask ablation
            history_entry = f"chunk_{idx} (score={score:.4f}) :: {snippet}"
            state["retrieval_history"].append(history_entry)
        state.setdefault("search_times", []).append(round(duration, 3))

    @staticmethod
    def _clean_chat_history(chat_history: List[dict]) -> str:
        return "\n".join(f"{entry['timestamp']} | {entry['speaker']}: {entry['text']}" for entry in chat_history)

    def _create_chunks(self, chat_history: List[dict]) -> Tuple[List[str], List[List[float]]]:
        chunks = self._chunk_chat_history(chat_history)
        embeddings = [self.retriever.embed(chunk) for chunk in tqdm(chunks, desc="Embedding chunks")]
        return chunks, embeddings

    def _chunk_chat_history(self, chat_history: List[dict]) -> List[str]:
        if self.chunk_size == -1:
            return [self._clean_chat_history(chat_history)]

        # Mirror rag.py: flatten history then slice by token count using the embedding encoding.
        document = self._clean_chat_history(chat_history)
        tokens = self.encoding.encode(document)

        chunks: List[str] = []
        for i in range(0, len(tokens), self.chunk_size):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunk = self.encoding.decode(chunk_tokens)
            chunks.append(chunk)

        return chunks

    def _prepare_conversation_context(
        self, conversation_id: str, chat_history: List[dict]
    ) -> Tuple[List[str], List[List[float]]]:
        conversation_key = conversation_id or "conversation"
        history_hash = self._hash_chat_history(chat_history)
        cached = self._load_cached_chunks(conversation_key, history_hash)
        if cached:
            return cached

        chunks, embeddings = self._create_chunks(chat_history)
        self._save_cached_chunks(conversation_key, history_hash, chunks, embeddings)
        return chunks, embeddings

    def _build_initial_state(
        self,
        question: str,
        conversation_context: Tuple[List[str], List[List[float]]],
    ) -> MemoryState:
        chunks, embeddings = conversation_context
        return {
            "question": question,
            "chunks": chunks,
            "embeddings": embeddings,
        }

    def _cache_file_for_conversation(self, conversation_id: str) -> Path:
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in conversation_id)
        return self.rag_cache_dir / f"{safe_id}.json"

    @staticmethod
    def _hash_chat_history(chat_history: List[dict]) -> str:
        serialized = json.dumps(chat_history, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _load_cached_chunks(
        self, conversation_id: str, history_hash: str
    ) -> Tuple[List[str], List[List[float]]] | None:
        cache_path = self._cache_file_for_conversation(conversation_id)
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("version") != 1 or payload.get("history_hash") != history_hash:
            return None
        chunks = payload.get("chunks")
        embeddings = payload.get("embeddings")
        if not isinstance(chunks, list) or not isinstance(embeddings, list):
            return None
        return chunks, embeddings

    def _save_cached_chunks(
        self,
        conversation_id: str,
        history_hash: str,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> None:
        cache_path = self._cache_file_for_conversation(conversation_id)
        payload = {
            "version": 1,
            "conversation_id": conversation_id,
            "history_hash": history_hash,
            "chunks": chunks,
            "embeddings": embeddings,
        }
        try:
            with cache_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        except OSError:
            pass
