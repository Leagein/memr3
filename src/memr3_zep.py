import asyncio
import os
import time
from typing import Any, Dict, List, Set, Tuple

from dotenv import load_dotenv
from zep_cloud.client import AsyncZep

from src.memr3_base import BaseMemR3Manager, MemoryState
from zep_cloud import EntityEdge, EntityNode

load_dotenv()

TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts for the conversation along with the datetime of the event that the fact refers to.
If a fact mentions something happening a week ago, then the datetime will be the date time of last week and not the datetime
of when the fact was stated.
Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.
    
<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""

def compose_search_context(edges: list[EntityEdge], nodes: list[EntityNode]) -> str:
    facts = [f'  - {edge.fact} (event_time: {edge.valid_at})' for edge in edges]
    entities = [f'  - {node.name}: {node.summary}' for node in nodes]
    return TEMPLATE.format(facts='\n'.join(facts), entities='\n'.join(entities))

class MemR3ZepManager(BaseMemR3Manager):
    """MemR3 memory agent backed by Zep graph search."""

    def __init__(
        self,
        data_path: str = "dataset/locomo10.json",
        max_iterations: int = 3,
        run_id: str | None = None,
        search_limit: int = 20,
        max_retrieval_attempts: int | None = None,
    ):
        self.run_id = run_id or os.getenv("ZEP_RUN_ID", "")
        self.search_limit = search_limit
        self.max_retrieval_attempts = max_retrieval_attempts or max_iterations
        self.zep_api_key = os.getenv("ZEP_API_KEY")
        self.zep_base_url = os.getenv("ZEP_BASE_URL", "https://api.getzep.com/api/v2")

        if not os.getenv("MODEL"):
            raise ValueError("MODEL environment variable must be set.")
        if not self.zep_api_key:
            raise ValueError("ZEP_API_KEY environment variable must be set.")

        super().__init__(data_path=data_path, top_k=search_limit, max_iterations=max_iterations)

    def _has_retrieval_opportunity(self, state: MemoryState) -> bool:
        attempts = state.get("retrieval_attempts", 0)
        return attempts < self.max_retrieval_attempts

    def _retrieve(self, state: MemoryState) -> None:
        query = state["question"]

        if state.get("retrieval_query"):
            query = f"{query}. {state['retrieval_query']}"

        conversation_id_raw = state.get("conversation_id")
        conversation_id = str(conversation_id_raw) if conversation_id_raw not in {None, ""} else None
        if conversation_id is None:
            state.setdefault("routing_trace", []).append("No conversation id available; skipping retrieval.")
            return

        graph_id = f"locomo_experiment_user_{conversation_id}"

        start = time.time()
        masked = set(state.get("masked_memory_keys") or [])
        context, used_keys = asyncio.run(self._search_single_graph(query, graph_id, masked))
        duration = time.time() - start
        state["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1

        state.setdefault("search_times", []).append(round(duration, 3))
        state.setdefault("retrieval_history", [])
        state.setdefault("masked_memory_keys", [])

        if not context:
            print("[Retrieval] Zep returned no new context (all results masked or empty).")
            state["forced_decision"] = "reflect"
            return

        state["masked_memory_keys"].extend(used_keys)

        snippet = context.replace("\n", " ").strip()
        if len(snippet) > 200:
            snippet = f"{snippet[:200]}..."

        state["raw_snippets"].append(context)
        history_entry = f"{graph_id} :: {snippet}"
        state["retrieval_history"].append(history_entry)

    def _prepare_conversation_context(
        self, conversation_id: str, chat_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {
            "conversation_id": str(conversation_id),
        }

    def _build_initial_state(self, question: str, conversation_context: Dict[str, Any]) -> MemoryState:
        conversation_id = None
        if isinstance(conversation_context, dict):
            conversation_id = conversation_context.get("conversation_id")
        elif isinstance(conversation_context, (str, int)):
            conversation_id = str(conversation_context)
        return {
            "conversation_id": str(conversation_id) if conversation_id is not None else None,
            "retrieval_attempts": 0,
            "last_seen_retrieval_attempts": 0,
        }

    def _extract_field(self, obj: Any, field: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(field)
        return getattr(obj, field, None)

    def _make_memory_key(self, item: Any, kind: str) -> str:
        id_fields = ["id", "uuid", "guid", "memory_id", "object_id"]
        for field in id_fields:
            value = self._extract_field(item, field)
            if value:
                return f"{kind}:{value}"

        fact = self._extract_field(item, "fact")
        valid_at = self._extract_field(item, "valid_at")
        name = self._extract_field(item, "name")
        summary = self._extract_field(item, "summary")
        timestamp = valid_at or self._extract_field(item, "timestamp") or self._extract_field(item, "created_at")

        primary = fact or name or summary or repr(item)
        return f"{kind}:{primary}:{timestamp}"

    def _filter_unmasked(
        self, items: List[Any], masked: Set[str], kind: str
    ) -> Tuple[List[Any], List[str]]:
        filtered: List[Any] = []
        new_keys: List[str] = []
        local_seen: Set[str] = set()

        for item in items or []:
            key = self._make_memory_key(item, kind)
            if key in masked or key in local_seen:
                continue
            filtered.append(item)
            new_keys.append(key)
            local_seen.add(key)
        return filtered, new_keys

    async def _search_single_graph(
        self, query: str, graph_id: str, masked_keys: Set[str] | None = None
    ) -> tuple[str, List[str]]:
        """Return context preferring unmasked items; backfill with masked ones up to search_limit."""
        zep = AsyncZep(api_key=self.zep_api_key, base_url=self.zep_base_url)

        masked_keys = set(masked_keys or [])
        fetch_limit = self.search_limit + len(masked_keys)  # over-fetch to offset masked items

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            node_task = zep.graph.search(
                query=query,
                graph_id=graph_id,
                scope="nodes",
                reranker="rrf",
                limit=min(fetch_limit, 50),
            )
            edge_task = zep.graph.search(
                query=query,
                graph_id=graph_id,
                scope="edges",
                reranker="cross_encoder",
                limit=min(fetch_limit, 50),
            )
            try:
                node_results, edge_results = await asyncio.gather(node_task, edge_task)
                break
            except Exception as exc:
                if attempt == max_attempts:
                    raise
                print(f"[Retrieval] Zep graph search attempt {attempt} failed for {graph_id}; retrying... {exc}")
                await asyncio.sleep(0.5 * attempt)
        nodes = getattr(node_results, "nodes", []) or []
        edges = getattr(edge_results, "edges", []) or []

        def _select(items: List[Any], kind: str) -> Tuple[List[Any], List[str]]:
            selected: List[Any] = []
            keys: List[str] = []
            seen: Set[str] = set()

            # First pass: unmasked items in original rank order.
            for item in items:
                key = self._make_memory_key(item, kind)
                if key in seen or key in masked_keys:
                    continue
                selected.append(item)
                keys.append(key)
                seen.add(key)
                if len(selected) >= self.search_limit:
                    return selected, keys

            # Second pass: allow masked items to backfill to the limit.
            for item in items:
                if len(selected) >= self.search_limit:
                    break
                key = self._make_memory_key(item, kind)
                if key in seen:
                    continue
                selected.append(item)
                keys.append(key)
                seen.add(key)

            return selected, keys

        filtered_edges, edge_keys = _select(edges, "edge")
        filtered_nodes, node_keys = _select(nodes, "node")

        if not filtered_edges and not filtered_nodes:
            return "", []

        used_keys = edge_keys + node_keys
        return compose_search_context(filtered_edges, filtered_nodes), used_keys
