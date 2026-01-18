import json
import os
import time
from datetime import datetime, timedelta, timezone
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, TypedDict

import tiktoken
from dotenv import load_dotenv
from jinja2 import Template
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from tqdm import tqdm
from json_repair import repair_json

load_dotenv()

FINAL_ANSWER_PROMPT = """
You are a helpful expert assistant answering questions from lme_experiment users based on the provided context.

# CONTEXT:
You have access to facts and entities from a conversation.
The memories include timestamps that represent when the events actually happened, not when they were mentioned.

# INSTRUCTIONS:
1. Carefully analyze all provided memories.
2. Pay special attention to the timestamps to determine the answer.
3. If the question asks about a specific event or fact, look for direct evidence in the memories.
4. If the memories contain contradictory information, prioritize the most recent memory.
5. Always convert relative time references to specific dates, months, or years.
6. Be as specific as possible when talking about people, places, and events.
7. Treat memory timestamps as the time the event occurred, not the time it was discussed.

Clarification:
When interpreting memories, use the timestamp to determine when the described event happened, not when someone talked about the event.

Example:
Memory: (2023-03-15T16:33:00Z) I went to the vet yesterday.
Question: What day did I go to the vet?
Correct Answer: March 15, 2023
Explanation:
Even though the phrase says "yesterday," the timestamp shows the event was recorded as happening on March 15th. Therefore, the actual vet visit happened on that date, regardless of the word "yesterday" in the text.

# APPROACH (Think step by step):
1. First, examine all memories that contain information related to the question.
2. Examine the timestamps and content of these memories carefully.
3. Look for explicit mentions of dates, times, locations, or events that answer the question.
4. If the answer requires calculation (e.g., converting relative time references), show your work.
5. Formulate a precise, concise answer based solely on the evidence in the memories.
6. Double-check that your answer directly addresses the question asked.
7. Ensure your final answer is specific and avoids vague time references.

# Evidence:
{{EVIDENCE}}

# Draft answer:
{{DRAFT_ANSWER}}

Answer:
"""

GENERATION_SYSTEM_PROMPT_TEMPLATE = """You are a memory agent that plans how to gather evidence before producing the final response shown to the user.
Always reply with a strict JSON object using this schema:
- evidence: JSON array of concise factual bullet strings relevant to the user's question; preserve key numbers/names/time references. If exact values are unavailable, include the most specific verified information (year/range) without speculation. Never mention missing or absent information here—"gaps" will do that.
- gaps: gaps between the question and evidence that prevent a complete answer.
- decision: one of ["retrieve","answer","reflect"]. Choose {decision_directive}.

Only include these conditional keys:
- retrieval_query: only when decision == "retrieve". Provide a STANDALONE search string; short (5-15 tokens).
    * BAD Query: "the date" (lacks context).
    * GOOD Query: "graduation ceremony date" (specific).
    * STRATEGY: 
        1. Search for the ANCHOR EVENT. (e.g. Question: "What happened 2 days after X?", Query: "timestamp of event X").
        2. Search for the MAPPED ENTITY. (e.g. Question: "Weather in the Windy City", Query: "weather in Chicago").
- detailed_answer: only when decision == "answer"; response using current evidence (keep absolute dates, avoid speculation). If evidence is limited, provide only what is known, or make cautious inferences grounded solely in that limited evidence. Do not mention missing or absent information in this field.
- reasoning: only when decision == "reflect"; if further retrieval is unlikely, use current evidence to think step by step through the evidence and gaps, and work toward the answer, including any time normalization.

Never include extra keys or any text outside the JSON object."""

normal_decision = """"reflect" if you need to think about the evidence and gaps; choose "answer" ONLY when evidence is solid and no gaps are noted; choose "retrieve" otherwise."""

GENERATION_USER_PROMPT = """# Question
{question}

# Evidence
{evidence_block}

# Gaps
{gap_block}

# Memory snippets
{raw_block}

# Reasoning
{reasoning_block}

# Prior Query
{last_query}

# INSTRUCTIONS:
1. Update the evidence as a JSON ARRAY of concise factual bullets that directly help answer the question (preserve key numbers/names/time references; use the most specific verified detail without speculation).
2. Update gaps: remove resolved items, add new missing specifics blocking a full answer, and set to "None" when nothing is missing.
3. If you produce a retrieval_query, make sure it differs from the previous query.
4. Decide the next action and return ONLY the JSON object described in the system prompt."""

class MemoryState(TypedDict, total=False):
    question: str
    conversation_id: Optional[str]
    chunks: List[str]
    embeddings: List[List[float]]
    evidence: List[str]
    raw_snippets: List[str]
    masked_indices: List[int]
    masked_memory_keys: List[str]
    generation_trace: List[str]
    generation_log: List[str]
    routing_trace: List[str]
    reasoning_trace: List[str]
    retrieval_history: List[str]
    retrieval_attempts: int
    last_seen_retrieval_attempts: int
    search_times: List[float]
    retrieval_query: Optional[str]
    retrieval_queries: List[str]
    evidence_gap_tracker: List[Dict[str, Any]]
    gaps: Optional[str]
    pending_answer: Optional[str]
    forced_decision: Optional[Literal["answer", "retrieve", "reflect"]]
    last_decision: Literal["answer", "retrieve", "reflect"]
    answer: Optional[str]
    latest_reasoning: Optional[str]
    response_time: float
    iteration_count: int
    consecutive_reflects: int
    total_retrieved_tokens: int
    total_completion_tokens: int
    last_retrieved_token_iteration: Optional[int]
    langmem_retriever: Any
    speakers: List[str]
    speaker_cursor: int


class BaseMemR3Manager(ABC):
    """Shared workflow and bookkeeping for MemR3 memory agents."""

    def __init__(self, data_path: str, top_k: int, max_iterations: int, max_reflect_streak: int = 1):
        self.data_path = data_path
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.max_reflect_streak = max_reflect_streak

        self.model = os.getenv("MODEL")
        if not self.model:
            raise ValueError("MODEL environment variable must be set.")

        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )

        target_model_for_encoding = os.getenv("EMBEDDING_MODEL") or self.model
        try:
            self.encoding = tiktoken.encoding_for_model(target_model_for_encoding)
        except Exception:
            try:
                self.encoding = tiktoken.get_encoding(target_model_for_encoding)
            except Exception:
                self.encoding = tiktoken.get_encoding("cl100k_base")

        self.answer_template = Template(FINAL_ANSWER_PROMPT)
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(MemoryState)

        def generation_node(state: MemoryState) -> MemoryState:
            # 1. Initialization and cleanup
            current_attempts = state.get("retrieval_attempts", 0)
            last_seen_attempts = state.get("last_seen_retrieval_attempts", 0)
            state["last_seen_retrieval_attempts"] = current_attempts
            
            # Check if retrieval is empty, if so and not forced, then force reflect
            saw_new_retrieval = current_attempts > last_seen_attempts
            empty_after_retrieval = saw_new_retrieval and not state.get("raw_snippets")
            
            if empty_after_retrieval and not state.get("forced_decision"):
                state.setdefault("routing_trace", []).append(
                    "Generation override: latest retrieval returned no evidence; forcing reflect decision."
                )
                state["forced_decision"] = "reflect"

            # 2. Prepare prompt data
            raw_snippets = state.get("raw_snippets", [])
            raw_block = "\n---\n".join(raw_snippets) if raw_snippets else "No retrieved snippets yet."
            retrieval_queries = state.get("retrieval_queries", [])
            last_query = retrieval_queries[-1] if retrieval_queries else "None"
            reasoning_block = state.get("latest_reasoning") or "No prior reasoning."
            evidence_block = self._format_evidence_block(self._normalize_evidence(state.get("evidence")))
            gap_block = state.get("gaps") or "No gaps noted."

            # 3. Iteration count
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            iteration = state["iteration_count"]
            # print(f"[Generation Iteration] {iteration}/{self.max_iterations}")

            # 4. Determine prompt directive
            forced_decision = state.get("forced_decision")
            decision_directive = forced_decision if forced_decision else normal_decision
            
            system_prompt = GENERATION_SYSTEM_PROMPT_TEMPLATE.format(decision_directive=decision_directive)
            user_prompt = GENERATION_USER_PROMPT.format(
                question=state["question"],
                raw_block=raw_block,
                evidence_block=evidence_block,
                gap_block=gap_block,
                reasoning_block=reasoning_block,
                last_query=last_query,
            )

            # 5. Call LLM
            max_retries = 5
            last_error = ""
            for attempt in range(max_retries + 1):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        temperature=0,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    )
                    self._record_usage(state, response)
                    content = response.choices[0].message.content.strip()
                    # print(f"[Generation Output] {content}")
                    parsed = self._parse_generation_response(content)
                    
                    # 6. Decision handling
                    decision = parsed["decision"]
                    decision_str = parsed.get("decision") or ""
                    detail_part = (
                        parsed.get("reasoning")
                        or parsed.get("detailed_answer")
                        or parsed.get("retrieval_query")
                        or ""
                    )
                    log_entry = f"{decision_str} | {detail_part}".strip(" |")
                    if log_entry:
                        state.setdefault("generation_log", []).append(str(log_entry))
                    
                    # If there is a forced decision, override the LLM's decision
                    if forced_decision and forced_decision in {"answer", "retrieve", "reflect"}:
                        if decision != forced_decision:
                            state.setdefault("routing_trace", []).append(
                                f"Generation override: enforcing forced decision {forced_decision} instead of {decision}."
                            )
                        decision = forced_decision
                    
                    # Check iteration limit
                    limit_reached = iteration >= self.max_iterations
                    if limit_reached and decision != "answer":
                        decision = "answer"
                        state["gaps"] = f"Reached iteration limit ({self.max_iterations}); forcing final answer."
                    else:
                        state["gaps"] = parsed.get("gaps")

                    # Update State
                    evidence_summary = self._normalize_evidence(parsed.get("evidence"))
                    state["evidence"] = evidence_summary
                    
                    state["last_decision"] = decision  # Routing Edge will read this value
                    
                    retrieval_query = parsed.get("retrieval_query")
                    state["retrieval_query"] = retrieval_query
                    if retrieval_query:
                        state.setdefault("retrieval_queries", []).append(retrieval_query)
                    
                    state["pending_answer"] = self._normalize_answer(parsed.get("detailed_answer"))
                    
                    if decision == "reflect":
                        state["latest_reasoning"] = parsed.get("reasoning") or state.get("latest_reasoning")
                        state["consecutive_reflects"] = state.get("consecutive_reflects", 0) + 1
                        state.setdefault("reasoning_trace", []).append(f"Reflect: {state['latest_reasoning']}")
                    else:
                        state["latest_reasoning"] = None
                        state["consecutive_reflects"] = 0

                    self._append_evidence_gap_snapshot(state, iteration, decision)
                        
                    return state
                except Exception as exc:
                    last_error = str(exc)
                    if attempt == max_retries:
                        break
                    time.sleep(20)
            raise RuntimeError(f"Generation node failed after retries: {last_error}")

        def routing_node(state: MemoryState) -> MemoryState:
            decision = state.get("last_decision", "reflect")
            forced = state.get("forced_decision")
            state.setdefault("routing_trace", []).append(f"Routing check. Last Decision: {decision}. Forced: {forced}")
            return state

        def routing_edges(state: MemoryState) -> Literal["answer", "retrieve", "reflect"]:
            return state.get("last_decision", "reflect")

        def retrieve_node(state: MemoryState) -> MemoryState:
            # Keep only the latest retrieval snippets to avoid unbounded growth.
            state["raw_snippets"] = []
            self._retrieve(state)
            state["retrieval_query"] = None
            state["last_decision"] = "reflect"
            state["consecutive_reflects"] = 0

            state['forced_decision'] = None
            return state

        def reflect_node(state: MemoryState) -> MemoryState:
            # reasoning = state.get("latest_reasoning") or "No reasoning provided."
            
            is_forced_reflect = state.get("forced_decision") == "reflect"
            is_streak_limit = state.get("consecutive_reflects", 0) >= self.max_reflect_streak
            
            has_opportunity = self._has_retrieval_opportunity(state)

            if (is_forced_reflect or is_streak_limit) and has_opportunity:
                state["forced_decision"] = "retrieve"
                reason_msg = "Forced reflect completed" if is_forced_reflect else "Max reflect streak reached"
                state.setdefault("routing_trace", []).append(
                    f"Reflect node: {reason_msg}. Chaining forced decision to 'retrieve'."
                )
            else:
                # if we reach here, either reflect was not forced, or there are no more retrieval opportunities
                state["forced_decision"] = None

            return state

        def answer_node(state: MemoryState) -> MemoryState:
            answer, duration = self._generate_answer(
                state["question"],
                state.get("evidence", []),
                state,
                draft_answer=state.get("pending_answer"),
            )
            state["answer"] = answer
            state["response_time"] = round(duration, 3)
            return state

        workflow.add_node("generation", generation_node)
        workflow.add_node("routing", routing_node)
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("reflect", reflect_node)
        workflow.add_node("answer", answer_node)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generation")
        workflow.add_edge("generation", "routing")
        workflow.add_conditional_edges(
            "routing",
            routing_edges,
            path_map={
                "answer": "answer",
                "retrieve": "retrieve",
                "reflect": "reflect",
            },
        )
        workflow.add_edge("reflect", "generation")
        workflow.add_edge("answer", END)

        return workflow.compile()

    @abstractmethod
    def _has_retrieval_opportunity(self, state: MemoryState) -> bool:
        """Return True when the retriever can still surface new evidence."""

    @abstractmethod
    def _retrieve(self, state: MemoryState) -> None:
        """Populate evidence using the concrete retriever implementation."""

    @abstractmethod
    def _prepare_conversation_context(self, conversation_id: str, chat_history: List[Dict[str, Any]]) -> Any:
        """Prepare the reusable retrieval context for a conversation."""

    @abstractmethod
    def _build_initial_state(self, question: str, conversation_context: Any) -> MemoryState:
        """Return the state additions needed before invoking the workflow."""

    def _parse_generation_response(self, content: str) -> Dict[str, Any]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            repaired = repair_json(content)
            data = json.loads(repaired)

        decision = data.get("decision", "reflect").lower()
        if decision not in {"answer", "retrieve"}:
            decision = "reflect"

        return {
            "evidence": data.get("evidence"),
            "gaps": data.get("gaps"),
            "retrieval_query": data.get("retrieval_query"),
            "detailed_answer": data.get("detailed_answer"),
            "reasoning": data.get("reasoning"),
            "decision": decision,
        }

    @staticmethod
    def _normalize_answer(answer: Any) -> Optional[str]:
        if answer is None:
            return None
        if isinstance(answer, str):
            trimmed = answer.strip()
            return trimmed or None
        try:
            return json.dumps(answer, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(answer)

    @staticmethod
    def _normalize_evidence(evidence: Any) -> List[str]:
        if evidence is None:
            return []
        if isinstance(evidence, list):
            normalized: List[str] = []
            for item in evidence:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    normalized.append(text)
            return normalized
        if isinstance(evidence, str):
            text = evidence.strip()
            return [text] if text else []
        text = str(evidence).strip()
        return [text] if text else []

    @staticmethod
    def _format_evidence_block(evidence: List[str]) -> str:
        if not evidence:
            return "No evidence yet."
        return "\n".join(f"- {item}" for item in evidence)

    @staticmethod
    def _extract_usage_value(usage: Any, keys: List[str]) -> int:
        for key in keys:
            value = None
            if hasattr(usage, key):
                value = getattr(usage, key)
            elif isinstance(usage, dict):
                value = usage.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return 0

    def _count_tokens(self, text: str) -> int:
        try:
            return len(self.encoding.encode(text))
        except Exception:
            return len(text.split())

    def _record_retrieved_tokens(self, state: MemoryState) -> None:
        raw_snippets = state.get("raw_snippets") or []
        if not raw_snippets:
            return

        raw_block = "\n---\n".join(str(snippet) for snippet in raw_snippets)
        tokens = self._count_tokens(raw_block)

        iteration = state.get("iteration_count")
        last_counted = state.get("last_retrieved_token_iteration")

        if iteration is not None and last_counted == iteration:
            return

        state["total_retrieved_tokens"] = state.get("total_retrieved_tokens", 0) + tokens
        state["last_retrieved_token_iteration"] = iteration

    def _record_usage(self, state: MemoryState, response: Any):
        if state is None:
            return

        self._record_retrieved_tokens(state)

        usage = getattr(response, "usage", None)
        if usage is None:
            return

        completion_tokens = self._extract_usage_value(
            usage, ["completion_tokens", "completion_tokens_total"]
        )
        state["total_completion_tokens"] = state.get("total_completion_tokens", 0) + completion_tokens

    @staticmethod
    def _append_evidence_gap_snapshot(state: MemoryState, iteration: int, decision: str) -> None:
        tracker = state.setdefault("evidence_gap_tracker", [])
        evidence_snapshot = BaseMemR3Manager._normalize_evidence(state.get("evidence"))
        snapshot = {
            "iteration": iteration,
            "decision": decision,
            "evidence": evidence_snapshot,
            "gaps": state.get("gaps"),
        }
        tracker.append(snapshot)

    def _generate_answer(
        self,
        question: str,
        evidence: List[str] | str,
        state: Optional[MemoryState] = None,
        draft_answer: Optional[str] = None,
    ) -> tuple[str, float]:
        evidence_list = evidence if isinstance(evidence, list) else self._normalize_evidence(evidence)
        evidence_block = self._format_evidence_block(evidence_list) if evidence_list else "No evidence provided."
        draft_block = draft_answer if draft_answer else "No draft answer."
        system_prompt = self.answer_template.render(
            EVIDENCE=evidence_block,
            DRAFT_ANSWER=draft_block,
        )
        start = time.time()
        max_retries = 1
        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    raise ValueError("Empty answer returned from model.")
                if state is not None:
                    self._record_usage(state, response)
                duration = time.time() - start
                return content, duration
            except Exception as exc:
                last_error = str(exc)
                if attempt == max_retries:
                    break
                time.sleep(20)
        raise RuntimeError(f"Answer generation failed after retries: {last_error}")

    def _ensure_final_answer(self, state: MemoryState) -> MemoryState:
        if state.get("answer"):
            return state

        answer, duration_raw = self._generate_answer(
            state["question"],
            state.get("evidence", []),
            state,
            draft_answer=state.get("pending_answer"),
        )
        state["answer"] = answer
        state["response_time"] = round(duration_raw, 3)
        state.setdefault("routing_trace", []).append("Answer generated after workflow terminated early.")
        return state

    def _base_initial_state(self, question: str, conversation_id: Optional[str] = None) -> MemoryState:
        normalized_id = str(conversation_id) if conversation_id is not None else None
        return {
            "question": question,
            "conversation_id": normalized_id,
            "evidence": [],
            "raw_snippets": [],
            "masked_indices": [],
            "masked_memory_keys": [],
            "generation_trace": [],
            "generation_log": [],
            "routing_trace": [],
            "reasoning_trace": [],
            "retrieval_history": [],
            "retrieval_attempts": 0,
            "search_times": [],
            "last_decision": "reflect",
            "latest_reasoning": None,
            "gaps": None,
            "evidence_gap_tracker": [],
            "response_time": 0.0,
            "iteration_count": 0,
            "consecutive_reflects": 0,
            "total_retrieved_tokens": 0,
            "total_completion_tokens": 0,
            "last_retrieved_token_iteration": None,
            "retrieval_queries": [],
            "forced_decision": None,
            "last_seen_retrieval_attempts": 0,
        }

    def _invoke_workflow(self, initial_state: MemoryState) -> MemoryState:
        recursion_limit = max(self.max_iterations * 4, 10)
        final_state = self.workflow.invoke(initial_state, config={"recursion_limit": recursion_limit})
        final_state["generation_trace"] = final_state.get("generation_log", [])
        return self._ensure_final_answer(final_state)

    def _run_graph_for_question(
        self, question: str, conversation_context: Any, conversation_id: Optional[str] = None
    ) -> MemoryState:
        base_state = self._base_initial_state(question, conversation_id)
        extra_state = self._build_initial_state(question, conversation_context)
        if (
            base_state.get("conversation_id") is not None
            and extra_state.get("conversation_id") is None
            and "conversation_id" in extra_state
        ):
            extra_state = {k: v for k, v in extra_state.items() if k != "conversation_id"}
        base_state.update(extra_state)
        return self._invoke_workflow(base_state)

    def process_all_conversations(self, output_file_path: str):
        with open(self.data_path, "r") as f:
            dataset = json.load(f)

        final_results, answered_questions = self._load_existing_results(output_file_path)

        if isinstance(dataset, dict):
            iterable = dataset.items()
        elif isinstance(dataset, list):
            iterable = enumerate(dataset)
        else:
            raise ValueError("Unsupported dataset format for memr3 processing.")

        for key_raw, value in tqdm(iterable, desc="Processing conversations"):
            key = str(key_raw)
            is_longmemeval = self._is_longmemeval_entry(value)
            if is_longmemeval:
                question_id = value.get("question_id") if isinstance(value, dict) else None
                if question_id:
                    key = str(question_id)
                chat_history = self._build_longmemeval_chat_history(value)
                questions = self._build_longmemeval_questions(value)
            else:
                chat_history = value.get("conversation") if isinstance(value, dict) else None
                questions = []
                if isinstance(value, dict):
                    questions = value.get("question") or value.get("qa") or []
            if chat_history is None:
                chat_history = []
            conversation_context = self._prepare_conversation_context(key, chat_history)
            answered_for_key = answered_questions.setdefault(key, set())

            for item in tqdm(questions, desc="Answering questions", leave=False):
                question = item.get("question") if isinstance(item, dict) else None
                answer = item.get("answer", "") if isinstance(item, dict) else ""
                category_raw = (
                    item.get("category", item.get("question_type", 0))
                    if isinstance(item, dict)
                    else 0
                )
                try:
                    category = int(category_raw)
                except (TypeError, ValueError):
                    category = category_raw

                if not question:
                    continue
                if question in answered_for_key:
                    continue
                if category == 5:  # Skip category 5 questions
                    continue
                # if category not in [2,3]:  # Only deal with category 3 questions
                #     continue

                state = self._run_graph_for_question(question, conversation_context, conversation_id=key)
                state["conversation_id"] = key
                result_entry = self._build_result_entry(question, answer, category, state)
                final_results[key].append(result_entry)
                answered_for_key.add(question)

                with open(output_file_path, "w") as output_file:
                    json.dump(final_results, output_file, indent=4)

        with open(output_file_path, "w") as f:
            json.dump(final_results, f, indent=4)

    @staticmethod
    def _is_longmemeval_entry(entry: Any) -> bool:
        return (
            isinstance(entry, dict)
            and isinstance(entry.get("haystack_sessions"), list)
            and isinstance(entry.get("question"), str)
        )

    @staticmethod
    def _format_longmemeval_timestamp(date_str: Optional[str], offset_seconds: int) -> str:
        if not date_str:
            return "unknown"
        try:
            base_time = datetime.strptime(date_str, "%Y/%m/%d (%a) %H:%M")
        except ValueError:
            return date_str
        base_time = base_time.replace(tzinfo=timezone.utc)
        timestamp = base_time + timedelta(seconds=offset_seconds)
        return timestamp.isoformat().replace("+00:00", "Z")

    def _build_longmemeval_chat_history(self, entry: Dict[str, Any]) -> List[Dict[str, str]]:
        sessions = entry.get("haystack_sessions") or []
        session_dates = entry.get("haystack_dates") or []
        chat_history: List[Dict[str, str]] = []

        for session_idx, session in enumerate(sessions):
            if not session:
                continue
            date_str = session_dates[session_idx] if session_idx < len(session_dates) else None
            for msg_idx, msg in enumerate(session):
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if content is None:
                    continue
                chat_history.append(
                    {
                        "timestamp": self._format_longmemeval_timestamp(date_str, msg_idx),
                        "speaker": msg.get("role") or "unknown",
                        "text": content,
                    }
                )

        return chat_history

    @staticmethod
    def _build_longmemeval_questions(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        question = entry.get("question")
        if not question:
            return []
        return [
            {
                "question": question,
                "answer": entry.get("answer", ""),
                "category": entry.get("question_type", 0),
            }
        ]

    def _load_existing_results(
        self, output_file_path: str
    ) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, set[str]]]:
        final_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        answered_questions: Dict[str, set[str]] = defaultdict(set)
        if os.path.exists(output_file_path):
            try:
                with open(output_file_path, "r") as existing_file:
                    existing_data = json.load(existing_file)
                if isinstance(existing_data, dict):
                    for key, entries in existing_data.items():
                        if isinstance(entries, list):
                            final_results[key] = entries
                            answered_questions[key] = {
                                entry.get("question")
                                for entry in entries
                                if isinstance(entry, dict) and entry.get("question")
                            }
            except (json.JSONDecodeError, OSError):
                pass
        return final_results, answered_questions

    def _build_result_entry(
        self,
        question: str,
        answer: str,
        category: str,
        state: MemoryState,
    ) -> Dict[str, Any]:
        total_search_time = round(sum(state.get("search_times", [])), 3)
        response_time = round(state.get("response_time", 0.0), 3)
        response_text = state.get("answer", "")
        evidence = self._normalize_evidence(state.get("evidence"))
        retrieved_tokens = state.get("total_retrieved_tokens", 0)
        completion_tokens = state.get("total_completion_tokens", 0)
        conversation_id = state.get("conversation_id")

        return {
            "question": question,
            "answer": answer,
            "category": category,
            "conversation_id": conversation_id,
            "response": response_text,
            "evidence": evidence,
            "evidence_gap_tracker": state.get("evidence_gap_tracker", []),
            "generation_trace": state.get("generation_trace", []),
            "routing_trace": state.get("routing_trace", []),
            "reasoning_trace": state.get("reasoning_trace", []),
            "retrieval_history": state.get("retrieval_history", []),
            "search_time": total_search_time,
            "response_time": response_time,
            "retrieved_tokens": retrieved_tokens,
            "completion_tokens": completion_tokens,
        }
