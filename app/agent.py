from __future__ import annotations

import ast
import math
import operator
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, TypedDict

import chromadb
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer


DEBUG = True
BASE_DIR = Path(__file__).resolve().parents[1]
CHROMA_DIR = BASE_DIR / "db" / "chroma_db"
COLLECTION_NAME = "ecommerce_kb"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIMENSION = 256
MAX_MESSAGES = 6
DEFAULT_GUIDANCE = (
    "I couldn't find exact info, but here's general guidance based on standard e-commerce operations."
)

NAME_PATTERN = re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z\s'-]{0,48})", re.IGNORECASE)
PREFERENCE_PATTERNS = (
    re.compile(r"\bi like\s+(.+)", re.IGNORECASE),
    re.compile(r"\bi prefer\s+(.+)", re.IGNORECASE),
    re.compile(r"\bmy favorite\s+(.+?)\s+is\s+(.+)", re.IGNORECASE),
)
GREETING_PATTERN = re.compile(r"\b(hi|hello|hey|good morning|good evening)\b", re.IGNORECASE)
THANKS_PATTERN = re.compile(r"\b(thanks|thank you)\b", re.IGNORECASE)
TIME_PATTERN = re.compile(r"\b(time|date|today|day)\b", re.IGNORECASE)
MATH_PATTERN = re.compile(r"^[\d\s\.\+\-\*\/\(\)%]+$")
ECOMMERCE_PATTERN = re.compile(
    r"\b(return|refund|exchange|shipping|delivery|payment|cod|cash on delivery|size|fabric|quality|"
    r"discount|offer|product|policy|timeline|wash|washing|brand|order)\b",
    re.IGNORECASE,
)
MEMORY_PROMPT_PATTERN = re.compile(
    r"\b(what is my name|who am i|do you remember|remember that|my name is|i like|i prefer|favorite)\b",
    re.IGNORECASE,
)

EMBEDDING_MODEL: SentenceTransformer | None = None
CHROMA_COLLECTION = None
THREAD_MEMORY: dict[str, dict[str, Any]] = {}


class CapstoneState(TypedDict, total=False):
    question: str
    messages: list[dict[str, str]]
    route: str
    retrieved: str
    sources: list[str]
    tool_result: str
    answer: str
    user_name: str
    preferences: list[str]
    eval_retries: int
    faithfulness: float
    confidence: str
    fallback_reason: str
    memory_update: str


def debug_log(message: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {message}")


class LocalEmbeddingModel:
    def encode(self, texts: list[str], show_progress_bar: bool = False) -> list[list[float]]:
        _ = show_progress_bar
        return [self._encode_text(text) for text in texts]

    def _encode_text(self, text: str) -> list[float]:
        vector = [0.0] * EMBED_DIMENSION
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            token_hash = int(sha256(token.encode("utf-8")).hexdigest(), 16)
            index = token_hash % EMBED_DIMENSION
            sign = 1.0 if (token_hash >> 8) % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def build_embedding_model() -> SentenceTransformer | LocalEmbeddingModel:
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
        debug_log(f"Using embedding model '{EMBEDDING_MODEL_NAME}'.")
        return model
    except Exception as exc:
        debug_log(f"Falling back to local embedding model: {exc}")
        return LocalEmbeddingModel()


def initialize_kb() -> tuple[SentenceTransformer | LocalEmbeddingModel | None, Any | None]:
    global EMBEDDING_MODEL, CHROMA_COLLECTION

    if EMBEDDING_MODEL is not None and CHROMA_COLLECTION is not None:
        return EMBEDDING_MODEL, CHROMA_COLLECTION

    if not CHROMA_DIR.exists():
        debug_log(f"ChromaDB path missing: {CHROMA_DIR}")
        return None, None

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        model = build_embedding_model()
    except Exception as exc:
        debug_log(f"KB initialization failed: {exc}")
        return None, None

    EMBEDDING_MODEL = model
    CHROMA_COLLECTION = collection
    debug_log("Knowledge base initialized successfully.")
    return EMBEDDING_MODEL, CHROMA_COLLECTION


def query_kb(query: str, n_results: int = 3) -> list[dict[str, str]]:
    embedding_model, collection = initialize_kb()
    if embedding_model is None or collection is None:
        return []

    try:
        raw_query_embedding = embedding_model.encode([query], show_progress_bar=False)
        query_embedding = (
            raw_query_embedding.tolist()[0] if hasattr(raw_query_embedding, "tolist") else raw_query_embedding[0]
        )
        result = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    except Exception as exc:
        debug_log(f"Retrieval error: {exc}")
        return []

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    formatted_results: list[dict[str, str]] = []

    for metadata, document_text in zip(metadatas, documents):
        text = str(document_text or "").strip()
        if not text:
            continue
        preview = text[:220]
        if len(text) > 220:
            preview += "..."
        formatted_results.append(
            {
                "topic": str((metadata or {}).get("topic", "Unknown")).strip() or "Unknown",
                "preview": preview,
            }
        )

    query_terms = set(re.findall(r"\w+", query.lower()))

    def score_item(item: dict[str, str]) -> int:
        topic_lower = item["topic"].lower()
        preview_lower = item["preview"].lower()
        score = (
            3 * sum(1 for term in query_terms if term in topic_lower)
            + sum(1 for term in query_terms if term in preview_lower)
        )
        if query_terms & {"shipping", "delivery", "timeline", "time"} and "shipping" in topic_lower:
            score += 10
        if query_terms & {"return", "refund", "exchange"} and "return" in topic_lower:
            score += 10
        if query_terms & {"payment", "payments", "cod", "upi", "card"} and "payment" in topic_lower:
            score += 10
        return score

    formatted_results.sort(key=score_item, reverse=True)
    return formatted_results


def _trim_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return messages[-MAX_MESSAGES:]


def _extract_name(question: str) -> str | None:
    match = NAME_PATTERN.search(question)
    if not match:
        return None
    return " ".join(match.group(1).split()).strip(" .,!?:;")


def _extract_preferences(question: str) -> list[str]:
    preferences: list[str] = []
    for pattern in PREFERENCE_PATTERNS:
        match = pattern.search(question)
        if not match:
            continue
        if len(match.groups()) == 1:
            value = match.group(1)
        else:
            value = " ".join(group for group in match.groups() if group)
        cleaned = " ".join(value.split()).strip(" .,!?:;")
        if cleaned:
            preferences.append(cleaned)
    return preferences


def _is_math_expression(question: str) -> bool:
    candidate = question.strip()
    if not candidate:
        return False
    lowered = candidate.lower()
    if lowered.startswith(("calculate ", "what is ", "solve ")):
        for prefix in ("calculate ", "what is ", "solve "):
            if lowered.startswith(prefix):
                candidate = candidate[len(prefix) :].strip()
                break
    return bool(candidate) and bool(MATH_PATTERN.fullmatch(candidate))


def _extract_calculation_input(question: str) -> str:
    lowered = question.lower().strip()
    for prefix in ("calculate ", "what is ", "solve "):
        if lowered.startswith(prefix):
            return question[len(prefix) :].strip()
    return question.strip()


def _safe_calculate(expression: str) -> str:
    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            operations: dict[type[ast.AST], Any] = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow,
            }
            op_type = type(node.op)
            if op_type not in operations:
                raise ValueError("Unsupported operator")
            return operations[op_type](left, right)
        if isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.USub):
                return -operand
            raise ValueError("Unsupported unary operator")
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Unsupported expression")

    parsed = ast.parse(expression, mode="eval")
    result = _eval_node(parsed)
    return str(int(result)) if result.is_integer() else str(result)


def _sentence_points(text: str, limit: int = 3) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+|\n+", text)
    points: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = candidate.strip(" -")
        if len(cleaned) < 20:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        points.append(cleaned.rstrip("."))
        if len(points) == limit:
            break
    return points


def _response_title(question: str, sources: list[str]) -> str:
    lowered = question.lower()
    if "return" in lowered or "refund" in lowered:
        return "✅ Return Policy"
    if "shipping" in lowered or "delivery" in lowered:
        return "🚚 Shipping Information"
    if "payment" in lowered or "cod" in lowered:
        return "💳 Payment Information"
    if sources:
        return f"✅ {sources[0]}"
    return "✅ Assistant Response"


def memory_node(state: CapstoneState) -> CapstoneState:
    messages = list(state.get("messages", []))
    question = state.get("question", "").strip()
    if question:
        messages.append({"role": "user", "content": question})

    user_name = state.get("user_name", "").strip()
    preferences = list(state.get("preferences", []))
    memory_updates: list[str] = []

    extracted_name = _extract_name(question)
    if extracted_name and extracted_name != user_name:
        user_name = extracted_name
        memory_updates.append(f"Nice to meet you {user_name}!")

    for preference in _extract_preferences(question):
        if preference not in preferences:
            preferences.append(preference)
            memory_updates.append(f"I'll remember that you like {preference}.")

    return {
        "messages": _trim_messages(messages),
        "user_name": user_name,
        "preferences": preferences,
        "memory_update": " ".join(memory_updates).strip(),
    }


def router_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "").strip()
    if _is_math_expression(question) or TIME_PATTERN.search(question):
        route = "tool"
    elif MEMORY_PROMPT_PATTERN.search(question) or GREETING_PATTERN.search(question) or THANKS_PATTERN.search(question):
        route = "memory"
    elif ECOMMERCE_PATTERN.search(question):
        route = "retrieve"
    else:
        route = "fallback"

    debug_log(f"route selected: {route}")
    return {"route": route}


def retrieval_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "").strip()
    results = query_kb(question)
    combined_chunks: list[str] = []
    sources: list[str] = []

    for item in results[:2]:
        topic = str(item.get("topic", "Unknown")).strip()
        preview = str(item.get("preview", "")).strip()
        if topic:
            sources.append(topic)
        if preview:
            combined_chunks.append(preview)

    deduped_sources = list(dict.fromkeys(source for source in sources if source))
    debug_log(f"retrieved docs: {deduped_sources}")

    if not combined_chunks:
        return {"retrieved": "", "sources": [], "tool_result": "", "fallback_reason": "empty_retrieval"}

    return {
        "retrieved": "\n".join(combined_chunks).strip(),
        "sources": deduped_sources,
        "tool_result": "",
        "fallback_reason": "",
    }


def tool_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "").strip()

    if TIME_PATTERN.search(question):
        tool_result = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elif _is_math_expression(question):
        try:
            tool_result = _safe_calculate(_extract_calculation_input(question))
        except Exception as exc:
            debug_log(f"Tool error: {exc}")
            tool_result = "I couldn't calculate that safely."
    else:
        tool_result = ""

    return {"retrieved": "", "sources": [], "tool_result": tool_result, "fallback_reason": ""}


def skip_node(state: CapstoneState) -> CapstoneState:
    return {"retrieved": "", "sources": [], "tool_result": ""}


def _confidence_label(route: str, retrieved: str, tool_result: str) -> str:
    if route == "retrieve" and retrieved:
        return "High"
    if route in {"memory", "tool"} and (tool_result or True):
        return "Medium"
    return "Low"


def _format_response(title: str, points: list[str], confidence: str, sources: list[str], greeting: str = "") -> str:
    lines = [title]
    if greeting:
        lines.append(greeting)
    lines.extend(f"- {point}" for point in points)
    lines.append(f"- Confidence: {confidence}")
    lines.append(f"- Source: {', '.join(sources) if sources else 'General Store Guidance'}")
    return "\n".join(lines)


def _memory_response(question: str, user_name: str, preferences: list[str], memory_update: str) -> str:
    lowered = question.lower()
    if memory_update:
        return _format_response(
            "✅ Memory Updated",
            [memory_update],
            "Medium",
            ["Conversation Memory"],
        )
    if re.search(r"\b(what is my name|who am i|do you remember my name)\b", lowered):
        message = f"Your name is {user_name}." if user_name else "I don't know your name yet. Try saying 'my name is ...'."
        return _format_response("✅ Memory", [message], "Medium", ["Conversation Memory"])
    if re.search(r"\bwhat do i like\b", lowered):
        if preferences:
            message = "I remember these preferences: " + ", ".join(preferences) + "."
        else:
            message = "I haven't stored any preferences yet."
        return _format_response("✅ Preferences", [message], "Medium", ["Conversation Memory"])
    if THANKS_PATTERN.search(question):
        return _format_response(
            "✅ You're Welcome",
            ["I'm here if you need help with products, delivery, returns, or payments."],
            "Medium",
            ["Conversation Memory"],
        )
    greeting = f"Hi {user_name}!" if user_name else "Hi!"
    return _format_response(
        "✅ Greeting",
        [f"{greeting} I can help with products, delivery, returns, and payment questions."],
        "Medium",
        ["Conversation Memory"],
    )


def answer_node(state: CapstoneState) -> CapstoneState:
    question = state.get("question", "").strip()
    route = state.get("route", "")
    retrieved = state.get("retrieved", "").strip()
    tool_result = state.get("tool_result", "").strip()
    user_name = state.get("user_name", "").strip()
    preferences = list(state.get("preferences", []))
    sources = list(state.get("sources", []))
    memory_update = state.get("memory_update", "").strip()

    if route == "tool" and tool_result:
        answer = _format_response(
            "🧮 Tool Result",
            [f"The result is {tool_result}."],
            "Medium",
            ["Built-in Tool"],
        )
    elif route == "memory":
        answer = _memory_response(question, user_name, preferences, memory_update)
    elif route == "retrieve" and retrieved:
        points = _sentence_points(retrieved)
        greeting = f"For you, {user_name}:" if user_name else ""
        answer = _format_response(
            _response_title(question, sources),
            points or [retrieved],
            "High",
            sources,
            greeting=greeting,
        )
    else:
        fallback_points = [
            DEFAULT_GUIDANCE,
            "Returns are usually limited to eligible items within the policy window.",
            "Delivery time depends on location and courier serviceability.",
            "Payment options typically include prepaid methods and COD where available.",
        ]
        if preferences:
            fallback_points.append("I also remember your preferences: " + ", ".join(preferences) + ".")
        answer = _format_response(
            "✅ General Guidance",
            fallback_points,
            "Low",
            ["General Store Guidance"],
        )

    debug_log(f"final answer: {answer}")
    confidence = re.search(r"Confidence: (\w+)", answer)
    return {"answer": answer, "confidence": confidence.group(1) if confidence else "Low"}


def eval_node(state: CapstoneState) -> CapstoneState:
    retrieved = state.get("retrieved", "").strip()
    route = state.get("route", "")
    retries = int(state.get("eval_retries", 0)) + 1
    faithfulness = 1.0 if route == "retrieve" and retrieved else 0.7 if route in {"memory", "tool"} else 0.4
    return {"faithfulness": faithfulness, "eval_retries": retries}


def save_node(state: CapstoneState) -> CapstoneState:
    messages = list(state.get("messages", []))
    answer = state.get("answer", "").strip()
    if answer:
        messages.append({"role": "assistant", "content": answer})
    return {"messages": _trim_messages(messages)}


def route_after_router(state: CapstoneState) -> str:
    route = state.get("route", "fallback")
    if route == "retrieve":
        return "retrieve"
    if route == "tool":
        return "tool"
    return "skip"


def build_graph():
    graph = StateGraph(CapstoneState)
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("skip", skip_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.add_edge(START, "memory")
    graph.add_edge("memory", "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"retrieve": "retrieve", "tool": "tool", "skip": "skip"},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("answer", "eval")
    graph.add_edge("eval", "save")
    graph.add_edge("save", END)
    return graph.compile()


GRAPH = build_graph()


def ask(question: str, thread_id: str = "1") -> str:
    stored_state = THREAD_MEMORY.get(
        thread_id,
        {"messages": [], "user_name": "", "preferences": [], "eval_retries": 0, "faithfulness": 0.0},
    )

    initial_state: CapstoneState = {
        "question": question,
        "messages": list(stored_state.get("messages", [])),
        "route": "",
        "retrieved": "",
        "sources": [],
        "tool_result": "",
        "answer": "",
        "user_name": str(stored_state.get("user_name", "")),
        "preferences": list(stored_state.get("preferences", [])),
        "eval_retries": int(stored_state.get("eval_retries", 0)),
        "faithfulness": float(stored_state.get("faithfulness", 0.0)),
        "confidence": "",
        "fallback_reason": "",
        "memory_update": "",
    }

    try:
        final_state = GRAPH.invoke(initial_state)
    except Exception as exc:
        debug_log(f"Agent execution error: {exc}")
        return _format_response(
            "✅ General Guidance",
            [DEFAULT_GUIDANCE, "Please try your question again in a simpler form."],
            "Low",
            ["General Store Guidance"],
        )

    THREAD_MEMORY[thread_id] = {
        "messages": list(final_state.get("messages", [])),
        "user_name": str(final_state.get("user_name", "")),
        "preferences": list(final_state.get("preferences", [])),
        "eval_retries": int(final_state.get("eval_retries", 0)),
        "faithfulness": float(final_state.get("faithfulness", 0.0)),
    }
    return str(final_state.get("answer", DEFAULT_GUIDANCE))


if __name__ == "__main__":
    print(ask("What is your return policy?"))
    print(ask("My name is Alex"))
    print(ask("How long is delivery?"))
