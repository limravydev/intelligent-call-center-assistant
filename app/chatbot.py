import os
import random
import re
from typing import Tuple, List, Optional

from google import genai
from google.genai import types as genai_types

from app.rag import retrieve_context

# ------------------------------------------------------
# Model + constants
# ------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"

# Banking / tech terms that strongly indicate a real case
BANK_KEYWORDS = [
    # products
    "account", "savings", "current", "checking",
    "fixed deposit", "time deposit", "term deposit",
    "deposit", "withdrawal", "withdraw", "transfer",
    "remittance", "loan", "credit", "card", "debit",
    "atm", "cif", "kyc",
    # channels / apps
    "mobile app", "mobile banking", "internet banking",
    "online banking", "ibanking", "i-banking", "app login",
    # auth / security
    "otp", "one time password", "one-time password",
    "password", "pin", "passcode", "login", "log in",
    "locked", "block", "blocked", "lock", "unlock",
    # docs / slips
    "statement", "passbook", "slip",
    # misc
    "fees", "charges", "interest", "rate", "limit",
    "transaction", "failed transaction",

     # login & authentication keywords
    "login", "log in", "cannot login", "can't login",
    "failed login", "mobile app login", "login issue",
    "password", "otp", "verification", "auth", "authentication",
]

# More generic “problem” tokens – when combined with banking words,
# they are very strong evidence this is NOT smalltalk.
PROBLEM_KEYWORDS = [
    "cannot", "can't", "cant",
    "error", "issue", "problem", "failed", "not working",
    "still cannot", "still can't", "doesn't work", "does not work",
]

# Short smalltalk / meta-intent phrases
SMALLTALK_KEYWORDS = {
    "hi", "hello", "hey",
    "good morning", "good afternoon", "good evening",
    "thanks", "thank you", "thank", "ok", "okay",
    "cool", "great", "nice",
    "bye", "goodbye", "see you",
    "got it", "understood",
}

SMALLTALK_REPLIES = [
    "Sure. What case are you working on?",
    "No problem. Tell me the customer's question when you're ready.",
    "Happy to help. What does the customer need?",
    "Got it. Please type the customer's question.",
]

# ------------------------------------------------------
# Client init
# ------------------------------------------------------
def init_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    return genai.Client(api_key=api_key)

# ------------------------------------------------------
# Utility functions
# ------------------------------------------------------
def contains_khmer(text: str) -> bool:
    """Detect if the text contains Khmer characters (Unicode range 1780–17FF)."""
    if not text:
        return False
    return bool(re.search(r"[\u1780-\u17FF]", text))


def build_history_prefix(history: List[Tuple[str, str]], max_turns: int = 6) -> str:
    """
    Convert chat history [(user, bot), ...] into plain text for short-term memory.
    We only keep the last `max_turns` exchanges.
    """
    if not history:
        return ""

    recent = history[-max_turns:]
    parts = []
    for user_msg, bot_msg in recent:
        parts.append(f"Agent: {user_msg}")
        parts.append(f"Assistant: {bot_msg}")
    return "\n".join(parts)


def keyword_score(text: str, keywords: List[str]) -> int:
    """Rough score: how many banking keywords appear in the text."""
    t = text.lower()
    return sum(1 for kw in keywords if kw in t)


def is_followup(question: str, history: List[Tuple[str, str]]) -> bool:
    """
    Heuristic: decide if this message is a follow-up to the previous turn.
    We treat short fragments / corrections as follow-ups.
    """
    if not history:
        return False

    q = question.lower().strip().rstrip("?!.")
    words = q.split()

    # Very short messages are almost always follow-ups, e.g. "without signed also valid?"
    if len(words) <= 6:
        return True

    # Continuation markers
    markers = [
        "what about", "how about", "and for", "and then",
        "also", "too", "again", "valid", "without",
        "are you sure", "really",
    ]
    if any(m in q for m in markers):
        return True

    # Keyword overlap with previous user question
    last_user_msg = history[-1][0].lower()
    last_words = set(last_user_msg.split())
    if last_words & set(words):
        return True

    return False


def classify_intent(text: str, history: List[Tuple[str, str]]) -> str:
    """
    Classify a message into:
      - 'smalltalk'  -> we answer with a short fixed reply
      - 'banking'    -> we run the RAG pipeline
      - 'other'      -> treat same as banking but we know it's not clearly smalltalk

    This is fully rule-based to avoid extra LLM calls.
    """
    t = text.lower().strip()
    if not t:
        return "smalltalk"

    # Basic stats
    stripped = t.rstrip(".!?")
    words = stripped.split()
    word_count = len(words)
    has_question_mark = "?" in text

    # Detect WH words (what / how / when / …)
    wh_words = ["what", "how", "when", "where", "why", "which"]
    has_wh = any(w in stripped for w in wh_words)

    bank_score = keyword_score(stripped, BANK_KEYWORDS)
    problem_score = keyword_score(stripped, PROBLEM_KEYWORDS)

    smalltalk_score = 0
    for kw in SMALLTALK_KEYWORDS:
        if kw in stripped:
            smalltalk_score += 1

    # ---- Strong banking signals ----
    # If we see login / OTP / card etc., we treat as banking even without a '?'
    if bank_score > 0 or problem_score > 0:
        return "banking"

    # If message looks like a real question with a question mark or WH-word,
    # prefer banking unless it's clearly chit-chat.
    if has_question_mark or has_wh:
        # something like "how are you?" is chit-chat
        if "how are you" in stripped or "how's it going" in stripped:
            return "smalltalk"
        return "banking"

    # ---- Smalltalk / meta-intent detection ----

    # Short greeting / thanks / meta messages with smalltalk keywords
    if smalltalk_score > 0 and word_count <= 12:
        # Example: "hi", "okay thanks", "cool", "thanks a lot"
        # Example: "I have a question" (meta-intent)
        if ("question" in stripped and "have" in stripped) or "want to ask" in stripped:
            return "smalltalk"
        if "can i ask" in stripped or "can you answer" in stripped or "you can answer" in stripped:
            return "smalltalk"
        # plain greetings
        if word_count <= 6:
            return "smalltalk"

    # Very short messages without clear banking terms
    if word_count <= 4 and bank_score == 0:
        return "smalltalk"

    # Fallback: treat as banking/other so we don't ignore real problems
    return "banking"


def build_prompt(
    question: str,
    context_docs: List[Tuple[str, dict, float]],
    history_text: str = "",
) -> Tuple[str, str]:
    """
    Build system instruction + user prompt for Gemini.
    context_docs: list of (doc_text, metadata, distance)
    """
    context_strs = []
    for i, (doc, meta, dist) in enumerate(context_docs):
        src = meta.get("source", "unknown source")
        page = meta.get("page", "")
        if page:
            src += f" (page {page})"
        context_strs.append(f"[Doc {i+1} | {src} | dist={dist:.2f}]\n{doc}")

    context_block = (
        "\n\n---\n\n".join(context_strs) if context_strs else "No context found."
    )

    system_instruction = (
        "You are an internal assistant helping call-center agents at a bank.\n"
        "You must follow these rules:\n"
        "1) Use ONLY the provided context documents and short conversation history. Do not invent policies.\n"
        "2) If the context does not contain enough information, say you are not sure and suggest escalating or "
        "checking the official system.\n"
        "3) Always respond in this structure:\n"
        "   Customer answer: <short, simple explanation the agent will say to the customer in English>.\n"
        "   Internal notes: <detailed internal explanation referring to context, numbers, and conditions>.\n"
        "   Steps: <1-3 bullet points on what the agent should do>.\n"
        "4) Keep tone polite, clear, and professional. Do not mention embeddings, vectors, or retrieval.\n"
    )

    user_prompt = (
        f"{history_text}\n\n"
        f"Knowledge base context:\n{context_block}\n\n"
        f"Agent's question: {question}\n\n"
        "Now produce the structured answer."
    )

    return system_instruction, user_prompt

# ------------------------------------------------------
# Main RAG function
# ------------------------------------------------------
def answer_question(
    question: str,
    collection,
    embed_model,
    gemini_client,
    k: int = 5,
    history: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[str, str]:
    """
    Main RAG function for the chatbot.
    Returns (answer_text, top_context_snippet).
    """
    history = history or []
    raw_q = question or ""
    normalized_q = raw_q.lower().strip()

    # 1. Khmer detection
    if contains_khmer(raw_q):
        return (
            "I detected Khmer text. Right now I can only search the internal knowledge base with English questions.\n"
            "Please retype the customer's question in English (product name, action, amount, etc.).",
            "Khmer detected — RAG not used.",
        )

    # 2. Intent classification (smalltalk vs banking)
    intent = classify_intent(normalized_q, history)

    if intent == "smalltalk":
        # no retrieval, just a canned reply
        return (
            random.choice(SMALLTALK_REPLIES),
            "Smalltalk / intent — no context used.",
        )

    # 3. Follow-up detection for retrieval query
    followup_flag = is_followup(normalized_q, history)
    retrieval_question = raw_q
    if followup_flag and history:
        last_user_question = history[-1][0]
        retrieval_question = f"{last_user_question}\nFollow-up: {raw_q}"

    # 4. Retrieve context from Chroma
    context_docs, best_distance = retrieve_context(
        retrieval_question, collection, embed_model, k=k
    )

    # 5. Low-confidence fallback
    LOW_CONF_THRESHOLD = 1.2
    low_conf = (not context_docs) or (best_distance is None) or (
        best_distance > LOW_CONF_THRESHOLD
    )

    if low_conf:
        fallback_answer = (
            "Customer answer: I'm not fully sure based on the available information. "
            "Please inform the customer that you will double-check and get back to them.\n\n"
            "Internal notes: Retrieval confidence was low or there is no closely related article "
            "in the current knowledge base.\n\n"
            "Steps:\n"
            "1) Confirm the customer's product and key details.\n"
            "2) Check the official product policy or internal system manually.\n"
            "3) If still unclear, escalate to a supervisor. (Human handoff recommended.)"
        )
        top_context = context_docs[0][0] if context_docs else "No context retrieved."
        return fallback_answer, top_context

    # 6. Build prompt with recent history as short-term memory
    history_text = build_history_prefix(history)
    system_instruction, user_prompt = build_prompt(
        raw_q, context_docs, history_text=history_text
    )

    # 7. Generate answer with Gemini
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=user_prompt)],
            )
        ],
        config=genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )

    answer_text = response.text or "Sorry, I could not generate an answer."
    top_context = context_docs[0][0] if context_docs else "No context retrieved."
    return answer_text, top_context