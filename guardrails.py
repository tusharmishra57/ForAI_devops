"""
Guardrails for the Research-Paper RAG Assistant.

Five layers of protection:
  1. validate_question  — blocks empty / too-long / injection inputs
  2. check_scope        — blocks questions clearly outside the AI/ML/NLP domain
  3. has_sufficient_evidence — requires a lexical match before calling the LLM
  4. validate_answer    — replaces hallucinated or off-topic output with a refusal
  5. check_format       — enforces minimum length and rejects pure code dumps as answers
"""
import re
from dataclasses import dataclass

# ── Constants ────────────────────────────────────────────────────────────────
REFUSAL = (
    "I don't have enough information in the provided documents "
    "to answer this question."
)
MAX_QUESTION_CHARS = 800
MIN_ANSWER_TOKENS  = 3   # answers shorter than this are too terse
MAX_ANSWER_CHARS   = 4000

# Stop-words stripped before keyword matching
_STOP = {
    "a","an","and","are","about","can","did","do","does","explain",
    "for","how","in","is","of","on","or","the","to","what","which",
    "who","why","with","this","that","these","those","was","were",
    "has","have","had","its","it","be","been","being","would","could",
    "should","may","might","will","just","like","also","than","then",
}

# Patterns that indicate prompt-injection attempts
_INJECTION = (
    "ignore previous",
    "ignore all previous",
    "system prompt",
    "developer message",
    "jailbreak",
    "forget your instructions",
    "act as",
    "pretend you are",
    "disregard",
    "override instructions",
)

# Keywords that indicate a question is clearly off-topic (fast blocklist)
_OUT_OF_SCOPE_PHRASES = (
    "capital of", "who won", "gold price", "silver price", "stock price",
    "share price", "today price", "current price", "today match", "cricket",
    "football", "ipl", "world cup", "weather", "recipe", "sports score",
    "movie review", "celebrity", "cryptocurrency", "bitcoin", "horoscope",
    "election", "politics", "news today", "box office", "bollywood",
    "hollywood", "covid", "visa", "passport", "government", "tax",
)

# Allowlist — question MUST contain at least one of these to pass scope check.
# This is the primary defence: anything not related to AI/ML/code is rejected.
_ALLOWED_TOPICS = {
    # AI / ML concepts
    "model", "models", "neural", "network", "attention", "transformer",
    "bert", "gpt", "llm", "llms", "language", "embedding", "embeddings",
    "vector", "vectors", "retrieval", "rag", "generation", "inference",
    "training", "fine", "tuning", "lora", "quantization", "tokenizer",
    "token", "tokens", "classification", "regression", "encoder", "decoder",
    "layer", "layers", "weight", "weights", "gradient", "loss", "accuracy",
    "precision", "recall", "rouge", "bleu", "hallucination", "grounding",
    "context", "prompt", "prompting", "chain", "agent", "agents",
    "diffusion", "generative", "discriminative", "supervised", "unsupervised",
    "reinforcement", "reward", "policy", "value", "softmax", "activation",
    "relu", "normalization", "dropout", "batch", "epoch", "learning",
    "dataset", "benchmark", "evaluation", "metric", "metrics", "score",
    "similarity", "cosine", "dot", "product", "matrix", "tensor",
    "huggingface", "openai", "anthropic", "google", "deepmind", "llama",
    "qwen", "smollm", "tinyllama", "mistral", "falcon", "vicuna",
    "sentence", "chunk", "chunking", "index", "indexing", "faiss", "chroma",
    "chromadb", "semantic", "search", "document", "documents", "paper",
    "papers", "research", "architecture", "pipeline", "workflow",
    # Code / codebase
    "code", "codebase", "function", "class", "method", "module", "file",
    "python", "docker", "container", "api", "endpoint", "parameter",
    "argument", "variable", "import", "library", "package", "install",
    "deploy", "deployment", "server", "streamlit", "app", "application",
    "rag_logic", "app.py", "coordinator", "manager", "embeddings",
}


# ── Data class ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    reason:  str = ""
    message: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────
def _keywords(text: str) -> set[str]:
    """Extract meaningful tokens (≥3 chars, not stop-words)."""
    return {
        w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", text.lower())
        if w not in _STOP
    }


# ── Guardrail 1 — Input validation ───────────────────────────────────────────
def validate_question(question: str) -> GuardrailDecision:
    """Block empty, too-long, and prompt-injection inputs."""
    cleaned = question.strip()

    if not cleaned:
        return GuardrailDecision(
            False, "empty_input",
            "Please enter a question about the uploaded documents."
        )

    if len(cleaned) > MAX_QUESTION_CHARS:
        return GuardrailDecision(
            False, "input_too_long",
            f"Please keep questions under {MAX_QUESTION_CHARS} characters "
            f"(yours: {len(cleaned)})."
        )

    lc = cleaned.lower()
    for pattern in _INJECTION:
        if pattern in lc:
            return GuardrailDecision(
                False, "prompt_injection",
                "I can only answer research-paper questions from the uploaded documents."
            )

    return GuardrailDecision(True)


# ── Guardrail 2 — Scope check ─────────────────────────────────────────────────
def check_scope(question: str) -> GuardrailDecision:
    """
    Two-stage scope enforcement:
      Stage 1 — Fast blocklist: instantly reject known off-topic phrases.
      Stage 2 — Allowlist:      question must contain at least one AI/ML/code
                                keyword; anything else is rejected.
    This ensures questions like 'today gold price' or 'India match score'
    are blocked even if they are not in the blocklist.
    """
    _SCOPE_REFUSAL = (
        "This assistant only answers questions about AI/ML research papers "
        "and the uploaded codebase. Please ask about topics like transformers, "
        "RAG, embeddings, model evaluation, or the project code."
    )
    lc = question.lower()

    # Stage 1: blocklist
    for phrase in _OUT_OF_SCOPE_PHRASES:
        if phrase in lc:
            return GuardrailDecision(False, "out_of_scope", _SCOPE_REFUSAL)

    # Stage 2: allowlist — extract words and check overlap with allowed topics
    words = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{1,}", lc))
    if not words & _ALLOWED_TOPICS:
        return GuardrailDecision(False, "out_of_scope", _SCOPE_REFUSAL)

    return GuardrailDecision(True)



# ── Guardrail 3 — Evidence sufficiency ───────────────────────────────────────
def has_sufficient_evidence(question: str, context: str) -> bool:
    """
    Require at least 1 shared keyword between the question and the
    retrieved context before passing the query to the LLM.
    """
    q_kw = _keywords(question)
    c_kw = _keywords(context)
    if not q_kw or not context.strip():
        return False
    return len(q_kw & c_kw) >= 1


# ── Guardrail 4 — Answer grounding ───────────────────────────────────────────
def validate_answer(answer: str, context: str) -> GuardrailDecision:
    """
    Accept the answer only if:
      - it IS the standard refusal phrase, OR
      - it shares ≥ 2 substantive keywords with the retrieved context.
    Otherwise replace with REFUSAL.
    """
    if REFUSAL.lower() in answer.lower():
        return GuardrailDecision(True, "accepted_refusal")

    a_kw = _keywords(answer)
    c_kw = _keywords(context)
    overlap = len(a_kw & c_kw)

    if len(a_kw) < MIN_ANSWER_TOKENS or overlap < 2:
        return GuardrailDecision(
            False, "unsupported_output", REFUSAL
        )

    return GuardrailDecision(True, "grounded_answer")


# ── Guardrail 5 — Format check ────────────────────────────────────────────────
def check_format(answer: str) -> GuardrailDecision:
    """
    Reject answers that are excessively long (likely model run-away) or
    consist almost entirely of raw code (not a prose answer).
    """
    if len(answer) > MAX_ANSWER_CHARS:
        return GuardrailDecision(
            False, "output_too_long",
            REFUSAL
        )

    _CODE_PREFIXES = ("def ", "class ", "import ", "from ", "return ",
                      ">>>", "...", "```", "#")
    code_lines = sum(
        1 for line in answer.splitlines()
        if line.strip() and (
            any(line.strip().startswith(p) for p in _CODE_PREFIXES)
            or (line.startswith(("    ", "\t")) and line.strip())
        )
    )
    total_lines = max(len(answer.splitlines()), 1)
    if code_lines / total_lines > 0.60:
        return GuardrailDecision(
            False, "code_only_output",
            "The answer should be a prose explanation, not raw code."
        )

    return GuardrailDecision(True, "format_ok")


# ── Convenience: run all output checks ───────────────────────────────────────
def run_output_tests(answer: str, context: str) -> dict:
    """
    Return a dict of pass/fail verdicts for every output-testing criterion.
    Suitable for displaying in the Streamlit 'Guardrails & Output Tests' tab.
    """
    grounding = validate_answer(answer, context)
    fmt       = check_format(answer)
    return {
        "is_refusal":          REFUSAL.lower() in answer.lower(),
        "grounding_pass":      grounding.allowed,
        "grounding_reason":    grounding.reason,
        "format_pass":         fmt.allowed,
        "format_reason":       fmt.reason,
        "answer_length":       len(answer),
        "keyword_overlap":     len(_keywords(answer) & _keywords(context)),
        "overall_pass":        grounding.allowed and fmt.allowed,
    }
