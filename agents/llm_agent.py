# llm_agent.py

import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Use a publicly available model
# distilgpt2: lightweight, fast, no authentication needed
# Alternative: "google/flan-t5-small", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_PATH = "distilgpt2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()
model.config.use_cache = True


_PLANNER_STOPWORDS = {
    "the",
    "and",
    "or",
    "a",
    "an",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "as",
    "by",
    "at",
    "from",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "about",
    "how",
    "what",
    "why",
    "when",
    "where",
    "which",
    "who",
}

_INTENT_PATTERNS = {
    "definition": ["what is", "what are", "define", "definition", "meaning"],
    "procedure": ["how to", "steps", "procedure", "method", "process", "implement"],
    "compare": ["compare", "difference", "versus", "vs", "contrast"],
    "formula": ["formula", "equation", "calculate", "compute", "derive"],
    "troubleshooting": ["error", "issue", "problem", "fails", "failure", "not working", "fix"],
}

_TERM_EXPANSIONS = {
    "definition": ["concept", "description", "terminology"],
    "procedure": ["workflow", "protocol", "stepwise"],
    "compare": ["contrast", "distinguish", "tradeoffs"],
    "formula": ["equation", "variables", "derivation"],
    "troubleshooting": ["root cause", "diagnosis", "resolution"],
    "glaucoma": ["optic neuropathy", "visual field loss", "ocular hypertension"],
    "iop": ["intraocular pressure", "eye pressure", "tonometry"],
    "vision": ["visual acuity", "sight", "visual function"],
    "diagnosis": ["assessment", "finding", "clinical impression"],
    "treatment": ["therapy", "management", "intervention"],
}


def _word_limited(text: str, max_words: int = 18) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def _classify_planner_intent(query: str) -> str:
    query_l = (query or "").lower().strip()
    scores = {key: 0 for key in _INTENT_PATTERNS}

    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in query_l:
                scores[intent] += 2

    if query_l.startswith("how"):
        scores["procedure"] += 1
    elif query_l.startswith("what"):
        scores["definition"] += 1
    elif query_l.startswith("why"):
        scores["troubleshooting"] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "explain"


def _extract_core_terms(query: str) -> list[str]:
    tokens = re.split(r"\W+", (query or "").lower())
    ordered = []
    seen = set()
    for token in tokens:
        if len(token) < 3 or token in _PLANNER_STOPWORDS:
            continue
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def _make_keyword_expansion(intent: str, core_terms: list[str]) -> list[str]:
    expanded = []
    seen = set()

    def add(term: str) -> None:
        if term and term not in seen:
            seen.add(term)
            expanded.append(term)

    add(intent)
    for term in core_terms:
        add(term)
        for syn in _TERM_EXPANSIONS.get(term, []):
            add(syn)

    for syn in _TERM_EXPANSIONS.get(intent, []):
        add(syn)

    while len(expanded) < 5:
        add("key terms")
        add("core concept")
        add("textbook guidance")

    return expanded[:12]


def plan_semantic_search_queries(user_query: str) -> dict:
    """Return strict semantic search planning JSON schema as a Python dict."""
    query = (user_query or "").strip()
    intent = _classify_planner_intent(query)
    core_terms = _extract_core_terms(query)
    core_head = " ".join(core_terms[:4]).strip() or query
    keyword_expansion = _make_keyword_expansion(intent, core_terms)
    ambiguous = len(core_terms) <= 2 or len(query.split()) <= 4

    if intent == "definition":
        direct = f"{query} exact definition key terms textbook"
        support = f"{core_head} characteristics context examples limitations"
    elif intent == "procedure":
        direct = f"{query} stepwise procedure textbook guidance"
        support = f"{core_head} prerequisites sequence constraints common errors"
    elif intent == "compare":
        direct = f"{query} differences similarities comparison criteria"
        support = f"{core_head} indications limitations tradeoffs examples"
    elif intent == "formula":
        direct = f"{query} formula equation variable definitions"
        support = f"{core_head} assumptions derivation units worked example"
    elif intent == "troubleshooting":
        direct = f"{query} root causes diagnostic checks fixes"
        support = f"{core_head} failure patterns risk factors corrective actions"
    else:
        direct = f"{query} concise explanation key concepts"
        support = f"{core_head} mechanism conditions exceptions examples"

    if ambiguous:
        disambig_seed = " ".join(keyword_expansion[:8])
        disambig = f"{core_head} synonyms related terms alternate terminology {disambig_seed}"
    else:
        disambig_seed = " ".join(keyword_expansion[:5])
        disambig = f"{core_head} synonyms alternate terminology domain-specific terms {disambig_seed}"

    return {
        "intent_classification": intent,
        "semantic_subqueries": {
            "direct_answer_query": _word_limited(" ".join(direct.split()), 18),
            "support_query": _word_limited(" ".join(support.split()), 18),
            "disambiguation_query": _word_limited(" ".join(disambig.split()), 18),
        },
        "keyword_expansion": keyword_expansion,
        "retrieval_params": {"top_k_per_query": 25, "final_k": 5},
    }

def run_summary_llm(patient_summary, knowledge_chunks, max_tokens=200):
    if not isinstance(patient_summary, dict):
        return "Error: patient summary is not a dict"

    if patient_summary.get("error"):
        return f"Error: {patient_summary.get('error')}"

    if "visits" not in patient_summary:
        return "Error: patient summary missing visits"

    context = "\n".join(knowledge_chunks)
    visits_text = "\n".join([
        f"- On {v['visitDate']}: IOP: {v.get('iop',{})}, Vision: {v.get('vision',{})}, Diagnosis: {v.get('diagnosis','')}, Notes: {v.get('notes','')}"
        for v in patient_summary["visits"]
    ])

    prompt = f"""
### Instruction
You are an ophthalmology AI assistant for doctors.
Write a concise, clinically useful one-paragraph summary based only on the visit history and the medical guidance excerpts.

### Patient
Name: {patient_summary['name']}
Age: {patient_summary['age']}
Sex: {patient_summary['sex']}

### Visit History
{visits_text}

### Medical Guidance (excerpts)
{context}

### One-paragraph summary
""".strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.12,
            no_repeat_ngram_size=5,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return response.strip()
