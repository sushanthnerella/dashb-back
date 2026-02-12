import warnings

warnings.filterwarnings(
    "ignore",
    message="`resume_download` is deprecated",
    category=FutureWarning,
)

from agents.retrival_agent1 import get_patient_summary
from agents.knowledge_retriever import search_knowledge
from agents.llm_agent import run_summary_llm

summary = get_patient_summary("65c4a2b3c9e5f7a1b2c3d4e5")
guidance = search_knowledge("Interpret vision drop and stable IOP in adult patient")

result = run_summary_llm(summary, guidance)
print(result)
