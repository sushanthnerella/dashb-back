# llm_agent.py

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = str(Path(__file__).resolve().parents[1] / "models" / "deepseek-1.5b")

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
