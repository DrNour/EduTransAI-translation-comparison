import os
import math
import time
from typing import List, Dict, Any

import numpy as np

from prompts import JUDGE_PROMPT, TERM_PROMPT

# --- OpenAI client ---
def _get_openai_client():
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Please install openai>=1.0.0 (pip install openai)") from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is not set.")
    return OpenAI(api_key=api_key)

def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> List[np.ndarray]:
    client = _get_openai_client()
    # Split into batches to avoid limits
    out = []
    B = 96
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=model, input=batch)
        for d in resp.data:
            out.append(np.array(d.embedding, dtype=np.float32))
    return out

def semantic_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    # cosine similarity
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)

def normalize_score(cos_sim: float) -> float:
    # map cosine [-1,1] -> [0,100]
    return round((cos_sim + 1.0) * 50.0, 2)

def _chunk_text(text: str, limit: int) -> List[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + limit
        chunks.append(text[start:end])
        start = end
    return chunks

def ai_extract_terms(source_text: str, max_terms: int = 10, model: str = "gpt-4o-mini", temperature: float = 0.2) -> List[str]:
    client = _get_openai_client()
    sys = "You are a terminologist extracting domain-relevant, translation-critical terms or multiword expressions from a source text. Return a concise JSON list of unique terms (strings) without extra commentary."
    user = TERM_PROMPT.format(max_terms=max_terms, source_text=source_text)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    content = resp.choices[0].message.content
    # Attempt to parse JSON list; fallback to simple split
    import json, re
    try:
        terms = json.loads(content)
        if isinstance(terms, list):
            return [str(t).strip() for t in terms if str(t).strip()][:max_terms]
    except Exception:
        pass
    # Fallback: extract between [ ... ]
    import re
    m = re.search(r"\[.*\]", content, flags=re.S)
    if m:
        try:
            terms = json.loads(m.group(0))
            if isinstance(terms, list):
                return [str(t).strip() for t in terms if str(t).strip()][:max_terms]
        except Exception:
            return []
    return []

def ai_judge_translation(source_text: str, translation_text: str, key_terms: List[str], model: str = "gpt-4o-mini", temperature: float = 0.2, chunk_limit: int = 2200) -> Dict[str, Any]:
    client = _get_openai_client()

    # If texts are long, evaluate chunk-by-chunk and average
    src_chunks = _chunk_text(source_text, chunk_limit)
    tr_chunks = _chunk_text(translation_text, chunk_limit)

    pairs = list(zip(src_chunks, tr_chunks))
    if not pairs:
        pairs = [("", translation_text)]

    agg = {"accuracy": [], "fluency": [], "style": [], "terminology": []}
    collected_errors = []
    explanations = {
        "accuracy_notes": "",
        "fluency_notes": "",
        "style_notes": "",
        "terminology_coverage": ""
    }

    for i, (src, tr) in enumerate(pairs):
        sys = "You are a meticulous bilingual translation assessor. Be concise, objective, and return **valid JSON** according to the schema. Use 0–100 integers for scores."
        user = JUDGE_PROMPT.format(
            source_text=src,
            translation_text=tr,
            key_terms=json.dumps(key_terms, ensure_ascii=False)
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        content = resp.choices[0].message.content

        import json
        try:
            data = json.loads(content)
        except Exception:
            # Try to extract JSON block
            import re
            m = re.search(r"\{.*\}", content, flags=re.S)
            if not m:
                raise RuntimeError("Judge returned non-JSON content.")
            data = json.loads(m.group(0))

        # Aggregate
        s = data.get("scores", {})
        for k in agg.keys():
            if k in s:
                agg[k].append(int(s[k]))

        errs = data.get("errors", [])
        if isinstance(errs, list):
            collected_errors.extend(errs)

        exp = data.get("explanations", {})
        for k in explanations.keys():
            if k in exp and isinstance(exp[k], str):
                explanations[k] += ("\n" if explanations[k] else "") + exp[k].strip()

    # Average scores
    final_scores = {k: int(round(sum(v)/len(v))) if v else 0 for k, v in agg.items()}

    return {
        "scores": final_scores,
        "errors": collected_errors,
        "explanations": explanations,
        "key_terms": key_terms,
    }
