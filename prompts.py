JUDGE_SCHEMA = """
Return **only** JSON with this schema:
{
  "scores": {
    "accuracy": 0-100,
    "fluency": 0-100,
    "style": 0-100,
    "terminology": 0-100
  },
  "errors": [
    {
      "category": "mistranslation|omission|addition|grammar|register|punctuation|spelling|terminology|cohesion",
      "source_excerpt": "string",
      "translation_excerpt": "string",
      "note": "brief explanation (<=30 words)"
    }
  ],
  "explanations": {
    "accuracy_notes": "2-4 bullet points",
    "fluency_notes": "2-4 bullet points",
    "style_notes": "2-4 bullet points",
    "terminology_coverage": "mention which key terms were preserved/altered"
  }
}
"""

JUDGE_PROMPT = f"""
You will assess a translation against its source text with an emphasis on professional translation criteria.
Assess on:
- **Accuracy** (faithfulness to meaning; preservation of nuances; no omissions/additions).
- **Fluency** (grammar, syntax, readability, naturalness).
- **Style** (register/voice consistency with source; idiomaticity; cohesion).
- **Terminology** (use and consistency of key terms provided).

Follow this process:
1) Carefully read the SOURCE and TRANSLATION.
2) Check each key term; note if translated consistently and appropriately.
3) Identify **concrete errors**; quote short evidence.

{JUDGE_SCHEMA}

SOURCE:
{{source_text}}

TRANSLATION:
{{translation_text}}

KEY_TERMS (from source; may be empty):
{{key_terms}}
"""

TERM_PROMPT = """
Extract up to {max_terms} domain-relevant, translation-critical key terms or multiword expressions from the SOURCE text.
Return **only a JSON list of strings** (no extra text). Merge inflectional variants.
SOURCE:
{source_text}
"""
