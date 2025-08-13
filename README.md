# EduTransAI – AI‑Powered Translation Comparison App

A Streamlit web app to compare **two or more translations** of the same source text. It produces **AI‑assessed scores** and detailed feedback for **Accuracy, Fluency, Style, Terminology, and Error Categories**, plus similarity/embedding metrics and visualizations.

## ✨ Features
- Paste a source text and **2–10 translations** or upload a CSV.
- **AI judge** (OpenAI) produces:
  - Scores (0–100) for **Accuracy, Fluency, Style**.
  - **Terminology** check (key terms coverage, consistency).
  - **Error categorization** (mistranslation, omission, addition, grammar, register, punctuation).
  - **Evidence with quotes** and line-level notes.
- **Embeddings-based similarity** to source (semantic cosine similarity).
- **Side-by-side comparison table** and **radar chart**.
- **Export** results to CSV/JSON.

## 🚀 Quickstart
1. **Python 3.10+** recommended.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="YOUR_KEY_HERE"    # macOS/Linux
   setx OPENAI_API_KEY "YOUR_KEY_HERE"      # Windows PowerShell
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📦 CSV Format
You can upload a CSV with columns:
- `source` – the source text (same for all rows) **or** leave blank and paste in the UI.
- `translation_id` – any label for the translation (e.g., T1, Human, ChatGPT).
- `translation_text` – the translation.

See `sample_data/sample_translations.csv`.

## 🧠 Models
- Chat & judging: uses `gpt-4o-mini` by default (configurable).
- Embeddings: `text-embedding-3-small` (configurable).

Set via environment variables:
```
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```

## 📤 Export
- Download a CSV or JSON with per-translation scores and AI feedback.

## 🛡️ Notes
- Keep texts under ~4–5k tokens per evaluation for speed/cost; longer texts are chunked.
- No user data is stored server-side; everything runs locally + OpenAI API calls.

## 🧰 Dev Scripts
Format & lint (optional):
```bash
pip install ruff black
ruff check .
black .
```

## 🗂️ Project Structure
```
.
├── app.py
├── metrics.py
├── prompts.py
├── requirements.txt
├── sample_data/
│   └── sample_translations.csv
└── README.md
```

## 📚 License
MIT
