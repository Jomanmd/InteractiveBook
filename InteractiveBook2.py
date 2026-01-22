# app.py
import os
import json
import time
import datetime
import base64
import re
from typing import List, Dict, Any, Optional

import numpy as np
import requests
import streamlit as st
from pypdf import PdfReader

# LangChain & Vector Store
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import os
import streamlit as st

# Local dev: .env may set os.environ
# Streamlit Cloud: st.secrets should provide the key
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None) or os.getenv("GROQ_API_KEY")

# =================== BACKGROUND IMAGE ===================

def set_bg_image(image_path: str):
    if not os.path.exists(image_path):
        return  # don't crash

    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/png;base64,{b64}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# =================== Retrieval guidance ===================

MIN_K_BY_MODE = {
    "Vector": 10,   # important: higher recall for figures / specific refs
    "Fusion": 6,
    "BM25": 6,
}

SEARCH_MODE_LABELS = {
    "Fusion": "Balanced (Hybrid Search)",
    "Vector": "Meaning Search (Semantic)",
    "BM25": "Keyword Search (Exact terms)",
}

SEARCH_MODE_HELP = {
    "Fusion": "Combines semantic + keyword search (RRF). Best overall accuracy.",
    "Vector": "Semantic search using embeddings. Best for concepts/definitions/synonyms.",
    "BM25": "Keyword-based retrieval. Best for exact terms, acronyms, and quotes.",
}


def enforce_min_k(search_mode: str, k: int) -> int:
    return max(int(k), MIN_K_BY_MODE.get(search_mode, 6))

def retrieval_hint(search_mode: str) -> str:
    mk = MIN_K_BY_MODE.get(search_mode, 6)
    if search_mode == "Vector":
        return f"Meaning Search (Semantic) works best with **k ≥ {mk}** (recommended for figures/definitions)."
    if search_mode == "Fusion":
        return f"Balanced (Hybrid Search) is stable with **k ≥ {mk}** (balanced accuracy + speed)."
    return f"Keyword Search (Exact terms) is stable with **k ≥ {mk}** (keyword matching)."


# =================== 1) SETUP & CONFIG ===================


GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

PDF_PATH = "intelligent_interactive_systems.pdf"
VECTOR_STORE_PATH = "./vector_stores/course_book_store"

# Passing / locking rules (from your friend's structure)
PASSING_THRESHOLD = 2          # pass if >= 2 correct out of 3
FINAL_PASSING_THRESHOLD = 4    # pass final if >= 4 correct out of 5

# =================== 2) UI STYLING (friend + your readability) ===================

APP_CSS = """
<style>
/* ====== Background overlay (keep bg visible + readable) ====== */
[data-testid="stAppViewContainer"]{
  position: relative;
}
[data-testid="stAppViewContainer"]::before{
  content:"";
  position: fixed;
  inset: 0;
  background: rgba(255,255,255,0.1);
  backdrop-filter: blur(2px);
  z-index: 0;
  pointer-events: none;
}
[data-testid="stAppViewContainer"] > .main,
.block-container{
  position: relative;
  z-index: 1;
}

/* ====== Main typography ====== */
h1, h2, h3, h4, h5, h6 { color:#0f172a !important; }
p, li, span, div { color:#0f172a; }
.stCaption, small { color:#475569 !important; }

/* ====== Sidebar (MAKE IT LOOK GOOD) ====== */
section[data-testid="stSidebar"]{
  background: rgba(15,23,42,0.92) !important;
  border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] *{
  color: rgba(255,255,255,0.92) !important;
}
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small{
  color: rgba(255,255,255,0.65) !important;
}

/* Sidebar spacing */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
  gap: 0.65rem;
}

/* Sidebar radio pills */
section[data-testid="stSidebar"] div[role="radiogroup"] label{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 12px;
  padding: 10px 12px;
  margin: 6px 0;
  transition: 0.15s ease;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
  background: rgba(255,255,255,0.10);
  border-color: rgba(255,255,255,0.14);
}

/* Make the selected radio look highlighted */
section[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div{
  font-weight: 800;
}

/* ====== Buttons (FIX missing text + make consistent) ====== */
.stButton > button{
  background: #0f172a !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 12px !important;
  padding: 0.55rem 1.1rem !important;
  font-weight: 800 !important;
  box-shadow: 0 8px 18px rgba(15,23,42,0.18) !important;
}
.stButton > button *{
  color: #ffffff !important; /* IMPORTANT: keeps button label visible */
}
.stButton > button:hover{
  background: #1f2937 !important;
  transform: translateY(-1px);
}

/* Make small buttons (like "Start") look better */
.stButton > button[kind="secondary"]{
  background: #0f172a !important;
}

/* ====== Input boxes (keep your dark theme but readable) ====== */
.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] > div{
  background: rgba(15,23,42,0.92) !important;
  color: #ffffff !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}
.stTextInput label, .stTextArea label, .stSelectbox label, .stSlider label{
  color:#0f172a !important;
  font-weight: 800 !important;
}

/* ====== Cards / sections look nicer ====== */
.stDivider { opacity: 0.25; }

/* ===== FIX: button text missing in forms / submit buttons ===== */
button[kind], button[kind] * {
  color: #ffffff !important;
  fill: #ffffff !important;
}

/* Streamlit sometimes uses <p> inside buttons */
.stButton button p,
.stButton button span,
.stButton button div,
.stForm button p,
.stForm button span,
.stForm button div {
  color: #ffffff !important;
}

/* Make ALL primary buttons consistent */
button[kind="primary"]{
  background: #0f172a !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 12px !important;
  font-weight: 800 !important;
}
button[kind="primary"]:hover{
  background: #1f2937 !important;
}

/* Make secondary buttons also readable */
button[kind="secondary"]{
  background: rgba(15,23,42,0.92) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 12px !important;
}
button[kind="secondary"]:hover{
  background: rgba(31,41,55,0.95) !important;
}
/* ===== Selectbox dropdown (BaseWeb) FIX: dark menu + readable text ===== */

/* The selected value area */
div[data-baseweb="select"] > div{
  background: rgba(15,23,42,0.92) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 12px !important;
}
div[data-baseweb="select"] *{
  color: #ffffff !important;
}

/* The dropdown menu panel */
div[data-baseweb="popover"]{
  z-index: 9999 !important;
}
div[data-baseweb="menu"]{
  background: rgba(15,23,42,0.98) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
}

/* Each option */
div[data-baseweb="option"]{
  background: transparent !important;
  color: #ffffff !important;
}
div[data-baseweb="option"]:hover{
  background: rgba(255,255,255,0.10) !important;
}

/* Some Streamlit versions wrap option text in spans/divs */
div[data-baseweb="option"] *{
  color: #ffffff !important;
}

/* ===== HARD FIX for selectbox dropdown text (portal menu) ===== */

/* The popover menu background */
body div[data-baseweb="popover"] div[data-baseweb="menu"]{
  background: rgba(15,23,42,0.98) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
}

/* Force ALL text inside dropdown to be white */
body div[data-baseweb="popover"] *{
  color: #ffffff !important;
}

/* Option hover */
body div[data-baseweb="popover"] div[data-baseweb="option"]:hover{
  background: rgba(255,255,255,0.10) !important;
}

/* Selected value area should be white too */
div[data-baseweb="select"] *{
  color: #ffffff !important;
}

/* ===== Fix Expander (Sources) text being black on dark header ===== */
div[data-testid="stExpander"] details summary{
  background: rgba(255,255,255,0.85) !important;  /* make header light */
  border: 1px solid rgba(15,23,42,0.10) !important;
  border-radius: 12px !important;
  padding: 10px 12px !important;
}

/* Header text */
div[data-testid="stExpander"] details summary *{
  color: #0f172a !important;
}

/* Expanded content text */
div[data-testid="stExpander"] div[role="region"] *{
  color: #0f172a !important;
}

/* ===== Certificate styling ===== */
.certificate-container{
  max-width: 980px;
  margin: 24px auto;
  padding: 38px 44px;
  background: rgba(255,255,255,0.92);
  border-radius: 18px;
  border: 1px solid rgba(15,23,42,0.10);
  box-shadow: 0 18px 40px rgba(15,23,42,0.18);
  position: relative;
  overflow: hidden;
}

.certificate-container::before{
  content:"";
  position:absolute;
  inset:-40px;
  background: radial-gradient(circle at 80% 30%, rgba(15,23,42,0.06), transparent 55%);
  pointer-events:none;
}

.seal{
  position:absolute;
  top: 28px;
  right: 28px;
  width: 110px;
  height: 110px;
  border-radius: 999px;
  background: rgba(15,23,42,0.92);
  color: #fff;
  display:flex;
  align-items:center;
  justify-content:center;
  text-align:center;
  font-weight: 900;
  letter-spacing: 1px;
  font-size: 12px;
  line-height: 1.2;
  box-shadow: 0 10px 22px rgba(15,23,42,0.25);
}

.cert-header{
  font-size: 34px;
  font-weight: 900;
  color: #0f172a;
  margin-bottom: 6px;
}

.cert-sub{
  font-size: 16px;
  color:#334155;
  margin-bottom: 16px;
}

.cert-name{
  font-size: 40px;
  font-weight: 900;
  color:#0f172a;
  margin: 10px 0 18px 0;
}

.cert-body{
  font-size: 16px;
  color:#0f172a;
  line-height: 1.6;
  margin-bottom: 18px;
}

.cert-grade{
  font-size: 18px;
  font-weight: 800;
  color:#0f172a;
  margin: 10px 0 18px 0;
}

.cert-footer{
  display:flex;
  justify-content: space-between;
  gap: 18px;
  margin-top: 18px;
}

.signature{
  flex: 1;
  border-top: 1px solid rgba(15,23,42,0.18);
  padding-top: 10px;
  color:#0f172a;
  font-weight: 700;
}

</style>
"""



def clear_dynamic_session_state(prefixes):
    keys_to_delete = [
        k for k in st.session_state.keys()
        if any(k.startswith(p) for p in prefixes)
    ]
    for k in keys_to_delete:
        del st.session_state[k]

# =================== OPTIONAL: BACKGROUND IMAGE (your code) ===================

def set_bg_image(image_path: str):
    if not image_path or not os.path.exists(image_path):
        return

    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: url("data:image/png;base64,{b64}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# =================== 3) LLM (friend retry + your env usage) ===================

def call_llm(prompt: str, temperature: float = 0.2, timeout: int = 45) -> str:
    """Groq caller with auto-retry on 429 (rate limit)."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "❌ Error: GROQ_API_KEY not set in environment (.env or system env)."

    max_retries = 3
    base_delay = 2

    for attempt in range(max_retries):
        try:
            r = requests.post(
                GROQ_ENDPOINT,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
                timeout=timeout,
            )

            if r.status_code == 429:
                wait_time = base_delay * (2 ** attempt)
                time.sleep(wait_time)
                continue

            if r.status_code != 200:
                return f"❌ API Error {r.status_code}: {r.text}"

            data = r.json()
            if "error" in data:
                return f"❌ API Error: {data['error'].get('message', 'Unknown error')}"

            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"❌ Connection Error: {str(e)}"

    return "❌ Error: Failed after multiple retries (Rate Limit)."

# =================== 4) PERSONA (friend structure) ===================

def get_persona_instruction() -> str:
    p = st.session_state.user_profile
    base = f"You are an AI Tutor customizing content for {p['name']}."
    style = "Use simple analogies and plain language." if p['level'] == "Beginner" else "Use technical, academic language."
    return f"{base} {style} The user is a {p['role']} with goal: '{p['goal']}'."

def build_level_instruction(level: str) -> str:
    if level == "Beginner":
        return (
            "You are teaching a BEGINNER student.\n"
            "- Use simple language.\n"
            "- Explain terms briefly when first used.\n"
            "- Avoid heavy jargon.\n"
        )
    return (
        "You are teaching an ADVANCED student.\n"
        "- Use academic terminology.\n"
        "- Emphasize relationships, trade-offs, design implications.\n"
        "- Keep it structured and precise.\n"
    )

# =================== 5) YOUR CURRICULUM BUILDER (chapter titles from TOC) ===================

def extract_chapter_titles_from_contents(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract chapter titles 1..10 from Table of Contents.
    Returns: [{"chapter_number": 1, "title": "..."}]
    """
    if not os.path.exists(pdf_path):
        return []

    reader = PdfReader(pdf_path)

    toc_start = None
    for i in range(min(30, len(reader.pages))):
        t = (reader.pages[i].extract_text() or "")
        if "Contents" in t:
            toc_start = i
            break
    if toc_start is None:
        return []

    toc_text = ""
    for i in range(toc_start, min(toc_start + 6, len(reader.pages))):
        toc_text += (reader.pages[i].extract_text() or "") + "\n"

    lines = [ln.strip() for ln in toc_text.splitlines() if ln.strip()]

    chap_start = re.compile(r"^(10|[1-9])\s+(?![.\d])(.+)$")
    author_page = re.compile(r"\.{5,}\s*\d+\s*$")

    chapters: Dict[int, str] = {}
    current_num: Optional[int] = None
    current_title_parts: List[str] = []

    for ln in lines:
        m = chap_start.match(ln)
        if m:
            if current_num is not None and current_title_parts and current_num not in chapters:
                chapters[current_num] = " ".join(current_title_parts).strip()

            current_num = int(m.group(1))
            current_title_parts = [m.group(2).strip()]
            continue

        if current_num is not None:
            if author_page.search(ln):
                title = " ".join(current_title_parts)
                title = " ".join(title.split()).strip(" ,.-")

                # Small cleanup for chapter 10 if TOC contains extra trailing parts
                if current_num == 10 and title.lower().startswith("pns:"):
                    low = title.lower()
                    cut = low.find("on the web")
                    if cut != -1:
                        title = title[:cut + len("on the web")].strip(" ,.-")

                if 1 <= current_num <= 10:
                    chapters[current_num] = title

                current_num = None
                current_title_parts = []
            else:
                if not re.match(r"^\d+\.\d+", ln):  # ignore subsections
                    current_title_parts.append(ln)

    if current_num is not None and current_title_parts and current_num not in chapters:
        chapters[current_num] = " ".join(current_title_parts).strip()

    result = []
    for n in range(1, 11):
        if n in chapters:
            result.append({"chapter_number": n, "title": chapters[n]})
    return result

def build_chapter_based_course_plan(pdf_path: str) -> dict:
    chapters = extract_chapter_titles_from_contents(pdf_path)
    if len(chapters) < 10:
        chapters = [{"chapter_number": i, "title": f"Chapter {i}"} for i in range(1, 11)]

    lessons = []
    for c in chapters:
        n = c["chapter_number"]
        title = c["title"]
        lessons.append({
            "lesson_id": n,
            "chapter_number": n,
            "title": title,
            "goal": f"Understand the full content of Chapter {n}: {title}.",
            "search_query": f"Chapter {n} {title}",  # strong retrieval hook
            "difficulty": "Beginner" if n <= 3 else ("Intermediate" if n <= 7 else "Advanced"),
        })

    return {
        "course_title": "Intelligent Interactive Systems (Chapter-Based Course)",
        "lessons": lessons
    }

# =================== 6) RAG HELPERS (your sources + formatting) ===================

def format_context_for_prompt(context_docs: List[Document]) -> str:
    chunks = []
    for i, d in enumerate(context_docs, start=1):
        page0 = d.metadata.get("page", None)
        page = (page0 + 1) if isinstance(page0, int) else "unknown"
        chunks.append(f"[Chunk {i} | page {page}]\n{d.page_content}")
    return "\n\n".join(chunks)

def build_sources(context_docs: List[Document], max_sources: int = 4) -> List[Dict[str, Any]]:
    sources = []
    for i, d in enumerate(context_docs[:max_sources], start=1):
        page0 = d.metadata.get("page", None)
        page = (page0 + 1) if isinstance(page0, int) else "unknown"
        snippet = (d.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 520:
            snippet = snippet[:520] + "…"
        sources.append({"label": f"Chunk {i} — page {page}", "page": page, "snippet": snippet})
    return sources

def is_contents_chunk_text(text: str) -> bool:
    t = (text or "").lower()
    if "contents" in t[:200]:
        return True
    if "........" in t:
        return True
    return False

# =================== 7) QUIZ / LESSON GENERATION (your strict-json + sources) ===================

def generate_lesson_markdown(lesson_title: str, context_text: str, user_level: str) -> str:
    level_instruction = build_level_instruction(user_level)
    persona = get_persona_instruction()
    prompt = f"""
{persona}
{level_instruction}
You are writing a lesson page for a course app.

Use ONLY the context below (from the book). Do not use outside knowledge.
If context is insufficient, say so and suggest what to search for (still book-related).

Write a structured lesson in Markdown with these sections (exact headers):
## 🎯 Learning Goal (Customized)
## 📖 Core Concepts
## 💡 Role-Relevant Example
## ⚠️ Common Pitfalls
## 📝 Summary

Also, when possible, reference (Chunk X, page Y) inline.

Context:
{context_text}

Lesson title:
{lesson_title}
"""
    return call_llm(prompt, temperature=0.2)

def generate_quiz_json(topic: str, context_text: str, user_level: str, n_questions: int = 3) -> dict:
    level_instruction = build_level_instruction(user_level)
    persona = get_persona_instruction()
    prompt = f"""
{persona}
{level_instruction}
You are generating a checkpoint quiz for a course lesson in a Streamlit app.

Use ONLY the context below (from the book). Do not use outside knowledge.
Return STRICT JSON ONLY (no markdown, no extra text).

JSON schema:
{{
  "title": "string",
  "questions": [
    {{
      "question": "string",
      "options": ["A ...", "B ...", "C ...", "D ..."],
      "correct_index": 0,
      "explanation": "1-3 sentences explanation based ONLY on context",
      "evidence": "A direct short quote from the context (<= 25 words)",
      "source": "[Chunk X | page Y]"
    }}
  ]
}}

Rules:
- Make exactly {n_questions} questions.
- Each question must have exactly 4 options.
- correct_index is 0-3.
- evidence must be a short quote (<= 25 words).
- source must cite a chunk/page from the given context.

Context:
{context_text}

Topic:
{topic}
"""
    raw = call_llm(prompt, temperature=0.1).strip()

    # safety strip fences
    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]

    try:
        return json.loads(raw)
    except:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end+1])
            except:
                pass
    return {"title": "Checkpoint", "questions": []}

def generate_final_exam_json(plan_titles: str, context_text: str, user_level: str, n_questions: int = 5) -> dict:
    level_instruction = build_level_instruction(user_level)
    persona = get_persona_instruction()
    prompt = f"""
{persona}
{level_instruction}
Create a FINAL EXAM covering the course based ONLY on the provided context from the book.
Return STRICT JSON ONLY (no markdown).

Schema:
{{
  "title": "Final Exam",
  "questions": [
    {{
      "question": "string",
      "options": ["A ...", "B ...", "C ...", "D ..."],
      "correct_index": 0,
      "explanation": "1-3 sentences based ONLY on context",
      "evidence": "A direct short quote (<= 25 words)",
      "source": "[Chunk X | page Y]"
    }}
  ]
}}

Rules:
- Exactly {n_questions} questions.
- Each question has 4 options.
- evidence must be <= 25 words.

Course plan titles:
{plan_titles}

Context:
{context_text}
"""
    raw = call_llm(prompt, temperature=0.2).strip()

    if "```json" in raw:
        raw = raw.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in raw:
        raw = raw.split("```", 1)[1].split("```", 1)[0]

    try:
        return json.loads(raw)
    except:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(raw[start:end+1])
            except:
                pass
    return {"title": "Final Exam", "questions": []}

def generate_performance_review() -> str:
    p = st.session_state.user_profile
    course_title = "Intelligent Interactive Systems"
    if st.session_state.course_plan and st.session_state.course_plan.get("course_title"):
        course_title = st.session_state.course_plan["course_title"]

    current_date = datetime.date.today().strftime("%B %d, %Y")
    final_score = st.session_state.exam_score

    prompt = f"""
Write a specific, professional performance review for {p['name']}, a {p['role']}.

MANDATORY VARIABLES:
- Course Name: {course_title}
- Date: {current_date}
- Score: {final_score}/5
- User Goal: {p['goal']}

INSTRUCTIONS:
1. Write a narrative summary of their performance.
2. Analyze how this course helps a {p['role']}.
3. Sign off EXACTLY as: "Best regards,\\nYour AI Tutor"
4. NO placeholders.
"""
    return call_llm(prompt, temperature=0.3)

# =================== 8) VECTOR + BM25 + FUSION DB (your strong retrieval + chapter filter) ===================

class FusionPDFVectorDB:
    def __init__(self, pdf_path: str, vector_store_path: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if not os.path.exists(pdf_path):
            st.error(f"❌ PDF not found: {pdf_path}")
            self.vector_store = None
            self.documents = []
            self.bm25 = None
            self.bm25_docs = []
            return

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=160,
            length_function=len,
        )
        self.documents: List[Document] = splitter.split_documents(documents)

        # Assign chapter_number metadata
        self._assign_chapters_to_chunks()

        # BM25
        self.bm25_docs = self.documents
        tokenized_docs = [doc.page_content.split() for doc in self.bm25_docs]
        self.bm25 = BM25Okapi(tokenized_docs)

        # Vector store
        try:
            if os.path.exists(vector_store_path):
                self.vector_store = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                test_emb = self.embeddings.embed_query("test")
                self.vector_store.index.search(np.array([test_emb]), k=1)
            else:
                self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
                os.makedirs(vector_store_path, exist_ok=True)
                self.vector_store.save_local(vector_store_path)
        except Exception:
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
            os.makedirs(vector_store_path, exist_ok=True)
            self.vector_store.save_local(vector_store_path)

    def _assign_chapters_to_chunks(self):
    # Match either:
    # 1) "Chapter 1"
    # 2) "1 Some Title" at the start of a line (but NOT "1.1")
        chap_pat = re.compile(r"(?im)^\s*(?:chapter\s+)?(10|[1-9])\s+(?!\.)")

        current_chapter = None
        for d in self.documents:
            text = (d.page_content or "")[:1500]  # look near the top of chunk
            m = chap_pat.search(text)
            if m:
                n = int(m.group(1))
                if 1 <= n <= 50:
                    current_chapter = n
            d.metadata["chapter_number"] = current_chapter


    def vector_search(self, query: str, k: int = 6) -> List[Document]:
        if not self.vector_store:
            return []
        try:
            return self.vector_store.similarity_search(query, k=k)
        except:
            return []

    def bm25_search(self, query: str, k: int = 6) -> List[Document]:
        if not self.bm25:
            return []
        try:
            scores = self.bm25.get_scores(query.split())
            sorted_idx = np.argsort(scores)[::-1][:k]
            return [self.bm25_docs[i] for i in sorted_idx if i < len(self.bm25_docs)]
        except:
            return []

    def fusion_search(self, query: str, k: int = 6) -> List[Document]:
        # RRF fusion between vector + bm25
        try:
            top_n = max(k, 10)
            vec = self.vector_store.similarity_search_with_score(query, k=top_n)
            vector_docs = [d for d, _ in vec]

            bm25_scores = self.bm25.get_scores(query.split())
            bm25_sorted = np.argsort(bm25_scores)[::-1][:top_n]
            bm25_docs = [self.bm25_docs[i] for i in bm25_sorted if i < len(self.bm25_docs)]

            rrf_k = 60
            scores: Dict[Any, Any] = {}

            def _key(d: Document):
                return (d.metadata.get("source", ""), d.metadata.get("page", -1), (d.page_content or "")[:120])

            for rank, d in enumerate(vector_docs, start=1):
                scores.setdefault(_key(d), {"doc": d, "score": 0.0})
                scores[_key(d)]["score"] += 1.0 / (rrf_k + rank)

            for rank, d in enumerate(bm25_docs, start=1):
                scores.setdefault(_key(d), {"doc": d, "score": 0.0})
                scores[_key(d)]["score"] += 1.0 / (rrf_k + rank)

            merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
            return [m["doc"] for m in merged[:k]]
        except:
            return self.vector_search(query, k=k)

# Retrieval helper with TOC filter + chapter filter
def retrieve(db: FusionPDFVectorDB, query: str, k: int, chapter_number: Optional[int], search_mode: str) -> List[Document]:
    kk = max(k * 3, 15)

    if search_mode == "Fusion":
        docs = db.fusion_search(query, k=kk)
    elif search_mode == "Vector":
        docs = db.vector_search(query, k=kk)
    else:
        docs = db.bm25_search(query, k=kk)

    # remove TOC-like chunks
    docs = [d for d in docs if not is_contents_chunk_text(d.page_content)]

    # optional chapter filter (STRICT)
    if chapter_number is not None:
        filtered = [d for d in docs if d.metadata.get("chapter_number") == chapter_number]
        if filtered:        # <-- even 1 chunk is better than none
            docs = filtered


    return docs[:k]

# Chat answer with sources
def answer_question(db: FusionPDFVectorDB, user_question: str, lesson_title: str, lesson_query: str,
                    user_level: str, k: int, chapter_number: int, search_mode: str):
    rag_query = f"{lesson_query} | {user_question}"
    context_docs = retrieve(db, rag_query, k=k, chapter_number=chapter_number, search_mode=search_mode)
    context_text = format_context_for_prompt(context_docs)

    level_instruction = build_level_instruction(user_level)
    persona = get_persona_instruction()

    prompt = f"""
{persona}
{level_instruction}
You are a helpful course tutor inside a Lesson page.

Answer the user's question using ONLY the context below (from the book).
If the context does not contain the answer, say:
"I can't find this in the provided book context for this lesson."
Then suggest 1-2 short book-related keywords to search inside the book.

Keep the answer clear and structured.
When possible, cite (Chunk X, page Y) inline.

Lesson title: {lesson_title}
User question: {user_question}

Context:
{context_text}
"""
    answer = call_llm(prompt, temperature=0.2)
    sources = build_sources(context_docs, max_sources=4)
    return answer, sources

# =================== 9) STREAMLIT APP ===================


st.set_page_config(page_title="Smart Book Tutor", page_icon="🧠📚", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
set_bg_image("assets/bg.png")  # optional

# ---- State init (friend structure) ----
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "setup_complete": False,
        "name": "",
        "role": "",
        "goal": "",
        "level": "Beginner",
        "k_chunks": 6,
        "search_mode": "Fusion",  # added (your retrieval)
    }

if "course_plan" not in st.session_state:
    st.session_state.course_plan = None
if "progress" not in st.session_state:
    st.session_state.progress = {}  # lid -> {"passed": bool, "score": int}

# # # ===== DEV TESTING ONLY =====
# if "course_plan" in st.session_state and st.session_state.course_plan:
#     for l in st.session_state.course_plan["lessons"]:
#         st.session_state.progress[l["lesson_id"]] = {
#             "passed": True,
#             "score": PASSING_THRESHOLD
#         }


if "lesson_cache" not in st.session_state:
    st.session_state.lesson_cache = {}  # lid -> {"content": md, "sources": [...] , "context_docs": [...]}
if "quiz_cache" not in st.session_state:
    st.session_state.quiz_cache = {}  # lid -> quiz_json
if "final_exam" not in st.session_state:
    st.session_state.final_exam = None
if "certified" not in st.session_state:
    st.session_state.certified = False
if "exam_score" not in st.session_state:
    st.session_state.exam_score = 0

# # ===== DEV TESTING ONLY =====
# if "certified" not in st.session_state:
#     st.session_state.certified = True   # force certificate
# if "exam_score" not in st.session_state:
#     st.session_state.exam_score = 4     # passing score

if "current_lesson_idx" not in st.session_state:
    st.session_state.current_lesson_idx = 0

@st.cache_resource
def get_db():
    return FusionPDFVectorDB(PDF_PATH, VECTOR_STORE_PATH)

db = get_db()

# =================== VIEW: ONBOARDING (friend) ===================

if not st.session_state.user_profile["setup_complete"]:
    st.markdown('<div class="onboarding-card">', unsafe_allow_html=True)
    st.title("👋 Hello! I am your Intelligent Book System.")
    st.write("I will analyze the book and create a **chapter-based** learning path (Curriculum → Lessons → Quizzes → Final Exam → Certificate).")
    st.markdown("</div>", unsafe_allow_html=True)

   
    # =================== Onboarding Form (SUBMIT - inside the form) ===================
    with st.form("onboarding_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", placeholder="e.g. Alex")
            role = st.text_input("Role/Profession", placeholder="e.g. Data Scientist")
        with col2:
            goal = st.text_input("Main Goal", placeholder="e.g. Master the basics")
            level = st.select_slider("Level", options=["Beginner", "Advanced"], value="Beginner")

        if st.form_submit_button("🚀 Generate Curriculum"):
            if name and role and goal:
                st.session_state.user_profile.update({
                    "setup_complete": True,
                    "name": name,
                    "role": role,
                    "goal": goal,
                    "level": level,
                    "search_mode": st.session_state.onboarding_search_mode,
                    "k_chunks": enforce_min_k(
                        st.session_state.onboarding_search_mode,
                        st.session_state.onboarding_k_chunks
                    ),
                })

                with st.spinner("Building chapter-based curriculum from the book…"):
                    plan = build_chapter_based_course_plan(PDF_PATH)
                    st.session_state.course_plan = plan

                # reset course state
                st.session_state.progress = {}
                st.session_state.lesson_cache = {}
                st.session_state.quiz_cache = {}
                st.session_state.final_exam = None
                st.session_state.certified = False
                st.session_state.exam_score = 0
                st.session_state.current_lesson_idx = 0

                st.rerun()
            else:
                st.error("Please fill in your Name, Role, and Goal.")

     # =================== Advanced Retrieval (DYNAMIC - outside the form) ===================
    with st.expander("⚙️ Advanced Settings"):
        search_mode = st.selectbox(
            "Search mode",
            options=["Fusion", "Vector", "BM25"],
            index=0,
            format_func=lambda x: SEARCH_MODE_LABELS.get(x, x),
            key="onboarding_search_mode",
        )

        # dynamic help text (updates immediately)
        st.caption(SEARCH_MODE_HELP.get(search_mode, ""))

        # dynamic "recommended" message (updates immediately)
        if search_mode == "Fusion":
            st.success("✅ Recommended: best overall reliability.")
        elif search_mode == "Vector":
            st.info("💡 Best for concept/definition questions.")
        else:
            st.info("🔎 Best for exact keywords, acronyms, and quotes.")

        # ----- Dynamic slider min/default behavior -----
        min_k = MIN_K_BY_MODE.get(search_mode, 6)

        # initialize slider state once
        if "onboarding_k_chunks" not in st.session_state:
            st.session_state.onboarding_k_chunks = min_k

        # if user switches mode and current k is below the new minimum, auto-bump it up
        if int(st.session_state.onboarding_k_chunks) < int(min_k):
            st.session_state.onboarding_k_chunks = min_k

        k_chunks = st.slider(
            "How much of the book the AI reads (chunks)",
            3, 12,
            value=int(st.session_state.onboarding_k_chunks),
            key="onboarding_k_chunks",
        )

        st.caption(
            "Each chunk is a small overlapping section of the book. "
            "More chunks = broader context (higher recall). Fewer chunks = more focused answers."
        )
        st.caption(retrieval_hint(search_mode))

        if k_chunks < min_k:
            st.warning(f"⚠️ For this mode, set **k ≥ {min_k}** for reliable retrieval.")


# =================== VIEW: MAIN APP ===================

else:
    with st.sidebar:
        p = st.session_state.user_profile
        st.markdown(f"### 👤 {p['name']}")
        st.caption(f"{p['role']} | {p['level']}")

        lessons_list = st.session_state.course_plan.get("lessons", []) if st.session_state.course_plan else []
        total_lessons = max(1, len(lessons_list))
        passed_lessons = len([k for k, v in st.session_state.progress.items() if v.get("passed")])

        st.progress(passed_lessons / total_lessons)
        st.caption(f"{passed_lessons}/{len(lessons_list)} Chapters Passed")

        st.markdown("---")
        page = st.radio("Navigation", ["📚 Curriculum", "📖 Active Lesson", "🎓 Certification", "⚙️ Settings"])

        st.markdown("---")
        if st.button("🔄 Reset Profile"):
            # core profile + course
            st.session_state.user_profile["setup_complete"] = False
            st.session_state.course_plan = None

            # learning progress
            st.session_state.progress = {}
            st.session_state.lesson_cache = {}
            st.session_state.quiz_cache = {}
            st.session_state.final_exam = None
            st.session_state.certified = False
            st.session_state.exam_score = 0
            st.session_state.current_lesson_idx = 0

            # 🔥 CLEAR dynamic chat & quiz state
            clear_dynamic_session_state([
                "chat_history_",
                "chat_input_",
                "quiz_submitted_",
                "q_",          # quiz radio buttons
                "fe_",         # final exam radios
            ])

            st.rerun()


    # ---------- PAGE: CURRICULUM ----------
    if page == "📚 Curriculum":
        if not st.session_state.course_plan:
            st.error("❌ Curriculum not found. Reset Profile and generate again.")
            st.stop()

        st.title(f"📚 {st.session_state.course_plan.get('course_title', 'Course Curriculum')}")

        lessons = st.session_state.course_plan.get("lessons", [])
        if not lessons:
            st.warning("No lessons found. Try regenerating curriculum.")
            st.stop()

        for i, lesson in enumerate(lessons):
            lid = lesson["lesson_id"]
            passed = st.session_state.progress.get(lid, {}).get("passed", False)
            status_icon = "✅" if passed else "🟦"

            prev_passed = True if i == 0 else st.session_state.progress.get(lessons[i - 1]["lesson_id"], {}).get("passed", False)

            with st.container():
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.subheader(f"{status_icon} Chapter {lid}: {lesson['title']}")
                    st.write(lesson.get("goal", ""))
                    st.caption(f"Retrieval query: {lesson.get('search_query','')}")
                with c2:
                    if prev_passed:
                        if st.button("Start", key=f"go_{lid}"):
                            st.session_state.current_lesson_idx = i
                            st.toast(f"Loading Chapter {lid}...")
                            st.info("Switch to '📖 Active Lesson'")
                    else:
                        st.button("🔒 Locked", disabled=True, key=f"lock_{lid}")
                st.divider()

    # ---------- PAGE: ACTIVE LESSON ----------
    elif page == "📖 Active Lesson":
        if not st.session_state.course_plan:
            st.info("Generate a curriculum first.")
            st.stop()

        lessons = st.session_state.course_plan.get("lessons", [])
        if not lessons:
            st.info("No lessons available.")
            st.stop()

        # Choose lesson (only unlocked)
        idx = st.session_state.current_lesson_idx
        idx = max(0, min(idx, len(lessons) - 1))
        lesson_data = lessons[idx]
        lid = lesson_data["lesson_id"]
        chapter_num = lesson_data.get("chapter_number", lid)

        # Lock enforcement: can't open if previous not passed (except first)
        if idx > 0:
            prev_lid = lessons[idx - 1]["lesson_id"]
            if not st.session_state.progress.get(prev_lid, {}).get("passed", False):
                st.warning("🔒 This chapter is locked. Pass the previous chapter quiz first.")
                st.stop()

        st.title(f"📖 Chapter {lid}: {lesson_data['title']}")

        # Build a strong chapter query (your approach)
        query = lesson_data.get("search_query") or f"Chapter {chapter_num} {lesson_data['title']}"

        # Lesson generation (cached)
        if lid not in st.session_state.lesson_cache:
            with st.spinner(f"Retrieving chapter context (k={p['k_chunks']}, mode={p['search_mode']})…"):
                context_docs = retrieve(
                    db=db,
                    query=query,
                    k=p["k_chunks"],
                    chapter_number=chapter_num,
                    search_mode=p["search_mode"],
                )
                context_text = format_context_for_prompt(context_docs)

            with st.spinner("Generating lesson from the book…"):
                content = generate_lesson_markdown(lesson_data["title"], context_text, p["level"])

            st.session_state.lesson_cache[lid] = {
                "content": content,
                "sources": build_sources(context_docs, max_sources=4),
                "context_docs": context_docs,
                "query": query,
            }

        # Render lesson
        st.markdown(st.session_state.lesson_cache[lid]["content"])

        # Sources (your feature)
        sources = st.session_state.lesson_cache[lid].get("sources", [])
        if sources:
            with st.expander("📌 Sources (from the book)", expanded=False):
                for s in sources:
                    st.markdown(f"**▶ {s['label']}**")
                    st.write(s["snippet"])

        st.divider()

        # Chat (your sources)
        st.subheader("💬 Chat with AI Tutor (chapter-scoped)")
        chat_key = f"chat_history_{lid}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for msg in st.session_state[chat_key]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    with st.expander("📌 Sources", expanded=False):
                        for s in msg["sources"]:
                            st.markdown(f"**▶ {s['label']}**")
                            st.write(s["snippet"])

        user_q = st.chat_input("Ask a question about this chapter…", key=f"chat_input_{lid}")
        if user_q:
            st.session_state[chat_key].append({"role": "user", "content": user_q})
            with st.chat_message("user"):
                st.markdown(user_q)

            with st.chat_message("assistant"):
                with st.spinner("Answering from the book…"):
                    ans, srcs = answer_question(
                        db=db,
                        user_question=user_q,
                        lesson_title=lesson_data["title"],
                        lesson_query=query,
                        user_level=p["level"],
                        k=enforce_min_k(p["search_mode"], p["k_chunks"]),
                        chapter_number=chapter_num,
                        search_mode=p["search_mode"],
                    )
                st.markdown(ans)
                if srcs:
                    with st.expander("📌 Sources", expanded=False):
                        for s in srcs:
                            st.markdown(f"**▶ {s['label']}**")
                            st.write(s["snippet"])

            st.session_state[chat_key].append({"role": "assistant", "content": ans, "sources": srcs})

        if st.button("🧹 Clear chat", key=f"clear_chat_{lid}"):
            st.session_state[chat_key] = []
            st.rerun()

        st.divider()
        st.subheader("✅ Active Learning Quiz (3 Questions)")

        quiz_data = st.session_state.quiz_cache.get(lid)

        if not quiz_data or not quiz_data.get("questions"):
            if st.button("📝 Generate Quiz", key=f"gen_quiz_{lid}"):
                with st.spinner("Generating quiz from the book (auto-retry if busy)…"):
                    context_docs = st.session_state.lesson_cache[lid]["context_docs"]
                    context_text = format_context_for_prompt(context_docs)
                    qz = generate_quiz_json(
                        topic=f"Checkpoint for Chapter {lid}: {lesson_data['title']}",
                        context_text=context_text,
                        user_level=p["level"],
                        n_questions=3
                    )
                    if qz and qz.get("questions"):
                        st.session_state.quiz_cache[lid] = qz
                        st.rerun()
                    else:
                        st.error("⚠️ Failed to generate quiz. Try again (maybe rate limit).")
        else:
            quiz = quiz_data
            quiz_key = f"quiz_submitted_{lid}"
            if quiz_key not in st.session_state:
                st.session_state[quiz_key] = False

            # Form quiz
            with st.form(key=f"quiz_{lid}"):
                user_selections = {}
                for i, q in enumerate(quiz.get("questions", [])):
                    st.markdown(f"**{i+1}. {q['question']}**")
                    opts = q["options"]
                    user_selections[i] = st.radio(
                        "Select:",
                        options=list(range(4)),
                        format_func=lambda idx_opt: opts[idx_opt],
                        key=f"q_{lid}_{i}"
                    )
                    st.caption(f"Source: {q.get('source','')}")
                    st.divider()
                sub_quiz = st.form_submit_button("Submit Answers")

            if sub_quiz or st.session_state[quiz_key]:
                st.session_state[quiz_key] = True
                current_score = 0

                for i, q in enumerate(quiz["questions"]):
                    correct_idx = q["correct_index"]
                    user_idx = user_selections.get(i)
                    is_correct = (user_idx == correct_idx)
                    if is_correct:
                        current_score += 1
                        st.success(f"✅ Q{i+1} Correct! {q.get('explanation','')}")
                        if q.get("evidence"):
                            st.markdown(f"<div class='evidence-text'>Evidence: “{q['evidence']}”</div>", unsafe_allow_html=True)
                    else:
                        correct_opt = q["options"][correct_idx]
                        st.error(f"❌ Q{i+1} Wrong. Correct: {correct_opt}. {q.get('explanation','')}")
                        if q.get("evidence"):
                            st.markdown(f"<div class='evidence-text'>Evidence: “{q['evidence']}”</div>", unsafe_allow_html=True)

                # pass/fail + locking (friend behavior)
                if current_score >= PASSING_THRESHOLD:
                    st.balloons()
                    st.success(f"🎉 PASSED! {current_score}/3")
                    st.session_state.progress[lid] = {"passed": True, "score": current_score}
                    if st.button("Update Progress"):
                        st.rerun()
                else:
                    st.warning(f"Failed. {current_score}/3. Retry required.")
                    if st.button("🔄 Retry Quiz"):
                        if lid in st.session_state.quiz_cache:
                            del st.session_state.quiz_cache[lid]
                        st.session_state[quiz_key] = False
                        st.rerun()

        # Navigation between lessons (locked)
        st.divider()
        nav1, nav2 = st.columns([1, 1])
        with nav1:
            if st.button("⬅ Previous", disabled=(idx == 0), key=f"prev_{lid}"):
                st.session_state.current_lesson_idx = max(0, idx - 1)
                st.rerun()
        with nav2:
            # next allowed only if passed current
            next_disabled = (idx >= len(lessons) - 1) or (not st.session_state.progress.get(lid, {}).get("passed", False))
            if st.button("Next ➡", disabled=next_disabled, key=f"next_{lid}"):
                st.session_state.current_lesson_idx = min(len(lessons) - 1, idx + 1)
                st.rerun()

    # ---------- PAGE: CERTIFICATION ----------
    elif page == "🎓 Certification":
        st.title("🎓 Final Certification")

        lessons = st.session_state.course_plan.get("lessons", []) if st.session_state.course_plan else []
        total = len(lessons)
        if total == 0:
            st.error("⚠️ Curriculum is empty. Please Reset Profile.")
            st.stop()

        completed = len([k for k, v in st.session_state.progress.items() if v.get("passed")])
        if completed < total:
            st.warning(f"⚠️ Complete all chapters to unlock the Final Exam. ({completed}/{total} completed)")
            st.progress(completed / max(1, total))
            st.stop()

        # FINAL EXAM (5 q) -> certificate
        if not st.session_state.certified:
            st.subheader("🏆 Grand Final Exam (5 Questions)")

            if not st.session_state.final_exam or not st.session_state.final_exam.get("questions"):
                if st.button("📝 Generate Final Exam", key="gen_final_exam"):
                    with st.spinner("Generating Final Exam from the book…"):
                        # Make a mixed context by sampling each chapter query (chapter-based)
                        merged_docs: List[Document] = []
                        seen = set()

                        # use smaller k per chapter to keep prompt size sane
                        per_ch_k = max(2, min(4, st.session_state.user_profile["k_chunks"] // 2))
                        for l in lessons:
                            q = l.get("search_query") or f"Chapter {l.get('chapter_number', l['lesson_id'])} {l.get('title','')}"
                            docs = retrieve(
                                db=db,
                                query=q,
                                k=per_ch_k,
                                chapter_number=l.get("chapter_number", l["lesson_id"]),
                                search_mode=st.session_state.user_profile["search_mode"],
                            )
                            for d in docs:
                                key = (d.metadata.get("page", -1), (d.page_content or "")[:120])
                                if key not in seen:
                                    seen.add(key)
                                    merged_docs.append(d)

                        merged_docs = merged_docs[:14]
                        ctx = format_context_for_prompt(merged_docs)

                        plan_titles = ", ".join([l.get("title", "") for l in lessons])
                        exam = generate_final_exam_json(
                            plan_titles=plan_titles,
                            context_text=ctx,
                            user_level=st.session_state.user_profile["level"],
                            n_questions=5
                        )
                        if exam and exam.get("questions"):
                            st.session_state.final_exam = exam
                            st.session_state.final_exam_context_sources = build_sources(merged_docs, max_sources=4)
                            st.rerun()
                        else:
                            st.error("⚠️ Failed to generate exam. Try again.")

                st.caption("Click to generate a final exam based strictly on the book.")
                st.stop()

            exam = st.session_state.final_exam
            st.markdown(f"#### {exam.get('title','Final Exam')}")

            with st.expander("📌 Final Exam Sources (book)", expanded=False):
                for s in st.session_state.get("final_exam_context_sources", []):
                    st.markdown(f"**▶ {s['label']}**")
                    st.write(s["snippet"])

            with st.form("final_exam_form"):
                answers = {}
                for i, q in enumerate(exam["questions"]):
                    st.markdown(f"**{i+1}. {q['question']}**")
                    opts = q["options"]
                    answers[i] = st.radio(
                        "Select:",
                        options=list(range(4)),
                        format_func=lambda idx_opt: opts[idx_opt],
                        key=f"fe_{i}"
                    )
                    st.caption(f"Source: {q.get('source','')}")
                    if q.get("evidence"):
                        st.markdown(
                            f"<div class='evidence-text'>Evidence: “{q['evidence']}”</div>",
                            unsafe_allow_html=True
                        )
                    st.divider()

                submitted = st.form_submit_button("Submit Final Exam")

            # ----- OUTSIDE the form -----
            if submitted:
                score = 0
                for i, q in enumerate(exam["questions"]):
                    if answers[i] == q["correct_index"]:
                        score += 1

                st.session_state.exam_score = score

                if score >= FINAL_PASSING_THRESHOLD:
                    st.balloons()
                    st.session_state.certified = True
                    st.rerun()
                else:
                    st.error(f"Score: {score}/5. You need {FINAL_PASSING_THRESHOLD}/5. Please try again.")

            # ✅ Regenerate button MUST be outside the form
            if not st.session_state.certified:
                if st.button("🔄 Regenerate Exam", key="regen_exam_btn"):
                    st.session_state.final_exam = None

                    # clear old final exam answers from session_state
                    for k in list(st.session_state.keys()):
                        if k.startswith("fe_"):
                            del st.session_state[k]

                    st.rerun()


        # CERTIFICATE VIEW
        if st.session_state.certified:
            st.success("🎉 CONGRATULATIONS! You have mastered this book.")
            course_name = st.session_state.course_plan.get("course_title", "Intelligent Interactive Systems")
            final_score_pct = int((st.session_state.exam_score / 5) * 100)

            with st.spinner("Generating Performance Review..."):
                review = generate_performance_review()
            st.info(f"**AI Tutor Feedback:**\n\n{review}")

            st.markdown(
                f"""
                <div class="certificate-container">
                    <div class="seal">OFFICIAL<br>SEAL</div>
                    <div class="cert-header">Certificate of Completion</div>
                    <div class="cert-sub">This certifies that</div>
                    <div class="cert-name">{st.session_state.user_profile['name']}</div>
                    <div class="cert-body">
                        Has successfully completed the comprehensive course on<br>
                        <strong>{course_name}</strong><br>
                        Demonstrating proficiency in all required modules for the role of <strong>{st.session_state.user_profile['role']}</strong>.
                    </div>
                    <div class="cert-grade">Final Grade: {final_score_pct}%</div>
                    <div class="cert-footer">
                        <div class="signature">AI Tutor Signature</div>
                        <div class="signature">Date: {datetime.date.today()}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.button("🖨️ Print Certificate (Ctrl+P)")

    # ---------- PAGE: SETTINGS ----------
    elif page == "⚙️ Settings":
        st.header("Settings")
        p = st.session_state.user_profile

    

        # ============ FORM (submit only) ============
        with st.form("edit_settings"):
            st.markdown("### Profile")
            new_name = st.text_input("Name", value=p["name"])
            new_role = st.text_input("Role", value=p["role"])
            new_goal = st.text_input("Goal", value=p["goal"])
            new_level = st.selectbox(
                "Level",
                ["Beginner", "Advanced"],
                index=0 if p["level"] == "Beginner" else 1
            )

            if st.form_submit_button("Save"):
                st.session_state.user_profile.update({
                    "name": new_name,
                    "role": new_role,
                    "goal": new_goal,
                    "level": new_level,
                    "search_mode": st.session_state.settings_search_mode,
                    "k_chunks": enforce_min_k(
                        st.session_state.settings_search_mode,
                        st.session_state.settings_k_chunks
                    ),
                })
                st.toast("Saved!")
                st.rerun()


        # ============ DYNAMIC RETRIEVAL CONTROLS (outside form) ============
        st.markdown("### Retrieval")

        dyn_search = st.selectbox(
            "Search mode",
            options=["Fusion", "Vector", "BM25"],
            index=["Fusion", "Vector", "BM25"].index(p["search_mode"]),
            format_func=lambda x: SEARCH_MODE_LABELS.get(x, x),
            key="settings_search_mode",
        )

        st.caption(SEARCH_MODE_HELP.get(dyn_search, ""))

        if dyn_search == "Fusion":
            st.success("✅ Recommended: best overall reliability.")
        elif dyn_search == "Vector":
            st.info("💡 Best for concept/definition questions.")
        else:
            st.info("🔎 Best for exact keywords, acronyms, and quotes.")

        min_k = MIN_K_BY_MODE.get(dyn_search, 6)

        # initialize / keep slider state
        if "settings_k_chunks" not in st.session_state:
            st.session_state.settings_k_chunks = int(p["k_chunks"])

        # if user changes mode and current k is too low, bump it
        if int(st.session_state.settings_k_chunks) < int(min_k):
            st.session_state.settings_k_chunks = int(min_k)

        dyn_k = st.slider(
            "How much of the book the AI reads (chunks)",
            3, 12,
            value=int(st.session_state.settings_k_chunks),
            key="settings_k_chunks",
        )

        st.caption(
            "Each chunk is a small overlapping section of the book. "
            "More chunks = broader context (higher recall). Fewer chunks = more focused answers."
        )
        st.caption(retrieval_hint(dyn_search))

        if dyn_k < min_k:
            st.warning(f"⚠️ For this mode, set **k ≥ {min_k}** for reliable retrieval.")

        st.markdown("---")
