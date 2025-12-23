import os
import json
import re
from typing import List, Dict, Any

import numpy as np
import requests
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import base64

def set_bg_image(image_path: str):
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        /* App background */
        .stApp {{
            background: url("data:image/png;base64,{b64}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* Make cards readable on bright background */
        .card {{
            background: rgba(255,255,255,0.08) !important;
            backdrop-filter: blur(6px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =================== CONFIG ===================

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

PDF_PATH = r"C:\Users\asus\Documents\pojectAI\intelligent_interactive_systems.pdf"
VECTOR_STORE_PATH = "./vector_stores/course_book_store"


# =================== APP STYLE ===================

APP_CSS = """
<style>

/* =========================
   GLOBAL LAYOUT
========================= */

.block-container {
  padding-top: 1.2rem;
  padding-bottom: 2rem;
  max-width: 1100px;
}

header { visibility: hidden; }
footer { visibility: hidden; }

/* =========================
   GLOBAL TEXT COLORS
========================= */

html, body, [class*="css"] {
  color: #000000 !important;
  font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
  color: #000000 !important;
  font-weight: 700;
}

/* Paragraphs & labels */
p, span, label, div, li {
  color: #000000 !important;
}

/* Muted text */
.muted {
  color: rgba(0,0,0,0.65) !important;
}

/* =========================
   CARD DESIGN
========================= */

.card {
  background: rgba(255,255,255,0.88);
  border-radius: 20px;
  padding: 20px;
  border: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 10px 22px rgba(0,0,0,0.14);
  margin-bottom: 16px;
}

/* =========================
   HERO SECTION
========================= */

.hero-title {
  font-size: 2.1rem;
  font-weight: 800;
  margin-bottom: 6px;
}

.hero-sub {
  font-size: 1.05rem;
  opacity: 0.75;
}

/* Divider */
.hr {
  height: 1px;
  background: rgba(0,0,0,0.14);
  margin: 14px 0;
  border-radius: 999px;
}

/* =========================
   BUTTONS (FIXED TEXT ISSUE)
========================= */

.stButton > button {
  background: rgba(18, 18, 28, 0.94) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,0.28) !important;
  border-radius: 16px !important;
  padding: 0.7rem 1.2rem !important;
  font-weight: 700 !important;
  box-shadow: 0 10px 20px rgba(0,0,0,0.22) !important;
  transition: all 0.18s ease-in-out;
}

/* Force text color inside button */
.stButton > button *,
.stButton > button span {
  color: #ffffff !important;
}

/* Hover */
.stButton > button:hover {
  background: rgba(32, 32, 52, 0.96) !important;
  border-color: rgba(255,255,255,0.4) !important;
  transform: translateY(-1px);
}

/* Active */
.stButton > button:active {
  transform: translateY(0);
}

/* =========================
   RADIO / SELECT / SLIDER
========================= */

div[role="radiogroup"] label {
  font-weight: 600;
}

.stSelectbox label,
.stSlider label {
  font-weight: 600;
}

/* Slider color */
div[data-baseweb="slider"] > div > div {
  background-color: #e74c3c !important;
}

/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.92);
  border-right: 1px solid rgba(0,0,0,0.08);
}

/* Sidebar titles */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  font-weight: 700;
}

/* =========================
   EXPANDERS
========================= */

details {
  background: rgba(255,255,255,0.75);
  border-radius: 14px;
  padding: 10px;
  border: 1px solid rgba(0,0,0,0.08);
}

details summary {
  font-weight: 600;
  cursor: pointer;
}

/* =========================
   PROGRESS & FEEDBACK
========================= */

.stSuccess {
  background: rgba(46, 204, 113, 0.12) !important;
  border-left: 4px solid #2ecc71 !important;
}

.stInfo {
  background: rgba(52, 152, 219, 0.12) !important;
  border-left: 4px solid #3498db !important;
}

.stWarning {
  background: rgba(241, 196, 15, 0.14) !important;
  border-left: 4px solid #f1c40f !important;
}

/* =========================
   MOBILE FRIENDLY
========================= */

@media (max-width: 768px) {
  .hero-title {
    font-size: 1.7rem;
  }

  .block-container {
    padding-left: 1rem;
    padding-right: 1rem;
  }
}

/* =========================
   SIDEBAR SELECTBOX (LESSON PICKER FIX)
========================= */

/* Label */
section[data-testid="stSidebar"] label {
  font-weight: 700 !important;
  color: #000000 !important;
}

/* Selectbox container */
section[data-testid="stSidebar"] div[data-baseweb="select"] {
  background: #ffffff !important;
  border-radius: 14px !important;
  border: 1px solid rgba(0,0,0,0.25) !important;
}

/* Selected value text */
section[data-testid="stSidebar"] div[data-baseweb="select"] span {
  color: #000000 !important;
  font-weight: 600 !important;
}

/* Dropdown arrow */
section[data-testid="stSidebar"] svg {
  fill: #000000 !important;
}

/* Dropdown menu */
div[data-baseweb="popover"] {
  background: #ffffff !important;
  border-radius: 14px !important;
  border: 1px solid rgba(0,0,0,0.2) !important;
}

/* Dropdown options */
div[data-baseweb="menu"] div {
  color: #000000 !important;
  font-weight: 500 !important;
}

/* Hovered option */
div[data-baseweb="menu"] div:hover {
  background: rgba(52, 152, 219, 0.12) !important;
}

</style>
"""


def extract_contents_text(pdf_path: str, max_scan_pages: int = 40, after_pages: int = 6) -> str:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    start_idx = None
    for i, d in enumerate(pages[:max_scan_pages]):
        if re.search(r"\bContents\b|\bTable of Contents\b", d.page_content, re.IGNORECASE):
            start_idx = i
            break

    if start_idx is None:
        return ""

    # take a few pages from the TOC section
    toc_docs = pages[start_idx:start_idx + after_pages]
    toc_text = "\n\n".join([d.page_content for d in toc_docs])
    return toc_text


def generate_course_plan_from_contents(contents_text: str, user_level: str, n_lessons: int = 6) -> dict:
    level_instruction = build_level_instruction(user_level)

    prompt = f"""
{level_instruction}
You are given the book's Table of Contents text (may include noisy lines).
Your job: create a clean 6-lesson curriculum that follows the TOC ORDER.

Use ONLY what appears in the Contents text (do not invent topics).

Rules:
- Ignore author names, editor names, affiliations.
- Ignore non-learning sections: References, Summary, Index, Acknowledgements, Preface.
- Prefer CHAPTER/SECTION titles that teach concepts.
- Return STRICT JSON ONLY.

Schema:
{{
  "course_title": "string",
  "lessons": [
    {{
      "lesson_id": 1,
      "title": "string",
      "goal": "1 sentence",
      "search_query": "string (use the exact section/chapter keywords)",
      "prereq": "string (can be empty)",
      "difficulty": "Beginner" | "Advanced"
    }}
  ]
}}

Make exactly {n_lessons} lessons, in order.

Contents text:
{contents_text}
"""
    raw = call_llm(prompt).strip()

    # robust JSON parse (same idea as your quiz)
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
    return {"course_title": "Course (From Contents)", "lessons": []}


TOC_LINE_RE = re.compile(r"^\s*(.+?)\s*\.{2,}\s*(\d+)\s*$")

def extract_toc_entries(vector_db, max_scan_pages: int = 25):
    """
    Returns a list of entries: [{"title": str, "start_printed_page": int}]
    based on the book's Contents pages.
    """
    # PyPDFLoader "documents" are split chunks; we want raw pages.
    # Luckily, vector_db was built from loader.load() then split.
    # We'll reconstruct per-page text by reading from loader again (simple + reliable).
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()  # one Document per page, metadata["page"] exists

    # join first pages into a buffer we can detect "Contents" block from
    scan_pages = pages[:max_scan_pages]
    text_by_page = [(p.metadata.get("page", 0) + 1, p.page_content) for p in scan_pages]

    # Find where Contents starts
    contents_start_idx = None
    for idx, (_, txt) in enumerate(text_by_page):
        if re.search(r"\bContents\b|\bTable of Contents\b", txt, flags=re.IGNORECASE):
            contents_start_idx = idx
            break

    if contents_start_idx is None:
        return []

    # Parse lines from contents pages (usually 1–3 pages)
    entries = []
    for _, txt in text_by_page[contents_start_idx:contents_start_idx + 6]:
        for line in txt.splitlines():
            m = TOC_LINE_RE.match(line.strip())
            if not m:
                continue
            title = m.group(1).strip()
            printed_page = int(m.group(2))
            # Filter junk
            if len(title) < 3:
                continue
            if title.lower() in {"contents", "table of contents"}:
                continue
            entries.append({"title": title, "start_printed_page": printed_page})

    # remove duplicates (sometimes repeated headers)
    uniq = []
    seen = set()
    for e in entries:
        key = (e["title"].lower(), e["start_printed_page"])
        if key not in seen:
            seen.add(key)
            uniq.append(e)

    return uniq

def estimate_printed_to_pdf_offset(toc_entries, pdf_pages_docs):
    """
    Estimate offset such that:
      pdf_page_number ≈ printed_page + offset
    Returns offset (int). If can't estimate, returns 0.
    """
    if not toc_entries:
        return 0

    first = toc_entries[0]
    title = first["title"]
    printed = first["start_printed_page"]

    # search where that title appears in actual PDF pages
    for doc in pdf_pages_docs:
        pdf_page_num = doc.metadata.get("page", 0) + 1
        if title.lower()[:25] in doc.page_content.lower():  # fuzzy-ish
            return pdf_page_num - printed

    return 0

def toc_to_lessons(toc_entries, pdf_pages_docs, user_level: str):
    """
    Returns lessons list with title + page_range in PDF pages.
    """
    # compute offset
    offset = estimate_printed_to_pdf_offset(toc_entries, pdf_pages_docs)

    # convert toc starts to pdf starts
    starts = []
    for e in toc_entries:
        pdf_start = e["start_printed_page"] + offset
        starts.append({"title": e["title"], "pdf_start": max(1, pdf_start)})

    # sort & build ranges
    starts.sort(key=lambda x: x["pdf_start"])

    lessons = []
    for i, cur in enumerate(starts, start=1):
        start = cur["pdf_start"]
        end = (starts[i]["pdf_start"] - 1) if i < len(starts) else (pdf_pages_docs[-1].metadata["page"] + 1)

        # sanity clamp
        end = max(start, end)

        lessons.append({
            "lesson_id": i,
            "title": cur["title"],
            "goal": "Learn this section from the book contents.",
            "difficulty": user_level,
            "page_range": f"{start}-{end}",
            "search_query": f'{cur["title"]} pages {start}-{end}'
        })

    return lessons




# =================== LLM ===================

def call_llm(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not set."

    try:
        r = requests.post(
            GROQ_ENDPOINT,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=60,
        )
        data = r.json()
        if "error" in data:
            return f"LLM Error: {data['error'].get('message', 'Unknown error')}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling Groq API: {e}"


# =================== HELPERS ===================

def build_level_instruction(level: str) -> str:
    if level == "Beginner":
        return (
            "You are teaching a BEGINNER student.\n"
            "- Use simple language, short sentences.\n"
            "- Explain terms briefly when first used.\n"
            "- Avoid heavy jargon.\n"
        )
    else:
        return (
            "You are teaching an ADVANCED student.\n"
            "- Use academic terminology.\n"
            "- Emphasize relationships, trade-offs, design implications.\n"
            "- Keep it structured and precise.\n"
        )

def answer_lesson_question(user_question: str, lesson_title: str, context_text: str, user_level: str) -> str:
    level_instruction = build_level_instruction(user_level)

    prompt = f"""
    {level_instruction}
    You are a lesson-specific assistant.

    You MUST answer using ONLY the context below (from the book chunks).
    - If the answer is not in the context, say: "I couldn't find that in this lesson's sources." and suggest what keyword to search in the book.
    - Keep answers structured and clear.
    - When relevant, cite like (Chunk X, page Y).

    Lesson title: {lesson_title}

    Context:
    {context_text}

    User question:
    {user_question}
    """
    return call_llm(prompt).strip()


def format_context_for_prompt(context_docs: List[Document]) -> str:
    chunks = []
    for i, d in enumerate(context_docs, start=1):
        page0 = d.metadata.get("page", None)
        page = (page0 + 1) if isinstance(page0, int) else "unknown"
        chunks.append(f"[Chunk {i} | page {page}]\n{d.page_content}")
    return "\n\n".join(chunks)

def build_sources(context_docs: List[Document], max_sources: int = 4):
    sources = []
    for i, d in enumerate(context_docs[:max_sources], start=1):
        page0 = d.metadata.get("page", None)
        page = (page0 + 1) if isinstance(page0, int) else "unknown"
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 520:
            snippet = snippet[:520] + "…"
        sources.append({
            "label": f"Chunk {i} — page {page}",
            "page": page,
            "snippet": snippet
        })
    return sources


# =================== QUIZ / CHECKPOINT ===================

def generate_quiz_json(topic: str, context_text: str, user_level: str, n_questions: int = 3) -> dict:
    level_instruction = build_level_instruction(user_level)
    quiz_prompt = f"""
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
      "source": "[Chunk X | page Y]"
    }}
  ]
}}

Rules:
- Make exactly {n_questions} questions.
- Each question must have exactly 4 options.
- correct_index is 0-3.
- source must cite a chunk/page from the given context.

Context:
{context_text}

Topic:
{topic}
"""
    raw = call_llm(quiz_prompt).strip()
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


# =================== COURSE PLAN GENERATION ===================

# =================== COURSE PLAN GENERATION ===================

def docs_in_page_range(all_docs: List[Document], page_range: str) -> List[Document]:
    """Return docs whose page is within the inclusive page_range 'a-b' (1-based)."""
    if not page_range:
        return []

    try:
        a, b = page_range.split("-")
        start, end = int(a.strip()), int(b.strip())
    except:
        return []

    out = []
    for d in all_docs:
        p0 = d.metadata.get("page", None)
        if isinstance(p0, int):
            p = p0 + 1
            if start <= p <= end:
                out.append(d)
    return out


def make_6_continuous_page_ranges(all_docs: List[Document], n_lessons: int = 6) -> List[str]:
    """Split the book into n continuous page ranges based on max page number."""
    pages = []
    for d in all_docs:
        p0 = d.metadata.get("page", None)
        if isinstance(p0, int):
            pages.append(p0 + 1)

    if not pages:
        return []

    max_page = max(pages)
    step = max(1, max_page // n_lessons)

    ranges = []
    start = 1
    for i in range(n_lessons):
        end = max_page if i == n_lessons - 1 else min(max_page, start + step - 1)
        ranges.append(f"{start}-{end}")
        start = end + 1

    return ranges[:n_lessons]


def _sample_evenly(docs: List[Document], k: int) -> List[Document]:
    """Pick ~k docs evenly across a list (to cover the whole range)."""
    if not docs:
        return []
    if len(docs) <= k:
        return docs
    idxs = np.linspace(0, len(docs) - 1, num=k, dtype=int)
    return [docs[i] for i in idxs]


def generate_page_based_curriculum(all_docs: List[Document], user_level: str, n_lessons: int = 6) -> dict:
    """
    Builds 6 lessons that follow the BOOK order (continuous page ranges).
    Beginner/Advanced: different titles/goals, but SAME page flow.
    """
    page_ranges = make_6_continuous_page_ranges(all_docs, n_lessons=n_lessons)

    lessons = []
    for i, pr in enumerate(page_ranges, start=1):
        # Take a few chunks from this range as context for naming
        range_docs = docs_in_page_range(all_docs, pr)
        sample_docs = _sample_evenly(range_docs, k=6)
        context_text = format_context_for_prompt(sample_docs)

        level_instruction = build_level_instruction(user_level)

        track_goal = (
            "Beginner track: make titles simple, focus on definitions and intuition, avoid heavy evaluation.\n"
            if user_level == "Beginner" else
            "Advanced track: make titles about trade-offs, architecture, evaluation, limitations. Assume basics.\n"
        )

        prompt = f"""
{level_instruction}
{track_goal}

You are creating ONE lesson title+goal for a course that follows the book order.

Use ONLY the context below (from the book pages in this range).
Return STRICT JSON ONLY:

{{
  "title": "string",
  "goal": "1 sentence",
  "difficulty": "Beginner" | "Advanced",
  "search_query": "string"
}}

Rules:
- Title must match the content of this page range.
- search_query must be specific (not generic), and MUST include the page range "{pr}".
- difficulty must match the track: Beginner -> "Beginner", Advanced -> "Advanced".

Context:
{context_text}
"""
        raw = call_llm(prompt).strip()
        try:
            one = json.loads(raw)
        except:
            one = {"title": f"Lesson {i}", "goal": "Study this section of the book.", "difficulty": user_level, "search_query": f"pages {pr}"}

        lessons.append({
            "lesson_id": i,
            "title": one.get("title", f"Lesson {i}"),
            "goal": one.get("goal", ""),
            "difficulty": one.get("difficulty", user_level),
            "prereq": "",
            "page_range": pr,                      # ✅ IMPORTANT
            "search_query": one.get("search_query", f"pages {pr}")  # kept for display, but lesson will use page_range
        })

    return {
        "track": user_level,
        "course_title": f"Course (Book order) — {user_level}",
        "lessons": lessons
    }


def generate_lesson_markdown(lesson_title: str, context_text: str, user_level: str) -> str:
    level_instruction = build_level_instruction(user_level)
    lesson_prompt = f"""
{level_instruction}
You are writing a lesson page for a course app.

Use ONLY the context below (from the book). Do not use outside knowledge.
If context is insufficient, say so and suggest what to search for (still book-related).

Write a structured lesson in Markdown with these sections (exact headers):
## Overview
## Key Concepts (max 5 bullets)
## Short Example (1 example)
## Common Mistakes (max 4 bullets)
## Summary (3-5 lines)

Also, when possible, reference the chunk/page like (Chunk X, page Y) inside the text.

Context:
{context_text}

Lesson title:
{lesson_title}
"""
    return call_llm(lesson_prompt).strip()


# =================== VECTOR + BM25 DB ===================

class FusionPDFVectorDB:
    def __init__(self, pdf_path: str, vector_store_path: str):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.documents: List[Document] = splitter.split_documents(documents)

        self.bm25_docs = self.documents
        self.bm25 = self._create_bm25_index(self.bm25_docs)

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

    def _create_bm25_index(self, documents: List[Document]) -> BM25Okapi:
        tokenized_docs = [doc.page_content.split() for doc in documents]
        return BM25Okapi(tokenized_docs)

    def vector_search(self, query: str, k: int = 5) -> List[Document]:
        try:
            return self.vector_store.similarity_search(query, k=k)
        except:
            return []

    def bm25_search(self, query: str, k: int = 5) -> List[Document]:
        try:
            query_tokens = query.split()
            scores = self.bm25.get_scores(query_tokens)
            sorted_idx = np.argsort(scores)[::-1]
            return [self.bm25_docs[i] for i in sorted_idx[:k] if i < len(self.bm25_docs)]
        except:
            return []

    def fusion_search(self, query: str, k: int = 5) -> List[Document]:
        try:
            top_n = max(k, 8)
            vec = self.vector_store.similarity_search_with_score(query, k=top_n)
            vector_docs = [d for d, _ in vec]

            query_tokens = query.split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            bm25_sorted = np.argsort(bm25_scores)[::-1][:top_n]
            bm25_docs = [self.bm25_docs[i] for i in bm25_sorted if i < len(self.bm25_docs)]

            rrf_k = 60
            scores: Dict[Any, Any] = {}

            def _key(d: Document):
                return (d.metadata.get("source", ""), d.metadata.get("page", -1), d.page_content[:120])

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


# =================== STREAMLIT APP ===================

st.set_page_config(page_title="Book → Course Learning System", page_icon="📚", layout="wide")
# st.markdown(APP_CSS, unsafe_allow_html=True)

st.markdown(APP_CSS, unsafe_allow_html=True)
set_bg_image("assets/bg.png")  # <-- change path if needed

# State init
if "user_level" not in st.session_state:
    st.session_state.user_level = "Beginner"
if "search_type" not in st.session_state:
    st.session_state.search_type = "Fusion"
if "k_value" not in st.session_state:
    st.session_state.k_value = 6
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"


# Course state
if "course_plans" not in st.session_state:
    st.session_state.course_plans = {"Beginner": None, "Advanced": None}
if "course_plan" not in st.session_state:
    st.session_state.course_plan = None
if "current_lesson_idx" not in st.session_state:
    st.session_state.current_lesson_idx = 0
if "lesson_cache" not in st.session_state:
    st.session_state.lesson_cache = {}  # lesson_id -> {"md":..., "sources":...}
if "checkpoint_cache" not in st.session_state:
    st.session_state.checkpoint_cache = {}  # lesson_id -> quiz_json
if "progress" not in st.session_state:
    st.session_state.progress = {}  # lesson_id -> {"done":bool, "score": int, "total": int}
if "final_exam" not in st.session_state:
    st.session_state.final_exam = None
if "final_answers" not in st.session_state:
    st.session_state.final_answers = {}
if "final_submitted" not in st.session_state:
    st.session_state.final_submitted = False

# Load DB once
if "vector_db" not in st.session_state:
    with st.spinner("Loading your book…"):
        st.session_state.vector_db = FusionPDFVectorDB(PDF_PATH, VECTOR_STORE_PATH)

vector_db: FusionPDFVectorDB = st.session_state.vector_db


# =================== HEADER ===================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">📚 Book → Course Learning System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Learn the book as a structured course: lessons → checkpoints → final exam.</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

top1, top2, top3 = st.columns([1.1, 1.2, 1.6])

with top1:
    st.session_state.user_level = st.radio(
        "Level",
        ["Beginner", "Advanced"],
        index=0 if st.session_state.user_level == "Beginner" else 1,
        horizontal=True,
    )

with top2:
    st.session_state.search_type = st.selectbox(
        "Search",
        ["Fusion", "Vector", "BM25"],
        index=["Fusion", "Vector", "BM25"].index(st.session_state.search_type),
    )

with top3:
    st.session_state.k_value = st.slider("Context chunks (k)", 3, 12, st.session_state.k_value)

st.markdown("</div>", unsafe_allow_html=True)


# =================== SIDEBAR NAV ===================

st.sidebar.title("📌 Navigation")
# page = st.sidebar.radio("Go to", ["🏠 Home", "📚 Curriculum", "📖 Lesson", "✅ Progress", "🏁 Final Exam"])

PAGES = ["🏠 Home", "📚 Curriculum", "📖 Lesson", "✅ Progress", "🏁 Final Exam"]

page = st.sidebar.radio(
    "Go to",
    PAGES,
    index=PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0,
    key="page_radio"
)

# keep session state synced
st.session_state.page = page

def retrieve(query: str, k: int) -> List[Document]:
    if st.session_state.search_type == "Fusion":
        return vector_db.fusion_search(query, k=k)
    if st.session_state.search_type == "Vector":
        return vector_db.vector_search(query, k=k)
    return vector_db.bm25_search(query, k=k)


# =================== HOME ===================

if page == "🏠 Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### מה זה?")
    st.write(
        "במקום Chatbot, זו מערכת שמלמדת את הספר כמו **קורס**.\n\n"
        "**איך זה עובד:**\n"
        "1) יוצרים Curriculum מהספר (שיעורים מסודרים)\n"
        "2) לכל שיעור יש תוכן לימודי + צ׳קפוינט קצר\n"
        "3) בסוף יש מבחן מסכם\n"
        "4) כל ההסברים והבדיקות הם **רק מתוך הספר** + Sources"
    )
    st.info("כדי להתחיל: כנסי ל־Curriculum ולחצי Generate Curriculum.")
    st.markdown("</div>", unsafe_allow_html=True)


# =================== CURRICULUM ===================

if page == "📚 Curriculum":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📚 Curriculum (סילבוס הקורס)")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("✨ Generate Curriculum from Book", key="gen_curr"):
            toc_text = extract_contents_text(PDF_PATH)
            if not toc_text.strip():
                st.error("Couldn't find the Contents pages in the PDF text.")
                st.stop()

            with st.spinner("Building course plan from Contents…"):
                plan = generate_course_plan_from_contents(
                    contents_text=toc_text,
                    user_level=st.session_state.user_level,
                    n_lessons=6
                )

            st.session_state.course_plan = plan
            st.session_state.current_lesson_idx = 0
            st.session_state.lesson_cache = {}
            st.session_state.checkpoint_cache = {}
            st.session_state.progress = {}
            st.session_state.final_exam = None
            st.session_state.final_answers = {}
            st.session_state.final_submitted = False
            st.toast("Curriculum created ✅", icon="📚")
            st.rerun()



    with c2:
        if st.button("🧹 Reset Course", key="reset_course"):
            st.session_state.course_plan = None
            st.session_state.current_lesson_idx = 0
            st.session_state.lesson_cache = {}
            st.session_state.checkpoint_cache = {}
            st.session_state.progress = {}
            st.session_state.final_exam = None
            st.session_state.final_answers = {}
            st.session_state.final_submitted = False
            st.toast("Reset done", icon="🧹")
            st.session_state.course_plans[st.session_state.user_level] = None
            st.rerun()

    plan = st.session_state.course_plans.get(st.session_state.user_level)
    if not plan:
        st.caption("אין Curriculum עדיין.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    st.markdown(f"#### 🎓 {plan.get('course_title','Course')}")
    lessons = plan.get("lessons", [])
    if not lessons:
        st.warning("לא הצלחתי לבנות שיעורים מהקונטקסט. נסי שוב.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    for i, lesson in enumerate(lessons):
        lid = lesson.get("lesson_id", i + 1)
        title = lesson.get("title", f"Lesson {lid}")
        goal = lesson.get("goal", "")
        diff = lesson.get("difficulty", "")
        done = st.session_state.progress.get(lid, {}).get("done", False)

        cols = st.columns([0.08, 0.7, 0.22])
        with cols[0]:
            st.write("✅" if done else "⬜")
        with cols[1]:
            st.markdown(f"**{lid}. {title}**  \n{goal}")
            st.caption(f"Difficulty: {diff}")
        with cols[2]:
            if st.button("Open", key=f"open_l_{lid}_{st.session_state.user_level}"):
                level = st.session_state.user_level
                st.session_state[f"lesson_selector_sidebar_{level}"] = i

                # IMPORTANT: update the actual sidebar radio value
                st.session_state["page_radio"] = "📖 Lesson"
                st.session_state.page = "📖 Lesson"

                # Optional: preselect the lesson in the sidebar selectbox
                st.session_state["lesson_selector_sidebar"] = i

                st.toast("Opening lesson 📖", icon="📖")
                st.rerun()


    st.markdown("</div>", unsafe_allow_html=True)



# =================== LESSON ===================

if page == "📖 Lesson":
    # plan = st.session_state.course_plan
    plan = st.session_state.course_plans.get(st.session_state.user_level)

    if not plan:
        st.warning("קודם צרי Curriculum בעמוד Curriculum.")
        st.stop()

    lessons = plan.get("lessons", [])
    if not lessons:
        st.warning("אין שיעורים להצגה.")
        st.stop()

    # Choose lesson
    st.sidebar.subheader("Lesson picker")
    idx = st.sidebar.selectbox(
        "Select lesson",
        list(range(len(lessons))),
        index=min(st.session_state.current_lesson_idx, len(lessons) - 1),
        format_func=lambda i: f"{lessons[i].get('lesson_id', i+1)}. {lessons[i].get('title','Lesson')}",
            key=f"lesson_selector_sidebar_{st.session_state.user_level}"


    )
    st.session_state.current_lesson_idx = idx

    lesson = lessons[idx]
    lid = lesson.get("lesson_id", idx + 1)
    title = lesson.get("title", f"Lesson {lid}")
    query = lesson.get("search_query", title)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### 📖 {lid}. {title}")
    st.caption(f"Query used for retrieval: {query}")

    # ------------------ Lesson content ------------------
    cache_key = f"{st.session_state.user_level}_{lid}"
    if cache_key not in st.session_state.lesson_cache:
        k = st.session_state.k_value
        page_range = lesson.get("page_range", "")
        range_docs = docs_in_page_range(vector_db.documents, page_range)

        # choose k chunks evenly from that page range
        context_docs = _sample_evenly(range_docs, k=k)
        context_text = format_context_for_prompt(context_docs[:min(k, 8)])



        with st.spinner("Generating lesson from the book…"):
            md = generate_lesson_markdown(title, context_text, st.session_state.user_level)

            st.session_state.lesson_cache[cache_key] = {
                "md": md,
                "sources": build_sources(context_docs, max_sources=4),
                "context_text": context_text
            }

    st.markdown(st.session_state.lesson_cache[cache_key]["md"])
    sources = st.session_state.lesson_cache[cache_key].get("sources", [])
    lesson_context = st.session_state.lesson_cache[cache_key].get("context_text", "")


    # Sources
    sources = st.session_state.lesson_cache[lid].get("sources", [])
    if sources:
        with st.expander("📌 Sources (for this lesson)", expanded=False):
            for s in sources:
                st.markdown(f"**▶ {s['label']}**")
                st.write(s["snippet"])

    st.markdown("---")
    chat_key = f"lesson_chat_{lid}"

    c_clear, c_hint = st.columns([1, 3])
    with c_clear:
        if st.button("🧹 Clear lesson chat", key=f"clear_chat_{lid}"):
            st.session_state[chat_key] = []
            st.rerun()
    with c_hint:
        st.caption("This chat answers using only the sources retrieved for this lesson.")


    st.markdown("## 💬 Ask about this lesson")

    # Make sure we have context text
    lesson_context = st.session_state.lesson_cache[lid].get("context_text", "")

    # Init chat history per lesson
    chat_key = f"lesson_chat_{lid}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []  # list of {"role": "user"/"assistant", "content": str}

    # Show chat history
    for m in st.session_state[chat_key]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # Input
    user_q = st.chat_input("Ask a question about this lesson…")
    if user_q:
        # show user message
        st.session_state[chat_key].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking with this lesson's sources…"):
                ans = answer_lesson_question(
                    user_question=user_q,
                    lesson_title=title,
                    context_text=lesson_context,
                    user_level=st.session_state.user_level
                )
                st.markdown(ans)

        st.session_state[chat_key].append({"role": "assistant", "content": ans})


    # ------------------ Checkpoint ------------------
    st.markdown("### ✅ Checkpoint (בסוף שיעור)")

    # Generate checkpoint if missing
    if lid not in st.session_state.checkpoint_cache:
        if st.button("Generate checkpoint questions", key=f"gen_cp_{lid}"):
            k = st.session_state.k_value
            context_docs = retrieve(query, k=k)
            context_text = format_context_for_prompt(context_docs)

            with st.spinner("Creating checkpoint…"):
                qz = generate_quiz_json(
                    topic=f"Checkpoint for lesson: {title}",
                    context_text=context_text,
                    user_level=st.session_state.user_level,
                    n_questions=3
                )

            st.session_state.checkpoint_cache[lid] = qz
            st.session_state.progress.setdefault(lid, {"done": False, "score": 0, "total": 0})
            st.rerun()
        else:
            st.caption("לחצי כדי ליצור שאלות קצרות שמוודאות הבנה.")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

    qz = st.session_state.checkpoint_cache.get(lid)
    questions = (qz or {}).get("questions", [])
    if not questions:
        st.warning("לא הצלחתי ליצור צ׳קפוינט. נסי להגדיל k או לשנות search type.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # Local state for answers
    if f"cp_answers_{lid}" not in st.session_state:
        st.session_state[f"cp_answers_{lid}"] = {}
    if f"cp_submitted_{lid}" not in st.session_state:
        st.session_state[f"cp_submitted_{lid}"] = False

    submitted = st.session_state[f"cp_submitted_{lid}"]

    # Render questions
    for qi, q in enumerate(questions):
        st.markdown(f"**Q{qi+1}. {q['question']}**")
        options = q["options"]
        chosen = st.radio(
            "Choose one:",
            options=list(range(4)),
            format_func=lambda i: options[i],
            key=f"cp_{lid}_q{qi}",
            disabled=submitted,
        )
        st.session_state[f"cp_answers_{lid}"][qi] = chosen
        st.divider()

    # Submit checkpoint
    if not submitted:
        if st.button("Submit checkpoint ✅", key=f"cp_submit_{lid}"):
            st.session_state[f"cp_submitted_{lid}"] = True

            score = 0
            for qi, q in enumerate(questions):
                if st.session_state[f"cp_answers_{lid}"].get(qi) == q["correct_index"]:
                    score += 1

            st.session_state.progress[lid] = {"done": True, "score": score, "total": len(questions)}
            st.toast(f"Saved! Score: {score}/{len(questions)}", icon="✅")
            st.rerun()

    # Results section
    if submitted:
        pr = st.session_state.progress.get(lid, {"score": 0, "total": len(questions)})
        st.success(f"Checkpoint completed: {pr['score']}/{pr['total']}")

        with st.expander("Review answers", expanded=False):
            for qi, q in enumerate(questions):
                correct = q["correct_index"]
                chosen = st.session_state[f"cp_answers_{lid}"].get(qi, None)
                st.write(f"Q{qi+1}: Your answer: {q['options'][chosen]} | Correct: {q['options'][correct]}")
                st.write(f"Why: {q['explanation']}")
                st.caption(f"Source: {q.get('source','')}")
                st.divider()

    # ------------------ Navigation ------------------
    st.markdown("---")
    st.markdown("### ⏭ Navigation")

    nav1, nav2, nav3 = st.columns([1, 1, 2])

    current_idx = st.session_state.current_lesson_idx
    lesson_done = st.session_state.progress.get(lid, {}).get("done", False)

    with nav1:
        prev_disabled = (current_idx == 0)
        if st.button("⬅ Previous", disabled=prev_disabled, key=f"prev_{lid}"):
            st.session_state.current_lesson_idx = max(0, current_idx - 1)
            st.rerun()

    with nav2:
        next_disabled = (current_idx >= len(lessons) - 1) or (not lesson_done)
        if st.button("Next ➡", disabled=next_disabled, key=f"next_{lid}"):
            st.session_state.current_lesson_idx = min(len(lessons) - 1, current_idx + 1)
            st.rerun()

    with nav3:
        if not lesson_done:
            st.info("Complete the checkpoint to unlock **Next** ✅")
        else:
            st.success("Checkpoint completed — Next lesson unlocked 🎉")

    st.markdown("</div>", unsafe_allow_html=True)
    # plan = st.session_state.course_plan
    # if not plan:
    #     st.warning("קודם צרי Curriculum בעמוד Curriculum.")
    #     st.stop()

    # lessons = plan.get("lessons", [])
    # if not lessons:
    #     st.warning("אין שיעורים להצגה.")
    #     st.stop()

    # # Choose lesson
    # st.sidebar.subheader("Lesson picker")
    # idx = st.sidebar.selectbox(
    #     "Select lesson",
    #     list(range(len(lessons))),
    #     index=min(st.session_state.current_lesson_idx, len(lessons) - 1),
    #     format_func=lambda i: f"{lessons[i].get('lesson_id', i+1)}. {lessons[i].get('title','Lesson')}",
    # )
    # st.session_state.current_lesson_idx = idx

    # lesson = lessons[idx]
    # lid = lesson.get("lesson_id", idx + 1)
    # title = lesson.get("title", f"Lesson {lid}")
    # query = lesson.get("search_query", title)

    # st.markdown('<div class="card">', unsafe_allow_html=True)
    # st.markdown(f"### 📖 {lid}. {title}")
    # st.caption(f"Query used for retrieval: {query}")

    # # Build / load lesson content
    # if lid not in st.session_state.lesson_cache:
    #     k = st.session_state.k_value
    #     context_docs = retrieve(query, k=k)
    #     context_text = format_context_for_prompt(context_docs)

    #     with st.spinner("Generating lesson from the book…"):
    #         md = generate_lesson_markdown(title, context_text, st.session_state.user_level)

    #     st.session_state.lesson_cache[lid] = {
    #         "md": md,
    #         "sources": build_sources(context_docs, max_sources=4)
    #     }

    # st.markdown(st.session_state.lesson_cache[lid]["md"])

    # # Sources
    # sources = st.session_state.lesson_cache[lid].get("sources", [])
    # if sources:
    #     with st.expander("📌 Sources (for this lesson)", expanded=False):
    #         for s in sources:
    #             st.markdown(f"**▶ {s['label']}**")
    #             st.write(s["snippet"])

    # st.markdown("---")
    # st.markdown("### ✅ Checkpoint (בסוף שיעור)")

    # # Generate checkpoint
    # if lid not in st.session_state.checkpoint_cache:
    #     if st.button("Generate checkpoint questions", key=f"gen_cp_{lid}"):
    #         k = st.session_state.k_value
    #         context_docs = retrieve(query, k=k)
    #         context_text = format_context_for_prompt(context_docs)

    #         with st.spinner("Creating checkpoint…"):
    #             qz = generate_quiz_json(
    #                 topic=f"Checkpoint for lesson: {title}",
    #                 context_text=context_text,
    #                 user_level=st.session_state.user_level,
    #                 n_questions=3
    #             )
    #         st.session_state.checkpoint_cache[lid] = qz
    #         st.session_state.progress.setdefault(lid, {"done": False, "score": 0, "total": 0})
    #         st.rerun()
    #     else:
    #         st.caption("לחצי כדי ליצור שאלות קצרות שמוודאות הבנה.")
    #         st.markdown("</div>", unsafe_allow_html=True)
    #         st.stop()

    # qz = st.session_state.checkpoint_cache.get(lid)
    # questions = (qz or {}).get("questions", [])
    # if not questions:
    #     st.warning("לא הצלחתי ליצור צ׳קפוינט. נסי להגדיל k או לשנות search type.")
    #     st.markdown("</div>", unsafe_allow_html=True)
    #     st.stop()

    # # Local state for answers
    # if f"cp_answers_{lid}" not in st.session_state:
    #     st.session_state[f"cp_answers_{lid}"] = {}
    # if f"cp_submitted_{lid}" not in st.session_state:
    #     st.session_state[f"cp_submitted_{lid}"] = False

    # submitted = st.session_state[f"cp_submitted_{lid}"]

    # for qi, q in enumerate(questions):
    #     st.markdown(f"**Q{qi+1}. {q['question']}**")
    #     options = q["options"]
    #     chosen = st.radio(
    #         "Choose one:",
    #         options=list(range(4)),
    #         format_func=lambda i: options[i],
    #         key=f"cp_{lid}_q{qi}",
    #         disabled=submitted,
    #     )
    #     st.session_state[f"cp_answers_{lid}"][qi] = chosen
    #     st.divider()

    # if not submitted:
    #     if st.button("Submit checkpoint ✅", key=f"cp_submit_{lid}"):
    #         st.session_state[f"cp_submitted_{lid}"] = True

    #         # Score
    #         score = 0
    #         for qi, q in enumerate(questions):
    #             if st.session_state[f"cp_answers_{lid}"].get(qi) == q["correct_index"]:
    #                 score += 1

    #         st.session_state.progress[lid] = {"done": True, "score": score, "total": len(questions)}
    #         st.toast(f"Saved! Score: {score}/{len(questions)}", icon="✅")
    #         st.rerun()

    # if submitted:
    #     pr = st.session_state.progress.get(lid, {"score": 0, "total": len(questions)})
    #     st.success(f"Checkpoint completed: {pr['score']}/{pr['total']}")
    #     with st.expander("Review answers", expanded=False):
    #         for qi, q in enumerate(questions):
    #             correct = q["correct_index"]
    #             chosen = st.session_state[f"cp_answers_{lid}"].get(qi, None)
    #             st.write(f"Q{qi+1}: Your answer: {q['options'][chosen]} | Correct: {q['options'][correct]}")
    #             st.write(f"Why: {q['explanation']}")
    #             st.caption(f"Source: {q.get('source','')}")
    #             st.divider()

    #     st.markdown("### ⏭ Navigation")

    #     nav1, nav2, nav3 = st.columns([1, 1, 2])

    #     # current lesson index
    #     current_idx = st.session_state.current_lesson_idx

    #     # Determine gating: Next is allowed only if checkpoint is done for this lesson
    #     lesson_done = st.session_state.progress.get(lid, {}).get("done", False)

    #     with nav1:
    #         prev_disabled = (current_idx == 0)
    #         if st.button("⬅ Previous", disabled=prev_disabled, key=f"prev_{lid}"):
    #             st.session_state.current_lesson_idx = max(0, current_idx - 1)
    #             st.rerun()

    #     with nav2:
    #         next_disabled = (current_idx >= len(lessons) - 1) or (not lesson_done)
    #         if st.button("Next ➡", disabled=next_disabled, key=f"next_{lid}"):
    #             st.session_state.current_lesson_idx = min(len(lessons) - 1, current_idx + 1)
    #             st.rerun()

    #     with nav3:
    #         if not lesson_done:
    #             st.info("Complete the checkpoint to unlock **Next** ✅")
    #         else:
    #             st.success("Checkpoint completed — Next lesson unlocked 🎉")

    #         st.markdown("</div>", unsafe_allow_html=True)


# =================== PROGRESS ===================

if page == "✅ Progress":
    # plan = st.session_state.course_plan
    plan = st.session_state.course_plans.get(st.session_state.user_level)

    if not plan:
        st.warning("אין קורס עדיין. צרי Curriculum.")
        st.stop()

    lessons = plan.get("lessons", [])
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✅ Progress dashboard")

    done_count = 0
    total_score = 0
    total_q = 0

    for lesson in lessons:
        lid = lesson.get("lesson_id")
        title = lesson.get("title", "")
        pr = st.session_state.progress.get(lid, {"done": False, "score": 0, "total": 0})

        if pr["done"]:
            done_count += 1
        total_score += pr["score"]
        total_q += pr["total"]

        st.markdown(f"**{lid}. {title}** — " + ("✅ Done" if pr["done"] else "⬜ Not yet"))
        if pr["total"] > 0:
            st.caption(f"Checkpoint score: {pr['score']}/{pr['total']}")
        st.divider()

    st.info(f"Lessons completed: {done_count}/{len(lessons)}")
    if total_q > 0:
        st.success(f"Overall checkpoint score: {total_score}/{total_q}")

    if done_count == len(lessons):
        st.success("🎉 כל השיעורים הושלמו! אפשר לגשת ל־Final Exam.")
    else:
        st.caption("כדי לפתוח Final Exam מומלץ להשלים את כל השיעורים.")

    st.markdown("</div>", unsafe_allow_html=True)


# =================== FINAL EXAM ===================

if page == "🏁 Final Exam":
    plan = st.session_state.course_plans.get(st.session_state.user_level)

    if not plan:
        st.warning("קודם צרי Curriculum.")
        st.stop()

    lessons = plan.get("lessons", [])
    completed = sum(1 for l in lessons if st.session_state.progress.get(l.get("lesson_id"), {}).get("done"))
    all_done = completed == len(lessons)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🏁 Final Exam")

    if not all_done:
        st.warning(f"עוד לא השלמת את כל השיעורים ({completed}/{len(lessons)}). מומלץ להשלים ואז לעשות מבחן מסכם.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    if st.session_state.final_exam is None:
        if st.button("Generate Final Exam (10 questions)"):
            # Collect context across lessons by merging top chunks for each lesson query
            k = max(4, st.session_state.k_value // 2)
            merged_docs = []
            seen = set()

            for l in lessons:
                q = l.get("search_query", l.get("title", ""))
                docs = retrieve(q, k=k)
                for d in docs:
                    key = (d.metadata.get("page", -1), d.page_content[:120])
                    if key not in seen:
                        seen.add(key)
                        merged_docs.append(d)

            # limit context size
            merged_docs = merged_docs[:12]
            context_text = format_context_for_prompt(merged_docs)

            with st.spinner("Creating final exam from the book…"):
                exam = generate_quiz_json(
                    topic="Final exam covering all lessons",
                    context_text=context_text,
                    user_level=st.session_state.user_level,
                    n_questions=10
                )

            st.session_state.final_exam = exam
            st.session_state.final_answers = {}
            st.session_state.final_submitted = False
            st.rerun()

        st.caption("לחצי כדי ליצור מבחן מסכם שמכסה את כל הקורס.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    exam = st.session_state.final_exam
    questions = exam.get("questions", [])
    if not questions:
        st.warning("המבחן יצא ריק. נסי שוב עם k גבוה יותר.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    st.markdown(f"#### {exam.get('title','Final Exam')}")
    for qi, q in enumerate(questions):
        st.markdown(f"**Q{qi+1}. {q['question']}**")
        options = q["options"]
        chosen = st.radio(
            "Choose one:",
            options=list(range(4)),
            format_func=lambda i: options[i],
            key=f"final_q{qi}",
            disabled=st.session_state.final_submitted,
        )
        st.session_state.final_answers[qi] = chosen
        st.divider()

    if not st.session_state.final_submitted:
        if st.button("Submit Final Exam ✅"):
            st.session_state.final_submitted = True
            st.rerun()

    if st.session_state.final_submitted:
        score = 0
        for qi, q in enumerate(questions):
            if st.session_state.final_answers.get(qi) == q["correct_index"]:
                score += 1
        st.success(f"Final score: {score}/{len(questions)}")

        with st.expander("Review final exam", expanded=False):
            for qi, q in enumerate(questions):
                correct = q["correct_index"]
                chosen = st.session_state.final_answers.get(qi, None)
                st.write(f"Q{qi+1}: Your answer: {q['options'][chosen]} | Correct: {q['options'][correct]}")
                st.write(f"Why: {q['explanation']}")
                st.caption(f"Source: {q.get('source','')}")
                st.divider()

    st.markdown("</div>", unsafe_allow_html=True)
