import os
import json
import base64
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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


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


# =================== CONFIG ===================

# ⚠️ Prefer env var / st.secrets instead of hardcoding.
# os.environ["GROQ_API_KEY"] = "PASTE_YOUR_KEY"

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

PDF_PATH = "intelligent_interactive_systems.pdf"
VECTOR_STORE_PATH = "./vector_stores/course_book_store"


# =================== APP STYLE ===================

APP_CSS = """
<style>
.block-container { 
  padding-top: 1.2rem; 
  padding-bottom: 2rem; 
  max-width: 1050px; 
}

header { visibility: hidden; }
footer { visibility: hidden; }

/* Make text readable on light theme */
html, body, [class*="css"] { color: #000000 !important; }
h1, h2, h3, h4, h5, h6 { color: #000000 !important; }
p, span, label, div, li { color: #000000 !important; }

/* Cards */
.card {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 18px;
  padding: 18px;
  background: rgba(255,255,255,0.85);
  box-shadow: 0 6px 18px rgba(0,0,0,0.12);
}

/* Header */
.hero-title { font-size: 2.0rem; font-weight: 750; margin-bottom: 6px; }
.hero-sub { font-size: 1.02rem; opacity: 0.8; margin-bottom: 2px; }

.hr { height: 1px; background: rgba(0,0,0,0.15); margin: 14px 0; border-radius: 999px; }

/* ===== Buttons: force readable text for ALL button variants ===== */

/* Streamlit base button */
.stButton > button,
.stButton > button p,
.stButton > button span {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important; /* fixes some themes */
  font-weight: 700 !important;
}

/* Ensure background is dark for visibility */
.stButton > button {
  background: #111827 !important;   /* dark navy */
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
  padding: 0.65rem 1.1rem !important;
}

/* Hover */
.stButton > button:hover {
  background: #1f2937 !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

/* Also target Streamlit's newer data-testid buttons */
button[data-testid="baseButton-secondary"],
button[data-testid="baseButton-primary"],
button[data-testid="baseButton-tertiary"],
button[data-testid="baseButton-minimal"],
button[data-testid="baseButton"] {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  background: #111827 !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 14px !important;
  font-weight: 700 !important;
}

/* If Streamlit wraps text inside these buttons */
button[data-testid^="baseButton"] * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

/* ===== Sidebar styling ===== */
section[data-testid="stSidebar"] {
  background: #0b1220 !important;
}

section[data-testid="stSidebar"] * {
  color: #ffffff !important;
}

/* Sidebar title */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
  color: #ffffff !important;
  font-weight: 800 !important;
}

/* Radio items container */
section[data-testid="stSidebar"] div[role="radiogroup"] {
  gap: 10px !important;
}

/* Each radio row */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 14px !important;
  padding: 10px 12px !important;
  margin: 6px 0 !important;
  transition: all 0.15s ease-in-out;
}

/* Hover effect */
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  background: rgba(255,255,255,0.12) !important;
  transform: translateX(2px);
}

/* Selected item (Streamlit marks checked radio input) */
section[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div {
  color: #ffffff !important;
  font-weight: 800 !important;
}

/* Make the circle radio nicer */
section[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"] {
  accent-color: #22c55e !important; /* green */
}

/* Sidebar selectbox background */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.08) !important;
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}

/* ===== Sidebar Lesson Dropdown (Selectbox) ===== */

/* Closed selectbox (selected value) */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.10) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 14px !important;
}

/* Selected text inside dropdown */
section[data-testid="stSidebar"] div[data-baseweb="select"] span {
  color: #ffffff !important;
  font-weight: 700 !important;
}

/* Dropdown arrow */
section[data-testid="stSidebar"] svg {
  fill: #ffffff !important;
}

/* Open dropdown menu */
section[data-testid="stSidebar"] ul[role="listbox"] {
  background: #0b1220 !important;
  border-radius: 14px !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
}

/* Dropdown options */
section[data-testid="stSidebar"] li[role="option"] {
  color: #ffffff !important;
  font-weight: 600 !important;
  padding: 10px 14px !important;
}

/* Hover option */
section[data-testid="stSidebar"] li[role="option"]:hover {
  background: rgba(255,255,255,0.15) !important;
}

/* Selected option */
section[data-testid="stSidebar"] li[aria-selected="true"] {
  background: rgba(34,197,94,0.25) !important; /* green highlight */
  font-weight: 800 !important;
}

/* ===== FORCE sidebar selectbox text visibility (strong override) ===== */
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

/* Selected value container */
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.12) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  border-radius: 14px !important;
}

/* Some Streamlit versions render the value in an input */
section[data-testid="stSidebar"] div[data-baseweb="select"] input {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
  caret-color: #ffffff !important;
}

/* Dropdown arrow */
section[data-testid="stSidebar"] div[data-baseweb="select"] svg {
  fill: #ffffff !important;
}

/* Open menu */
div[role="listbox"] {
  background: #0b1220 !important;
  border: 1px solid rgba(255,255,255,0.15) !important;
  border-radius: 14px !important;
}

/* Options */
div[role="option"] {
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

/* Hover + selected */
div[role="option"]:hover {
  background: rgba(255,255,255,0.15) !important;
}
div[aria-selected="true"][role="option"] {
  background: rgba(34,197,94,0.25) !important;
}


</style>
"""


# =================== LLM ===================

def call_llm(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "Error: GROQ_API_KEY not set. Put it in environment variables (or st.secrets)."

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
    return (
        "You are teaching an ADVANCED student.\n"
        "- Use academic terminology.\n"
        "- Emphasize relationships, trade-offs, design implications.\n"
        "- Keep it structured and precise.\n"
    )

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
        sources.append({"label": f"Chunk {i} — page {page}", "page": page, "snippet": snippet})
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

def generate_course_plan_json(context_text: str, user_level: str, n_lessons: int = 6) -> dict:
    level_instruction = build_level_instruction(user_level)
    plan_prompt = f"""
{level_instruction}
You are building a mini-course syllabus from a course book.

Use ONLY the context below (from the book). Do not use outside knowledge.
Return STRICT JSON ONLY (no markdown).

Schema:
{{
  "course_title": "string",
  "lessons": [
    {{
      "lesson_id": 1,
      "title": "string",
      "goal": "1 sentence",
      "search_query": "string",
      "prereq": "string",
      "difficulty": "Beginner" | "Intermediate" | "Advanced"
    }}
  ]
}}

Rules:
- Create exactly {n_lessons} lessons.
- The lessons must progress from easier to harder.
- search_query should be short and retrieve the right parts from the book.
- Keep titles concise and course-like.

Context:
{context_text}
"""
    raw = call_llm(plan_prompt).strip()
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
    return {"course_title": "Course", "lessons": []}

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
    return call_llm(lesson_prompt)


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
st.markdown(APP_CSS, unsafe_allow_html=True)
set_bg_image("assets/bg.png")

# State init
if "user_level" not in st.session_state:
    st.session_state.user_level = "Beginner"
if "search_type" not in st.session_state:
    st.session_state.search_type = "Fusion"
if "k_value" not in st.session_state:
    st.session_state.k_value = 6

# Course state
if "course_plan" not in st.session_state:
    st.session_state.course_plan = None
if "current_lesson_idx" not in st.session_state:
    st.session_state.current_lesson_idx = 0
if "lesson_cache" not in st.session_state:
    st.session_state.lesson_cache = {}
if "checkpoint_cache" not in st.session_state:
    st.session_state.checkpoint_cache = {}
if "progress" not in st.session_state:
    st.session_state.progress = {}
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


# Retrieval helper
def retrieve(query: str, k: int) -> List[Document]:
    if st.session_state.search_type == "Fusion":
        return vector_db.fusion_search(query, k=k)
    if st.session_state.search_type == "Vector":
        return vector_db.vector_search(query, k=k)
    return vector_db.bm25_search(query, k=k)


# Chatbot answer (now retrieve exists ✅)
def answer_lesson_question(user_question: str, lesson_title: str, lesson_query: str, user_level: str, k: int):
    rag_query = f"{lesson_query} | {user_question}"
    context_docs = retrieve(rag_query, k=k)
    context_text = format_context_for_prompt(context_docs)

    level_instruction = build_level_instruction(user_level)

    prompt = f"""
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
    answer = call_llm(prompt)
    sources = build_sources(context_docs, max_sources=4)
    return answer, sources


# =================== HEADER ===================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">📚 Book → Course Learning System</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Learn the book as a structured course: lessons → checkpoints → final exam.</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# =================== SIDEBAR NAV ===================

st.sidebar.title("📌 Navigation")
# Apply pending navigation change BEFORE the radio widget is created
if "pending_nav_radio" in st.session_state:
    st.session_state["nav_radio"] = st.session_state.pop("pending_nav_radio")

page = st.sidebar.radio("Go to", ["🏠 Home", "📚 Curriculum", "📖 Lesson", "✅ Progress", "🏁 Final Exam"], key="nav_radio")


# =================== HOME ===================

# =================== HOME ===================

if page == "🏠 Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("## 👋 Hi! I’m your Interactive Book Tutor")
    st.write(
        "I’ll help you learn this book like a **real course** — step by step, with lessons, quick checkpoints, and a final exam.\n\n"
        "**Important:** I only use information from *your book* (no outside sources). "
        "Whenever possible, I’ll show you **sources** so you can verify everything."
    )

    st.markdown("### How we’ll learn (simple flow)")
    st.markdown(
        """
        1. **Generate a Curriculum** from the book (a structured set of lessons)  
        2. Open a lesson → I generate a clear explanation *from the book*  
        3. Ask me questions in the lesson chatbot (I answer using book context + sources)  
        4. Take a short **Checkpoint** to confirm understanding  
        5. After all lessons → unlock the **Final Exam**
        """
    )

    st.markdown("### Choose your learning style")

    top1, top2, top3 = st.columns([1.1, 1.2, 1.6])
    with top1:
         st.session_state.user_level = st.radio(
        "Level",
        ["Beginner", "Advanced"],
        index=0 if st.session_state.user_level == "Beginner" else 1,
        horizontal=True,
        key="level_radio_home"
    )

    with top2:
        st.session_state.search_type = st.selectbox(
        "Search mode",
        ["Fusion", "Vector", "BM25"],
        index=["Fusion", "Vector", "BM25"].index(st.session_state.search_type),
        key="search_select_home"
    )

    with top3:
        st.session_state.k_value = st.slider(
        "Context chunks (k)",
        3, 12,
        st.session_state.k_value,
        key="k_slider_home"
    )
        
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🎓 Level**")
        st.caption("Beginner = simple + friendly.\n\n Advanced = more technical + precise.")
    with c2:
        st.markdown("**🔎 Search mode**")
        st.caption("Fusion is best (Vector + BM25).\n\n If results feel off, try BM25.")
    with c3:
        st.markdown("**🧠 Context (k)**")
        st.caption("Higher k = more book context (better answers, slower).")
        st.caption("**Tip**: If a lesson feels too generic, increase **k** to 8–10.")

    st.markdown("---")
    st.markdown("### 🚀 Start here")
    st.write("Go to **Curriculum** and click **Generate Curriculum from Book**. Then open Lesson 1 and I’ll guide you from there.")
   
    # Navigation button (Home -> Curriculum)
    nav_col1, nav_col2 = st.columns([1, 2])
    with nav_col1:
        if st.button("📚 Go to Curriculum", key="home_go_curriculum"):
          st.session_state["pending_nav_radio"] = "📚 Curriculum"
          st.rerun()
    with nav_col2:
        st.caption("Next step: generate your course plan from the book.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)



# =================== CURRICULUM ===================

if page == "📚 Curriculum":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📚 Curriculum (סילבוס הקורס)")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("✨ Generate Curriculum from Book", key="gen_curr"):
            outline_context_docs = vector_db.documents[:10]
            outline_context = format_context_for_prompt(outline_context_docs)

            with st.spinner("Building course plan from the book…"):
                plan = generate_course_plan_json(outline_context, st.session_state.user_level, n_lessons=6)

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
            st.rerun()

    plan = st.session_state.course_plan
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
            if st.button("Open", key=f"open_l_{lid}"):
                st.session_state.current_lesson_idx = i
                st.toast("Go to Lesson page 📖", icon="📖")
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# =================== LESSON ===================

if page == "📖 Lesson":
    plan = st.session_state.course_plan
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
        key="lesson_selector_sidebar"
    )
    st.session_state.current_lesson_idx = idx

    lesson = lessons[idx]
    lid = lesson.get("lesson_id", idx + 1)
    title = lesson.get("title", f"Lesson {lid}")
    query = lesson.get("search_query", title)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"### 📖 {lid}. {title}")
    st.caption(f"Query used for retrieval: {query}")

    # Lesson content
    if lid not in st.session_state.lesson_cache:
        k = st.session_state.k_value
        context_docs = retrieve(query, k=k)
        context_text = format_context_for_prompt(context_docs)

        with st.spinner("Generating lesson from the book…"):
            md = generate_lesson_markdown(title, context_text, st.session_state.user_level)

        st.session_state.lesson_cache[lid] = {
            "md": md,
            "sources": build_sources(context_docs, max_sources=4),
        }

    st.markdown(st.session_state.lesson_cache[lid]["md"])

    # Lesson sources
    sources = st.session_state.lesson_cache[lid].get("sources", [])
    if sources:
        with st.expander("📌 Sources (for this lesson)", expanded=False):
            for s in sources:
                st.markdown(f"**▶ {s['label']}**")
                st.write(s["snippet"])

    # ------------------ Chatbot ------------------
    st.markdown("---")
    st.markdown("### 💬 Lesson Chatbot (Ask about this lesson)")

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

    user_q = st.chat_input("Ask a question about this lesson…", key=f"chat_input_{lid}")

    if user_q:
        st.session_state[chat_key].append({"role": "user", "content": user_q})

        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking from the book…"):
                ans, srcs = answer_lesson_question(
                    user_question=user_q,
                    lesson_title=title,
                    lesson_query=query,
                    user_level=st.session_state.user_level,
                    k=st.session_state.k_value
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

    # ------------------ Checkpoint ------------------
    st.markdown("---")
    st.markdown("### ✅ Checkpoint (בסוף שיעור)")

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

    if f"cp_answers_{lid}" not in st.session_state:
        st.session_state[f"cp_answers_{lid}"] = {}
    if f"cp_submitted_{lid}" not in st.session_state:
        st.session_state[f"cp_submitted_{lid}"] = False

    submitted = st.session_state[f"cp_submitted_{lid}"]

    for qi, q in enumerate(questions):
        st.markdown(f"**Q{qi+1}. {q['question']}**")
        options = q["options"]
        chosen = st.radio(
            "Choose one:",
            options=list(range(4)),
            format_func=lambda i: options[i],
            key=f"cp_{lid}_q{qi}",   # now unique because Lesson page is not duplicated
            disabled=submitted,
        )
        st.session_state[f"cp_answers_{lid}"][qi] = chosen
        st.divider()

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

    # Navigation
    st.markdown("---")
    st.markdown("### ⏭ Navigation")

    nav1, nav2, nav3 = st.columns([1, 1, 2])
    current_idx = st.session_state.current_lesson_idx
    lesson_done = st.session_state.progress.get(lid, {}).get("done", False)

    with nav1:
        if st.button("⬅ Previous", disabled=(current_idx == 0), key=f"prev_{lid}"):
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


# =================== PROGRESS ===================

if page == "✅ Progress":
    plan = st.session_state.course_plan
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
    plan = st.session_state.course_plan
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
        if st.button("Generate Final Exam (10 questions)", key="gen_final_exam"):
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
        if st.button("Submit Final Exam ✅", key="submit_final_exam"):
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