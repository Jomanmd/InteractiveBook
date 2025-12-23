import os
import json
import re
from typing import List
import uuid

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


# =================== CONFIG ===================

GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

PDF_PATH = r"C:\Users\asus\Documents\pojectAI\intelligent_interactive_systems.pdf"
VECTOR_STORE_PATH = "./vector_stores/course_book_store"


# =================== APP STYLE ===================

APP_CSS = """
<style>
/* overall spacing */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 980px; }

/* hide the default streamlit header/footer look */
header { visibility: hidden; }
footer { visibility: hidden; }

/* cards */
.card {
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 18px;
  padding: 18px 18px;
  background: rgba(255,255,255,0.04);
  box-shadow: 0 6px 18px rgba(0,0,0,0.18);
}

/* small muted text */
.muted { opacity: 0.75; font-size: 0.95rem; }

/* big title */
.hero-title { font-size: 2.0rem; font-weight: 750; margin-bottom: 6px; }
.hero-sub { font-size: 1.02rem; opacity: 0.82; margin-bottom: 2px; }

/* nicer divider */
.hr { height: 1px; background: rgba(255,255,255,0.10); margin: 14px 0; border-radius: 999px; }

/* make buttons more “app-like” */
.stButton>button {
  border-radius: 14px !important;
  padding: 0.65rem 1rem !important;
  font-weight: 650 !important;
}

/* tabs look cleaner */
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] {
  border-radius: 999px;
  padding: 10px 14px;
  background: rgba(255,255,255,0.04);
}
.stTabs [aria-selected="true"] {
  background: rgba(255,255,255,0.10) !important;
}

/* radio pills (kinda) */
div[role="radiogroup"] > label {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 999px;
  padding: 6px 12px;
  margin-right: 8px;
}
</style>
"""


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


# =================== INTENT ===================

import json
import re

def detect_intent(user_text: str) -> dict:
    """
    Returns dict:
    {
      "intent": "GREETING" | "BOOK_QUESTION" | "OTHER",
      "confidence": float,
      "is_question": bool,
      "rewrite_query": str | ""
    }
    """
    t = user_text.strip()

    # Tiny heuristic for obvious greetings / noise (fast + reliable)
    if re.fullmatch(r"(hi|hey|hello|hii+|yo|sup|thanks|thx|ok|okay|k|👍|😊|👋|\.)+", t.lower()):
        return {"intent": "GREETING", "confidence": 0.99, "is_question": False, "rewrite_query": ""}

    cls_prompt = f"""
You are an intent classifier for a course-book tutor chatbot.

Return STRICT JSON ONLY:
{{
  "intent": "GREETING" | "BOOK_QUESTION" | "OTHER",
  "confidence": 0.0-1.0,
  "is_question": true/false,
  "rewrite_query": "string (empty if not needed)"
}}

Guidelines:
- GREETING: greeting/small talk without a real info request (e.g., "hi", "thanks", "ok")
- BOOK_QUESTION: any question or request that should be answered using the book (even if broad, like "what is the most important topic?")
- OTHER: unclear / incomplete / meta, but if it still looks like a question, set is_question=true and suggest rewrite_query.

User message: "{user_text}"
"""
    raw = call_llm(cls_prompt).strip()

    # Robust JSON extraction
    try:
        return json.loads(raw)
    except:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end+1])
            except:
                pass

    # Safe fallback: treat as BOOK_QUESTION so we don't block
    return {"intent": "BOOK_QUESTION", "confidence": 0.4, "is_question": "?" in t, "rewrite_query": ""}

# =================== HELPERS ===================

def build_level_instruction(level: str) -> str:
    if level == "Beginner":
        return (
            "You are explaining to a BEGINNER student.\n"
            "- Identify the MAIN overarching topic of the book, not a single technique.\n"
            "- Use ONE simple central idea.\n"
            "- You may mention ONE example, but do not make it the main topic.\n"
            "- Use simple, non-academic language.\n"

        )
    else:
        return (
            "You are explaining to an ADVANCED student.\n"
            "- Present ONE clear central thesis.\n"
            "- Support it by synthesizing multiple concepts from the book.\n"
            "- Avoid long lists of topics.\n"
            "- Use academic terminology.\n"
            "- Emphasize relationships, trade-offs, and design implications.\n"
        )


def format_context_for_prompt(context_docs: List[Document]) -> str:
    chunks = []
    for i, d in enumerate(context_docs, start=1):
        page0 = d.metadata.get("page", None)
        page = (page0 + 1) if isinstance(page0, int) else "unknown"
        chunks.append(f"[Chunk {i} | page {page}]\n{d.page_content}")
    return "\n\n".join(chunks)

def generate_quiz_json(topic: str, context_text: str, user_level: str) -> dict:
    level_instruction = build_level_instruction(user_level)
    quiz_prompt = f"""
{level_instruction}
You are generating an interactive quiz for a Streamlit app.

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
- Make exactly 2 questions.
- Each question must have exactly 4 options.
- correct_index is 0-3.
- source must cite a chunk/page from the given context.

Context:
{context_text}

Topic (user asked):
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
    return {"title": "Quiz", "questions": []}



# =================== VECTOR + BM25 DB ===================

class FusionPDFVectorDB:
    def __init__(self, pdf_path: str, vector_store_path: str):
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split to chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.documents: List[Document] = splitter.split_documents(documents)

        # BM25 (always build on full chunks)
        self.bm25_docs = self.documents
        self.bm25 = self._create_bm25_index(self.bm25_docs)

        # Vector store load/create
        try:
            if os.path.exists(vector_store_path):
                print("Loading existing vector store...")
                self.vector_store = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                # sanity check
                test_emb = self.embeddings.embed_query("test")
                self.vector_store.index.search(np.array([test_emb]), k=1)
                print("Vector store loaded successfully!")
            else:
                print("Creating new vector store from PDF...")
                self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
                os.makedirs(vector_store_path, exist_ok=True)
                self.vector_store.save_local(vector_store_path)
                print("New vector store created and saved!")

        except Exception as e:
            print("Error initializing vector store:", e)
            print("Rebuilding vector store from scratch...")
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
            os.makedirs(vector_store_path, exist_ok=True)
            self.vector_store.save_local(vector_store_path)

    def _create_bm25_index(self, documents: List[Document]) -> BM25Okapi:
        tokenized_docs = [doc.page_content.split() for doc in documents]
        return BM25Okapi(tokenized_docs)

    def vector_search(self, query: str, k: int = 5) -> List[Document]:
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            print("Error vector_search:", e)
            return []

    def bm25_search(self, query: str, k: int = 5) -> List[Document]:
        try:
            query_tokens = query.split()
            scores = self.bm25.get_scores(query_tokens)
            sorted_idx = np.argsort(scores)[::-1]
            return [self.bm25_docs[i] for i in sorted_idx[:k] if i < len(self.bm25_docs)]
        except Exception as e:
            print("Error bm25_search:", e)
            return []

    def fusion_search(self, query: str, k: int = 5) -> List[Document]:
        """
        RRF Fusion of vector + BM25 by rank (stable)
        """
        try:
            top_n = max(k, 8)

            vec = self.vector_store.similarity_search_with_score(query, k=top_n)
            vector_docs = [d for d, _ in vec]

            query_tokens = query.split()
            bm25_scores = self.bm25.get_scores(query_tokens)
            bm25_sorted = np.argsort(bm25_scores)[::-1][:top_n]
            bm25_docs = [self.bm25_docs[i] for i in bm25_sorted if i < len(self.bm25_docs)]

            rrf_k = 60
            scores = {}

            def _key(d: Document):
                return (
                    d.metadata.get("source", ""),
                    d.metadata.get("page", -1),
                    d.page_content[:120],
                )

            for rank, d in enumerate(vector_docs, start=1):
                scores.setdefault(_key(d), {"doc": d, "score": 0.0})
                scores[_key(d)]["score"] += 1.0 / (rrf_k + rank)

            for rank, d in enumerate(bm25_docs, start=1):
                scores.setdefault(_key(d), {"doc": d, "score": 0.0})
                scores[_key(d)]["score"] += 1.0 / (rrf_k + rank)

            merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
            return [m["doc"] for m in merged[:k]]

        except Exception as e:
            print("Error fusion_search:", e)
            return self.vector_search(query, k=k)


# =================== STREAMLIT APP ===================

st.set_page_config(page_title="Book Knowledge Bot", page_icon="📚", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

# State init
if "user_level" not in st.session_state:
    st.session_state.user_level = "Beginner"
if "search_type" not in st.session_state:
    st.session_state.search_type = "Fusion"
if "k_value" not in st.session_state:
    st.session_state.k_value = 5

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_context_docs" not in st.session_state:
    st.session_state.last_context_docs = []

if "pending_user_msg" not in st.session_state:
    st.session_state.pending_user_msg = None

# Quiz state
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_topic" not in st.session_state:
    st.session_state.quiz_topic = ""
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

# Load DB once
if "vector_db" not in st.session_state:
    with st.spinner("Loading your book…"):
        st.session_state.vector_db = FusionPDFVectorDB(PDF_PATH, VECTOR_STORE_PATH)

vector_db: FusionPDFVectorDB = st.session_state.vector_db


# ======= APP HEADER (always visible) =======

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="hero-title">📚 Book Knowledge Bot</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Ask questions and get answers only from your course book — or practice with quizzes.</div>', unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

top1, top2, top3 = st.columns([1.2, 1.2, 1.6])

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
    st.session_state.k_value = st.slider("Context chunks (k)", 1, 10, st.session_state.k_value)

st.markdown("</div>", unsafe_allow_html=True)


# ======= TABS (app pages) =======

tab_welcome, tab_chat, tab_quiz = st.tabs(["🏠 Welcome", "💬 Chat", "📝 Quiz"])


# ---------- WELCOME ----------
with tab_welcome:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👋 Hi!")
    st.write(
        "This app reads your PDF book, builds a search index, then answers **only from the book**.\n\n"
        "**Use it for:**\n"
        "- understanding chapters\n"
        "- definitions & explanations\n"
        "- exam practice quizzes"
    )
    st.markdown('<div class="muted">Tip: You can change your level anytime at the top.</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("✅ Book-only answers")
    with c2:
        st.info("🧠 Beginner / Advanced")
    with c3:
        st.warning("📝 Interactive quizzes")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------- CHAT ----------
with tab_chat:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask the book")

    # init chat history (messages can include sources)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # each item: {"role","content","sources"?}

    if "pending_user_msg" not in st.session_state:
        st.session_state.pending_user_msg = None

    # helper: build sources list from retrieved docs
    def build_sources(context_docs: List[Document], max_sources: int = 3):
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

    # A) Process pending message FIRST (so input stays at bottom after rerun)
    if st.session_state.pending_user_msg:
        prompt = st.session_state.pending_user_msg
        st.session_state.pending_user_msg = None  # consume

        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Detect intent (use your LLM intent)
        intent_obj = detect_intent(prompt)
        intent = intent_obj.get("intent", "BOOK_QUESTION")
        conf = float(intent_obj.get("confidence", 0.0))

        # GREETING (good UX, not "be more specific")
        if intent == "GREETING" and conf >= 0.8:
            reply = (
                "Hi! 👋\n\n"
                "Ask me *any* question and I’ll answer **only from the book**.\n\n"
                "Examples:\n"
                "- What is the book about?\n"
                "- Explain K-Means (from the book)\n"
                "- Summarize chapter 4\n"
            )
            st.session_state.chat_history.append({"role": "assistant", "content": reply, "sources": []})
            st.rerun()

        # OTHER but not a question → gentle nudge
        if intent == "OTHER" and not intent_obj.get("is_question", False):
            rq = (intent_obj.get("rewrite_query") or "").strip()
            msg = "I’m here 🙂 What would you like to know from the book?"
            if rq:
                msg += f"\n\nIf you want, try: **{rq}**"
            st.session_state.chat_history.append({"role": "assistant", "content": msg, "sources": []})
            st.rerun()

        # Otherwise: treat as BOOK_QUESTION (even if broad)
        k = st.session_state.k_value

        if st.session_state.search_type == "Fusion":
            context_docs = vector_db.fusion_search(prompt, k=k)
            search_method = "Fusion"
        elif st.session_state.search_type == "Vector":
            context_docs = vector_db.vector_search(prompt, k=k)
            search_method = "Vector"
        else:
            context_docs = vector_db.bm25_search(prompt, k=k)
            search_method = "BM25"

        context_text = format_context_for_prompt(context_docs)
        level_instruction = build_level_instruction(st.session_state.user_level)

        explain_prompt = f"""
{level_instruction}
You MUST base your answer ONLY on the information in the context below,
which comes from the book. If the context is not enough to answer,
say that clearly and do not invent information.

When possible, mention which [Chunk # | page #] you used.

(Found using {search_method} search)

Context:
{context_text}

Question: {prompt}

Answer:
"""

        with st.spinner("Thinking…"):
            response = call_llm(explain_prompt)

        # Save assistant message WITH sources (so each answer has its own expander)
        sources = build_sources(context_docs, max_sources=4)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })
        st.rerun()

    # B) Render chat history (sources appear under EACH assistant answer)
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

            if msg["role"] == "assistant":
                sources = msg.get("sources", [])
                if sources:
                    with st.expander("📌 Sources (click to open)", expanded=False):
                        for s in sources:
                            st.markdown(f"**▶ {s['label']}**")
                            st.write(s["snippet"])

    # C) Actions under chat
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🧹 Clear chat", key="chat_clear_btn"):
            st.session_state.chat_history = []
            st.rerun()

    with c2:
        if st.button("📝 Create quiz from last question", key="chat_make_quiz_btn"):
            last_user = next((m["content"] for m in reversed(st.session_state.chat_history) if m["role"] == "user"), "")
            if not last_user:
                st.toast("Ask a question first 🙂", icon="💬")
            else:
                st.session_state.quiz_topic = last_user
                st.session_state.quiz_data = None
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.toast("Now open the **Quiz** tab ✨", icon="📝")

    # D) Input LAST (always at bottom)
    new_prompt = st.chat_input("Ask anything about the book…", key="chat_input_main")
    if new_prompt:
        st.session_state.pending_user_msg = new_prompt
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ---------- QUIZ ----------
with tab_quiz:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📝 Quiz mode")

    if st.session_state.quiz_topic:
        st.caption(f"Quiz topic from Chat: {st.session_state.quiz_topic}")
    else:
        st.caption("No quiz topic yet. Create a quiz from a Chat question.")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Generate quiz"):
            topic = (st.session_state.quiz_topic or "").strip()

            if not topic:
                st.warning("Go to the Chat tab, ask a question, then click “Create quiz from this question”.")
            else:
                k = st.session_state.k_value
                if st.session_state.search_type == "Fusion":
                    context_docs = vector_db.fusion_search(topic, k=k)
                elif st.session_state.search_type == "Vector":
                    context_docs = vector_db.vector_search(topic, k=k)
                else:
                    context_docs = vector_db.bm25_search(topic, k=k)

                context_text = format_context_for_prompt(context_docs)
                st.session_state.quiz_data = generate_quiz_json(
                    topic, context_text, st.session_state.user_level
                )
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()


    with c2:
        if st.button("Try again"):
            st.session_state.quiz_data = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.rerun()

    with c3:
        if st.button("Back to Chat (hint)"):
            st.info("Just click the **Chat** tab at the top 🙂")

    quiz_data = st.session_state.quiz_data
    if not quiz_data:
        st.markdown('<div class="muted">Generate a quiz to see interactive questions here.</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    st.markdown(f"#### {quiz_data.get('title', 'Quiz')}")
    questions = quiz_data.get("questions", [])
    if not questions:
        st.warning("Couldn’t generate quiz from the context. Try a more specific topic.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    for qi, q in enumerate(questions):
        st.markdown(f"**Q{qi+1}. {q['question']}**")
        options = q["options"]

        chosen = st.radio(
            "Choose one:",
            options=list(range(4)),
            format_func=lambda i: options[i],
            key=f"quiz_q{qi}",
            disabled=st.session_state.quiz_submitted,
        )
        st.session_state.quiz_answers[qi] = chosen
        st.divider()

    if not st.session_state.quiz_submitted:
        if st.button("Done ✅"):
            st.session_state.quiz_submitted = True
            st.rerun()

    if st.session_state.quiz_submitted:
        st.markdown("### Results")
        for qi, q in enumerate(questions):
            correct = q["correct_index"]
            chosen = st.session_state.quiz_answers.get(qi, None)

            if chosen == correct:
                st.success(f"Q{qi+1}: Correct ✅")
            else:
                st.error(f"Q{qi+1}: Wrong ❌")

            st.write(f"**Correct answer:** {q['options'][correct]}")
            st.write(f"**Why:** {q['explanation']}")
            st.caption(f"Source: {q.get('source', '')}")
            st.divider()

    st.markdown("</div>", unsafe_allow_html=True)
