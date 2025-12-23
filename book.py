import os
import json
from typing import List

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


# =================== API KEY ===================

def get_groq_api_key() -> str:
    # 1) Streamlit secrets
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass

    # 2) Environment variable
    return os.getenv("GROQ_API_KEY", "").strip()


# =================== LLM CALL ===================

def call_llm(prompt: str) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        return "Error: GROQ_API_KEY not set. Set it in env or .streamlit/secrets.toml"

    try:
        resp = requests.post(
            GROQ_ENDPOINT,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=60,
        )
        data = resp.json()

        if "error" in data:
            return f"LLM Error: {data['error'].get('message', 'Unknown error')}"

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error calling Groq API: {e}"


# =================== INTENT DETECTION ===================

def detect_intent(user_text: str) -> str:
    """
    Returns: GREETING, BOOK_QUESTION, OTHER
    """
    cls_prompt = f"""
You are an intent classifier for a course-book tutor chatbot.
Classify the user's message into exactly ONE label:

- GREETING: hello/hi/thanks/okay/small talk with no real question
- BOOK_QUESTION: a question/request that should be answered using the book context
- OTHER: unclear/incomplete/meta

User message: "{user_text}"

Reply with ONLY one label: GREETING or BOOK_QUESTION or OTHER.
"""
    label = call_llm(cls_prompt).strip().upper()
    return label if label in {"GREETING", "BOOK_QUESTION", "OTHER"} else "OTHER"


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

        # BM25 always on all chunks
        self.bm25_docs = self.documents
        self.bm25 = self._create_bm25_index(self.bm25_docs)

        try:
            if os.path.exists(vector_store_path):
                self.vector_store = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                # sanity check
                test = self.embeddings.embed_query("test")
                self.vector_store.index.search(np.array([test]), k=1)
            else:
                self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
                os.makedirs(vector_store_path, exist_ok=True)
                self.vector_store.save_local(vector_store_path)

        except Exception:
            # rebuild if store is corrupted
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
            os.makedirs(vector_store_path, exist_ok=True)
            self.vector_store.save_local(vector_store_path)

    def _create_bm25_index(self, docs: List[Document]) -> BM25Okapi:
        tokenized = [d.page_content.split() for d in docs]
        return BM25Okapi(tokenized)

    def vector_search(self, query: str, k: int = 5) -> List[Document]:
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception:
            return []

    def bm25_search(self, query: str, k: int = 5) -> List[Document]:
        try:
            tokens = query.split()
            scores = self.bm25.get_scores(tokens)
            order = np.argsort(scores)[::-1]
            return [self.bm25_docs[i] for i in order[:k] if i < len(self.bm25_docs)]
        except Exception:
            return []

    def fusion_search(self, query: str, k: int = 5) -> List[Document]:
        """
        RRF fusion of vector + BM25 by rank.
        """
        try:
            top_n = max(k, 8)

            vec = self.vector_store.similarity_search_with_score(query, k=top_n)
            vec_docs = [d for d, _ in vec]

            tokens = query.split()
            bm25_scores = self.bm25.get_scores(tokens)
            bm25_sorted = np.argsort(bm25_scores)[::-1][:top_n]
            bm_docs = [self.bm25_docs[i] for i in bm25_sorted if i < len(self.bm25_docs)]

            rrf_k = 60
            scores = {}

            def key(d: Document):
                return (d.metadata.get("source", ""), d.metadata.get("page", -1), d.page_content[:120])

            for rank, d in enumerate(vec_docs, start=1):
                scores.setdefault(key(d), {"doc": d, "score": 0.0})
                scores[key(d)]["score"] += 1.0 / (rrf_k + rank)

            for rank, d in enumerate(bm_docs, start=1):
                scores.setdefault(key(d), {"doc": d, "score": 0.0})
                scores[key(d)]["score"] += 1.0 / (rrf_k + rank)

            merged = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
            return [m["doc"] for m in merged[:k]]

        except Exception:
            return self.vector_search(query, k=k)


# =================== PROMPT HELPERS ===================

def build_level_instruction(level: str) -> str:
    if level == "Beginner":
        return (
            "You are explaining to a BEGINNER student. "
            "Use simple language, short sentences, and clear examples. "
            "Avoid heavy math and advanced jargon. "
        )
    return (
        "You are explaining to an ADVANCED student who already knows the basics. "
        "You may use technical terms from the book, give more detailed arguments, "
        "and connect multiple concepts together. "
    )

def format_context_for_prompt(context_docs: List[Document]) -> str:
    chunks = []
    for i, doc in enumerate(context_docs, start=1):
        page0 = doc.metadata.get("page", None)
        page = (page0 + 1) if isinstance(page0, int) else "unknown"
        chunks.append(f"[Chunk {i} | page {page}]\n{doc.page_content}")
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
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except Exception:
                pass
        return {"title": "Quiz", "questions": []}


# =================== STREAMLIT APP ===================

st.set_page_config(page_title="Book Knowledge Bot", page_icon="📚", layout="wide")

# Session state init
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "fusion_search_type" not in st.session_state:
    st.session_state.fusion_search_type = "Fusion"

if "user_level" not in st.session_state:
    st.session_state.user_level = "Beginner"

if "mode" not in st.session_state:
    st.session_state.mode = "Explain"

# Quiz persistent state
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_id" not in st.session_state:
    st.session_state.quiz_id = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False
if "quiz_feedback" not in st.session_state:
    st.session_state.quiz_feedback = None
if "quiz_feedback_for_id" not in st.session_state:
    st.session_state.quiz_feedback_for_id = None


# Init DB
if "vector_db" not in st.session_state:
    with st.spinner("Loading book and vector store..."):
        st.session_state.vector_db = FusionPDFVectorDB(PDF_PATH, VECTOR_STORE_PATH)

vector_db: FusionPDFVectorDB = st.session_state.vector_db


# ---------- UI Header ----------
st.title("📚 Book Knowledge Bot")
st.write("Ask me anything about the course book. The bot answers ONLY from the book.")

st.session_state.user_level = st.radio(
    "Select your level:",
    ["Beginner", "Advanced"],
    index=0 if st.session_state.user_level == "Beginner" else 1,
    horizontal=True,
)

st.session_state.mode = st.radio(
    "Mode:",
    ["Explain", "Quiz"],
    index=0 if st.session_state.mode == "Explain" else 1,
    horizontal=True,
)

# Chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Controls row
col0, col1, col2 = st.columns([1, 1, 3])

with col0:
    st.selectbox("RAG Type", ["Fusion RAG"], index=0, label_visibility="collapsed")

with col1:
    st.session_state.fusion_search_type = st.selectbox(
        "Search Type",
        ["Fusion", "Vector", "BM25"],
        index=["Fusion", "Vector", "BM25"].index(st.session_state.fusion_search_type),
        label_visibility="collapsed",
    )

with col2:
    prompt = st.chat_input("What would you like to know about the book?")


# ---------- Handle user message ----------
if prompt:
    import uuid
    message_id = str(uuid.uuid4())

    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    intent = detect_intent(prompt)

    if intent == "GREETING":
        reply = "Hi! 👋\n\nI’m your interactive tutor for the course book.\nAsk a concept/question, or switch to **Quiz** mode to practice."
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.stop()

    if intent == "OTHER":
        reply = "I can help, but I need a bit more detail.\n\nTry asking a specific question about a concept from the book, or say what topic you’re on."
        with st.chat_message("assistant"):
            st.write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.stop()

    # Retrieve context
    k_value = st.session_state.get("k_value", 5)

    if st.session_state.fusion_search_type == "Fusion":
        context_docs = vector_db.fusion_search(prompt, k=k_value)
        search_method = "Fusion Search (Vector + BM25)"
    elif st.session_state.fusion_search_type == "Vector":
        context_docs = vector_db.vector_search(prompt, k=k_value)
        search_method = "Vector Search"
    else:
        context_docs = vector_db.bm25_search(prompt, k=k_value)
        search_method = "BM25 Keyword Search"

    context_text = format_context_for_prompt(context_docs)
    level_instruction = build_level_instruction(st.session_state.user_level)

    # QUIZ MODE: generate and rerun (persist)
    if st.session_state.mode == "Quiz":
        quiz_data = generate_quiz_json(prompt, context_text, st.session_state.user_level)
        st.session_state.quiz_data = quiz_data
        st.session_state.quiz_id = message_id
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.session_state.quiz_feedback = None
        st.session_state.quiz_feedback_for_id = None

        # also reset per-question selection states
        for i in range(10):
            key = f"selected_{message_id}_{i}"
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # EXPLAIN MODE
    explain_prompt = f"""
{level_instruction}
You MUST base your answer ONLY on the information in the context below,
which comes from the book. If the context is not enough to answer,
say that clearly and do not invent information.

When possible, mention which [Chunk # | page #] you used.

Information from the book (retrieved via {search_method}):

{context_text}

Question: {prompt}

Answer:
"""
    response = call_llm(explain_prompt)

    with st.chat_message("assistant"):
        st.write(response)

        with st.expander("Sources (from the book)"):
            for i, d in enumerate(context_docs[:3], start=1):
                page0 = d.metadata.get("page", None)
                page = (page0 + 1) if isinstance(page0, int) else "unknown"
                st.markdown(f"**Source {i} — page {page}**")
                st.write(d.page_content[:450] + ("..." if len(d.page_content) > 450 else ""))

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.stop()


# =================== QUIZ RENDER (PERSISTENT) ===================

if st.session_state.mode == "Quiz" and st.session_state.quiz_data:
    quiz_data = st.session_state.quiz_data
    quiz_id = st.session_state.quiz_id or "quiz"

    with st.chat_message("assistant"):
        st.subheader(quiz_data.get("title", "Quiz"))

        questions = quiz_data.get("questions", [])
        if not questions:
            st.write("I couldn't generate a quiz from the available context. Try a more specific question.")
        else:
            for qi, q in enumerate(questions):
                st.markdown(f"**Q{qi+1}. {q['question']}**")
                options = q["options"]

                # selection state for "box buttons"
                sel_key = f"selected_{quiz_id}_{qi}"
                if sel_key not in st.session_state:
                    st.session_state[sel_key] = None

                selected = st.session_state[sel_key]

                # show as 2-column "boxes"
                cols = st.columns(2)
                for oi, opt in enumerate(options):
                    col = cols[oi % 2]
                    with col:
                        label = opt
                        if selected == oi:
                            label = f"✅ {opt}"

                        if st.button(
                            label,
                            key=f"pick_{quiz_id}_{qi}_{oi}",
                            disabled=st.session_state.quiz_submitted,
                            use_container_width=True,
                        ):
                            st.session_state[sel_key] = oi
                            st.session_state.quiz_answers[qi] = oi
                            st.rerun()

                st.write("")  # spacing

            all_answered = all(qi in st.session_state.quiz_answers for qi in range(len(questions)))

            c1, c2 = st.columns([1, 1])
            with c1:
                if not st.session_state.quiz_submitted:
                    if st.button("Done ✅", key=f"quiz_{quiz_id}_done"):
                        st.session_state.quiz_submitted = True

                        # Generate feedback ONCE for this quiz_id
                        if st.session_state.quiz_feedback_for_id != quiz_id:
                            questions = st.session_state.quiz_data.get("questions", [])
                            answers = st.session_state.quiz_answers

                            # Build a compact summary for the model
                            qa_lines = []
                            for qi, q in enumerate(questions):
                                chosen = answers.get(qi, None)
                                correct = q["correct_index"]
                                chosen_txt = q["options"][chosen] if chosen is not None else "No answer"
                                correct_txt = q["options"][correct]
                                qa_lines.append(
                                    f"Q{qi+1}: {q['question']}\n"
                                    f"- User chose: {chosen_txt}\n"
                                    f"- Correct: {correct_txt}\n"
                                    f"- Explanation (book-based): {q['explanation']} {q.get('source','')}\n"
                                )

                            feedback_prompt = f"""
                    You are a helpful course-book tutor.
                    Use ONLY the explanations and sources below (from the book context). Do NOT add outside knowledge.

                    Write a short follow-up tutor message:
                    - 3–6 sentences
                    - Explain what the topic is in simple terms (based on the provided explanations)
                    - Tell the user what they missed in their wrong answers
                    - End with one suggested next question the user can ask

                    Quiz details:
                    {chr(10).join(qa_lines)}
                    """
                            st.session_state.quiz_feedback = call_llm(feedback_prompt)
                            st.session_state.quiz_feedback_for_id = quiz_id

                        st.rerun()

            with c2:
                if st.session_state.quiz_submitted:
                    if st.button("Try again 🔄", key=f"quiz_{quiz_id}_retry"):
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_answers = {}
                        # reset selection markers
                        for qi in range(len(questions)):
                            st.session_state[f"selected_{quiz_id}_{qi}"] = None
                        st.rerun()
                        
                    if st.session_state.quiz_feedback:
                        st.divider()
                        st.markdown("### Tutor feedback")
                        st.write(st.session_state.quiz_feedback)

            if st.session_state.quiz_submitted:
                st.divider()
                st.markdown("### Results")

                for qi, q in enumerate(questions):
                    correct = int(q["correct_index"])
                    chosen = st.session_state.quiz_answers.get(qi, None)

                    if chosen == correct:
                        st.success(f"Q{qi+1}: Correct ✅")
                    else:
                        st.error(f"Q{qi+1}: Wrong ❌")

                    st.write(f"**Correct answer:** {q['options'][correct]}")
                    st.write(f"**Why:** {q['explanation']}")
                    st.caption(f"Source: {q.get('source', '')}")


# =================== SIDEBAR ===================

with st.sidebar:
    st.header("About")
    st.write(
        """
**Fusion RAG** uses multiple retrieval methods:
- **Fusion**: Combines vector and keyword search
- **Vector**: Semantic search (conceptual)
- **BM25**: Keyword search (exact terms)

Answers are always based on the selected book.
"""
    )

    st.subheader("Search Parameters")
    st.slider("Number of contexts (k)", 1, 10, 5, key="k_value")

    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.session_state.quiz_data = None
        st.session_state.quiz_id = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.rerun()
