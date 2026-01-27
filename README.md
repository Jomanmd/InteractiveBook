# 📘 InteractiveBook: AI-Powered Intelligent Book Tutor

## Description
**InteractiveBook** is an AI-powered learning system that transforms a static academic textbook into a **fully interactive, personalized course**.  
Instead of passively reading a PDF, students receive a **chapter-based curriculum** with lessons, quizzes, adaptive explanations, and a final certification — all generated directly from the book itself.

The system supports **different learner profiles**, dynamically adjusting explanations, examples, and language complexity based on the user’s background and learning goals.  
To ensure factual accuracy and prevent hallucinations, all content is generated using **Retrieval-Augmented Generation (RAG)** and is **strictly grounded in the original textbook**.

---

## 🚀 Live Demo
👉 **InteractiveBook on Streamlit**  
https://interactivebook2py-7a2b9mtpdfcctl6xivujiq.streamlit.app/

> ⚠️ **Note:**  
> Due to Streamlit hosting policies, the app may enter **Sleep Mode** if inactive.  
> Click **“Yes, get this app back up”** to reactivate it.

---

## 💡 Key Features

- **Chapter-Based Curriculum Generation**  
  Automatically converts a textbook into a structured learning path:  
  *Curriculum → Lessons → Quizzes → Final Exam → Certificate*

- **Hybrid Retrieval (Fusion / Vector / BM25)**  
  - **Fusion Search** (semantic + keyword, recommended)
  - **Vector Search** (conceptual understanding)
  - **BM25** (exact keyword matching)

- **Personalized Learning Personas**  
  Content adapts to:
  - User role (e.g., student, engineer, researcher)
  - Learning level (*Beginner* / *Advanced*)
  - Personal learning goal

- **Explainable AI (XAI)**  
  Every lesson, quiz answer, and chat response is backed by:
  - Exact book chunks
  - Page numbers
  - Short textual evidence

- **Chapter-Scoped AI Chat**  
  Users can ask questions during a lesson and receive answers **only from the relevant chapter**, ensuring accuracy and focus.

- **Active Learning & Locking Mechanism**
  - Each chapter includes a quiz
  - Next chapters unlock only after passing
  - Encourages mastery-based learning

- **Final Exam & Certification**
  - Comprehensive final exam
  - Automatic certificate generation with performance feedback

---

## 🧠 How It Works

1. **User Onboarding**  
   The user defines their name, role, learning goal, level, and retrieval strategy.

2. **Curriculum Generation**  
   The system extracts chapter titles from the book’s Table of Contents and builds a full course structure.

3. **Context-Injected Learning (RAG)**  
   Lessons and quizzes are generated using only retrieved book chunks.

4. **Interactive Learning Loop**
   - Read lesson
   - Ask questions
   - Take quiz
   - Unlock next chapter

5. **Final Assessment**
   Users complete a final exam and receive a personalized certificate.

---

## 🏗️ System Architecture

InteractiveBook follows a **Hybrid Intelligence Architecture**:

- **Knowledge Source**  
  The uploaded academic textbook (PDF) is the single source of truth.

- **Retrieval Layer**
  - FAISS (semantic vector search)
  - BM25 (keyword-based retrieval)
  - Reciprocal Rank Fusion (RRF)

- **Reasoning Layer**
  - LLM (LLaMA 3.1 via Groq API)
  - Strict prompting to prevent hallucinations

- **Deterministic Logic**
  - Quiz grading
  - Progress tracking
  - Chapter locking
  - Certification thresholds

- **Human-in-the-Loop**
  The learner remains in control, supported by transparent sources.

---

## 🛠️ Technology Stack

- **Frontend & Deployment:** Streamlit  
- **LLM Engine:** LLaMA 3.1 (via Groq API)  
- **Retrieval:** FAISS, BM25, Hybrid RRF  
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)  
- **PDF Processing:** PyMuPDF, PyPDF  
- **Language:** Python  
- **Architecture:** Retrieval-Augmented Generation (RAG)

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `InteractiveBook2.py` | Main Streamlit application |
| `intelligent_interactive_systems.pdf` | Source textbook |
| `vector_stores/` | FAISS vector index |
| `.streamlit/secrets.toml` | API key configuration (local only) |
| `assets/` | Background images and UI assets |
| `requirements.txt` | Python dependencies |

---

## 💻 Local Installation

### Prerequisites
- Python **3.8+**
- Groq API Key
- Internet connection

### 1. Clone the Repository
```bash
git clone https://github.com/Jomanmd/InteractiveBook
cd InteractiveBook

the code in branch clean2

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

> [!WARNING]
>
> **3. Configure API Key: Since our API keys are kept private for security, you need to create a local secrets file.**
>
> add in .streamlit folder a file with a name secrets.toml and paste this `GROQ_API_KEY="your_key" ` with the full line provided in the `API_Key.txt` file included in our **Moodle submission**.
> This line already contains the **actual API key** required for **InteractiveBook** to function, so you can simply copy and paste it directly into your terminal without any manual changes.

```bash
mkdir .streamlit
```

```bash
echo GROQ_API_KEY="your_key" > .streamlit/secrets.toml
```

**4. Run the app:**
```bash
python -m streamlit run InteractiveBook2.py
```
