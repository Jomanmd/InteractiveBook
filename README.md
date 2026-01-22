# InteractiveBook – AI-Powered Book-to-Course Learning System

> 🧠 Course Project – Intelligent Interactive Systems  
> Milestone 4: Final Prototype + Evaluation

## 📘 Project Description
InteractiveBook is an **interactive Streamlit web application** built around a **fixed course textbook** (*Intelligent Interactive Systems*).  
Instead of acting as a general-purpose chatbot, the system transforms the textbook into a **structured learning experience** that guides users through a complete course.

Learning flow:  
**Curriculum → Lessons → Checkpoint Quizzes → Final Exam → Certificate**

All explanations, answers, and assessments are **grounded in retrieved passages from the textbook**, improving transparency and reducing hallucinations.

---

## ✨ Main Features
- **Interactive web interface** implemented with Streamlit  
- **Personalized onboarding** (role, learning goal, level)
- **Automatic curriculum generation** based on the book’s table of contents
- **Structured lessons** with summaries and examples
- **Lesson-specific Q&A** grounded in textbook content
- **Checkpoint quizzes** after each chapter (must pass to continue)
- **Final exam and certificate** upon course completion
- **User-controlled retrieval modes**:
  - Fusion (hybrid: Vector + BM25)
  - Vector (semantic search)
  - BM25 (keyword-based search)
- **Transparent evidence** shown in explanations and quiz feedback

---

## 🧠 System Overview (RAG Pipeline)
1. The textbook PDF is split into overlapping text chunks with metadata.
2. Retrieval indices are built using:
   - FAISS (vector embeddings)
   - BM25 (lexical keyword search)
   - Fusion mode (combined ranking)
3. Relevant chunks are retrieved according to the selected retrieval mode.
4. A Groq-hosted large language model generates lessons, answers, quizzes, and exams  
   **using only the retrieved context**.

---

## 🛠️ Technologies Used
- Python  
- Streamlit  
- LangChain  
- FAISS  
- BM25  
- HuggingFace sentence-transformers  
- Groq LLM API  

---

## 📂 Repository Structure
Main files (branch `clean2`):
- `InteractiveBook2.py` – main Streamlit application  
- `requirements.txt` – Python dependencies  
- `runtime.txt` – runtime configuration  
- `intelligent_interactive_systems.pdf` – source textbook (required locally)  
- `assets/` – UI assets  

---

## ✅ Requirements
- Python **3.9+**
- A **Groq API key** (environment variable)

---

## 🚀 Running the Project Locally
The project is designed to run **locally** as an interactive website.

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Jomanmd/InteractiveBook.git
cd InteractiveBook
git checkout clean2
