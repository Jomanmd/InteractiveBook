# InteractiveBook – AI-Powered Learning Assistant

## 📘 Project Description
InteractiveBook is an AI-powered interactive learning system that allows users to ask questions about a specific textbook and receive accurate, context-aware answers.  
Unlike general-purpose chatbots, this system retrieves information **only from the provided book**, ensuring relevance, transparency, and reduced hallucinations.

The system is designed as an educational assistant that supports different user levels and provides explanations grounded in the source material.

---

## 🧠 Key Features
- Question answering based on a single textbook (PDF)
- Retrieval-Augmented Generation (RAG)
- Combination of semantic search (FAISS) and lexical search (BM25)
- Transparent answers grounded in retrieved book sections
- Interactive web interface built with Streamlit

---

## 🛠️ Technologies Used
- Python
- Streamlit
- LangChain
- FAISS (vector similarity search)
- BM25 (lexical retrieval)
- Groq LLM API
- python-dotenv

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Jomanmd/InteractiveBook.git
cd InteractiveBook
