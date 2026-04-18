# 📄 Chat2PDF – AI Chatbot (RAG)

## 🚀 Overview
Chat2PDF is a Streamlit-based web app that allows users to upload a PDF and interact with it using AI.  
It uses Retrieval-Augmented Generation (RAG) to provide accurate answers strictly from the document.

---
## ✨ Features

- 📄 Upload and chat with any PDF
- 🧠 AI-powered answers using RAG (Retrieval-Augmented Generation)
- 🔍 Smart document retrieval using FAISS
- 💬 Conversational memory for follow-up questions
- 🏷️ Automatic topic detection
- 👥 Multi-user support (session-based isolation)
- ⚡ Fast responses using Groq LLM

---

## ▶️ Usage

1. Upload a PDF file (max 2MB)
2. Wait for processing to complete
3. Ask questions related to the document
4. Get accurate answers based only on the PDF

---

## ⚠️ Limitations

- Maximum file size: 2MB
- Works best with text-based PDFs (no OCR support)
- Answers depend strictly on PDF content
- Large PDFs may take longer to process

---
## 👨‍💻 Author

**Shubham Dhole**

- GitHub: https://github.com/Shubham976300
---

## 🧠 Architecture

### 1. Document Loading
- **Library:** LangChain
- **Component:** PyPDFLoader
- Purpose: Load PDF and extract text

---

### 2. Text Splitting
- **Component:** RecursiveCharacterTextSplitter
- Chunk Size: 1500
- Purpose: Break document into smaller chunks

---

### 3. Embeddings
- **Model:** sentence-transformers/paraphrase-MiniLM-L3-v2
- **Library:** HuggingFaceEmbeddings
- Purpose: Convert text into vectors

---

### 4. Vector Database
- **Tool:** FAISS
- Purpose: Store and retrieve embeddings efficiently

---

### 5. Retriever
- Type: MMR (Max Marginal Relevance)
- Parameters:
  - k: 3
  - fetch_k: 10
  - lambda_mult: 0.5

---

### 6. LLM (Language Model)
- **Provider:** Groq
- **Model:** llama-3.1-8b-instant
- Purpose: Generate answers from context

---

### 7. Prompt Engineering
- Custom Prompt Template:
  - Uses only PDF context
  - Returns bullet points
  - Handles missing information

---

### 8. Memory
- **Component:** ConversationBufferMemory
- Purpose: Maintain chat history

---

### 9. Chain
- **Type:** ConversationalRetrievalChain
- Purpose: Combine retriever + LLM + memory

---

## ⚙️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace
- Groq API

---

## 🔄 Workflow

1. Upload PDF
2. Extract text
3. Split into chunks
4. Generate embeddings
5. Store in FAISS
6. Retrieve relevant chunks
7. Send to LLM
8. Return answer

---

## 🔐 Environment Variables

GROQ_API_KEY=your_key
HF_API_KEY=your_key
