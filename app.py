import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import uuid

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    
# ── 🔑 API Keys (hardcoded – never shown to user) ─────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY   = os.getenv("HF_API_KEY")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY

# ✅ ADD HERE
@st.cache_resource
def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="TalkToPDF", page_icon="📄", layout="wide")
st.title("📄 TalkToPDF")
st.caption("Upload a PDF and chat with it using AI")

# ── Sidebar – only upload + clear (no API key inputs) ────────────────────────
with st.sidebar:
    st.header("📂 Upload")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages       = []
        st.session_state.qa_chain       = None
        st.session_state.topic          = None
        st.session_state.processed_file = None
        st.rerun()

# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("messages",       []),
    ("qa_chain",       None),
    ("topic",          None),
    ("processed_file", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── PDF processing with step-by-step progress ─────────────────────────────────
# def build_qa_chain(pdf_bytes, step_placeholder):
#     """Build the full RAG pipeline with visible step-by-step progress."""

#     def show(icon, message):
#         step_placeholder.markdown(f"{icon} {message}")

#     # Step 1 – Load PDF
#     show("📄", "**Step 1/5** — Loading PDF document…")
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(pdf_bytes)
#         tmp_path = tmp.name
#     loader = PyPDFLoader(tmp_path)
#     docs   = loader.load()
#     os.unlink(tmp_path)
#     show("✅", f"**Step 1/5 done** — {len(docs)} page(s) loaded.")

#     # Step 2 – Split text
#     show("✂️", "**Step 2/5** — Splitting text into chunks…")
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
#     chunks   = splitter.split_documents(docs)
#     show("✅", f"**Step 2/5 done** — {len(chunks)} chunks created.")

#     # Step 3 – Embeddings
#     show("🧠", "**Step 3/5** — Generating embeddings (may take a moment)…")
#     embedding = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
#     )
#     vectordb  = FAISS.from_documents(documents=chunks, embedding=embedding)
#     # store per user session
#     st.session_state["vectordb"] = vectordb
#     retriever = st.session_state["vectordb"].as_retriever(
#         search_type="mmr",
#         search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
#     )
#     show("✅", "**Step 3/5 done** — Embeddings generated, vector store ready.")

#     # Step 4 – Detect topic
#     show("🔍", "**Step 4/5** — Detecting PDF topic via LLM…")
#     llm         = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
#     sample_text = " ".join([doc.page_content for doc in chunks[:3]])
#     topic_prompt = (
#         "You are an expert topic classifier.\n"
#         "Identify the main topic of the given text in maximum 3 words only.\n"
#         "Return ONLY the topic — no explanation, no punctuation.\n\n"
#         f"Text:\n{sample_text}"
#     )
#     topic = llm.invoke(topic_prompt).content.strip()
#     show("✅", f"**Step 4/5 done** — Topic detected: **{topic}**")

#     # Step 5 – Build QA chain
#     show("⚙️", "**Step 5/5** — Building conversational QA chain…")
#     prompt_template = """
#     You are a helpful assistant.
    
#     Use ONLY the information provided in the context below to answer the question.
#     Do NOT use any external knowledge or assumptions.
    
#     If the answer is not clearly available in the context, respond with:
#     Strictly - "This information is not available in the provided PDF."
    
#     Context:
#     {context}
    
#     Question:
#     {question}
    
#     Instructions:
#     - Provide the answer in simple, clear language.
#     - Use bullet points only.
#     - Keep the answer short and concise.
#     - Do not repeat the question or add irrelevant information.
#     - If the question makes no sense, say "Not a valid question."
    
#     Exception:
#     - If the user explicitly asks for a detailed explanation, provide a more detailed answer (still in bullet points).
#     """
#     prompt = PromptTemplate(
#         template=prompt_template,
#         input_variables=["context", "question"],
#     )
#     memory = ConversationBufferMemory(
#         memory_key=f"chat_history_{st.session_state.session_id}",
#         return_messages=True
#     )
#     qa_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=retriever,
#         memory=memory,
#         combine_docs_chain_kwargs={"prompt": prompt},
#     )
#     show("🎉", "**All steps complete!** You can now chat with your PDF below.")
#     return qa_chain, topic


# ── Trigger processing when a new file is uploaded ────────────────────────────
if uploaded_file is not None:
    if uploaded_file.size > 2 * 1024 * 1024:  # 2MB
        st.error("❌ File size should be less than 2MB")
        st.stop
        
    file_id = uploaded_file.name + str(uploaded_file.size) + st.session_state.session_id

    if file_id != st.session_state.processed_file:
        st.session_state.messages = []  # clear chat for new PDF
         # ✅ ADD THESE LINES HERE
        st.session_state.qa_chain = None
        st.session_state["vectordb"] = None

        st.markdown("---")
        st.markdown("### ⏳ Processing your PDF…")
        progress_bar  = st.progress(0, text="Starting…")
        step_display  = st.empty()

        # Wrap build to also update progress bar
        class ProgressTracker:
            def __init__(self):
                self.step = 0
            def update(self, icon, message):
                self.step += 1
                pct = int((self.step / 5) * 100)
                progress_bar.progress(pct, text=f"Step {self.step}/5")
                step_display.markdown(f"{icon} {message}")

        tracker = ProgressTracker()

        try:
            # Monkey-patch a simpler callable into build_qa_chain
            def step_fn(icon, message):
                tracker.update(icon, message)

            # Run with step-level updates
            def build_with_progress(pdf_bytes):
                def show(icon, message):
                    
                    if "done" in message.lower() or "all done" in message.lower():
                        tracker.update(icon, message)
                    else:
                        step_display.markdown(f"{icon} {message}")

                # Step 1
                show("📄", "**Step 1/5** — Loading PDF document…")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name
                loader = PyPDFLoader(tmp_path)
                docs   = loader.load()
                os.unlink(tmp_path)
                show("✅", f"**Step 1/5 done** — {len(docs)} page(s) loaded.")

                # Step 2
                show("✂️", "**Step 2/5** — Splitting text into chunks…")
                splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
                chunks   = splitter.split_documents(docs)
                show("✅", f"**Step 2/5 done** — {len(chunks)} chunks created.")

                # Step 3
                show("🧠", "**Step 3/5** — Generating embeddings (may take a moment)…")
                embedding = get_embedding()
                vectordb  = FAISS.from_documents(documents=chunks, embedding=embedding)
                # store per user session
                if st.session_state.get("vectordb") is None:
                    vectordb = FAISS.from_documents(chunks, embedding)
                    st.session_state["vectordb"] = vectordb
                else:
                    vectordb = st.session_state["vectordb"]
                retriever = st.session_state["vectordb"].as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.5},
                )
                show("✅", "**Step 3/5 done** — Embeddings generated.")

                # Step 4
                show("🔍", "**Step 4/5** — Detecting PDF topic…")
                llm         = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
                sample_text = " ".join([doc.page_content for doc in chunks[:3]])
                topic_prompt = (
                    "You are an expert topic classifier.\n"
                    "Identify the main topic of the given text in maximum 3 words only.\n"
                    "Return ONLY the topic — no explanation, no punctuation.\n\n"
                    f"Text:\n{sample_text}"
                )
                topic = llm.invoke(topic_prompt).content.strip()
                show("✅", f"**Step 4/5 done** — Topic: **{topic}**")

                # Step 5
                show("⚙️", "**Step 5/5** — Building QA chain…")
                prompt_template = """
                You are a helpful assistant.
                
                Use ONLY the information provided in the context below to answer the question.
                Do NOT use any external knowledge or assumptions.
                
                If the answer is not clearly available in the context, respond with:
                Strictly - "This information is not available in the provided PDF."
                
                Context:
                {context}
                
                Question:
                {question}
                
                Instructions:
                - Provide the answer in simple, clear language.
                - Use bullet points only.
                - Keep the answer short and concise.
                - Do not repeat the question or add irrelevant information.
                - If the question makes no sense, say "Not a valid question."
                
                Exception:
                - If the user explicitly asks for a detailed explanation, provide a more detailed answer (still in bullet points).
                """
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"],
                )
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": prompt},
                )
                show("🎉", "**All done!** Chat with your PDF below 👇")
                return qa_chain, topic

            qa_chain, topic = build_with_progress(uploaded_file.read())
            progress_bar.progress(100, text="✅ Complete!")

            st.session_state.qa_chain       = qa_chain
            st.session_state.topic          = topic
            st.session_state.processed_file = file_id
            st.sidebar.success(f"✅ Ready! Topic: **{topic}**")

        except Exception as e:
            st.error(f"❌ Error processing PDF: {e}")

# ── Chat UI ───────────────────────────────────────────────────────────────────
if st.session_state.topic:
    st.markdown("---")
    st.info(f"📚 PDF topic: **{st.session_state.topic}**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt_input := st.chat_input("Ask something about your PDF…"):
    if st.session_state.qa_chain is None:
        st.warning("⚠️ Please upload a PDF first.")
    else:
        if prompt_input.lower().strip() in ["hi", "hello", "hey"]:
            answer = "How may I help you? 😊"
        else:
            enriched = f"[Topic: {st.session_state.topic}] {prompt_input}"
            with st.spinner("🤔 Thinking…"):
                result = st.session_state.qa_chain.invoke({"question": enriched})
            answer = result["answer"]

        st.session_state.messages.append({"role": "user",      "content": prompt_input})
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("user"):
            st.markdown(prompt_input)
        with st.chat_message("assistant"):
            st.markdown(answer)
