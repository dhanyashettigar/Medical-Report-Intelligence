import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="MedReport AI", page_icon="🩺", layout="wide")
st.title("🩺 Medical Report Intelligence Assistant")
st.divider()

# Try to get the key from Streamlit secrets first; only show the input box if it's missing
groq_key = st.secrets.get("GROQ_API_KEY", None)

with st.sidebar:
    st.header("⚙️ Setup")

    if not groq_key:
        groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    uploaded_file = st.file_uploader("Upload your medical report (PDF)", type=["pdf"])

    load_btn = st.button("⚡ Load Report & Start", use_container_width=True)

    st.warning("⚠️ For informational purposes only. Always consult a doctor.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if load_btn:
    if not groq_key:
        st.sidebar.error("Please provide your Groq API key.")
    elif uploaded_file is None:
        st.sidebar.error("Please upload a PDF report first.")
    else:
        with st.spinner("Reading and indexing your report..."):
            # Save the uploaded file temporarily so PyPDFLoader can read it
            temp_path = "temp_report.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Extract text from the PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            # Split into chunks for embedding
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)

            # Build embeddings + FAISS index on the fly from the uploaded report
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

            llm = ChatOpenAI(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )

            prompt = PromptTemplate(
                template="""You are a medical report assistant. Use the context below to explain the medical report in simple language, highlight abnormal values, give a health summary, and suggest lifestyle tips.

Context: {context}

Question: {question}

Answer in simple, clear language a patient can understand:""",
                input_variables=["context", "question"],
            )

            def format_docs(docs):
                return "\n\n".join(d.page_content for d in docs)

            st.session_state.qa_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            st.session_state.messages = []

            # Clean up the temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        st.sidebar.success("✅ Ready! Ask your questions below.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.qa_chain:
    user_input = st.chat_input("Ask about your report...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.qa_chain.invoke(user_input)
                st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Upload your report, enter your Groq API key, and click 'Load Report & Start'.")
