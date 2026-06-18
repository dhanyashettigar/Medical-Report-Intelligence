import streamlit as st
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="MedReport AI", page_icon="🩺", layout="wide")
st.title("🩺 Medical Report Intelligence Assistant")
st.divider()

FAISS_INDEX_PATH = "FAISS DB"

groq_key = st.secrets.get("GROQ_API_KEY", None)

with st.sidebar:
    st.header("⚙️ Setup")
    if not groq_key:
        groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")

    uploaded_file = st.file_uploader("📄 Upload your medical report (PDF)", type=["pdf"])
    st.caption(f"If no file is uploaded, loads saved embeddings from: `{FAISS_INDEX_PATH}`")

    load_btn = st.button("⚡ Load Report & Start", use_container_width=True)
    st.warning("⚠️ For informational purposes only. Always consult a doctor.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

def get_full_text_from_vectorstore(vs):
    try:
        return "\n\n".join(doc.page_content for doc in vs.docstore._dict.values())
    except Exception:
        return ""

if load_btn:
    if not groq_key:
        st.sidebar.error("Please provide your Groq API key.")
    elif not uploaded_file and not os.path.exists(FAISS_INDEX_PATH):
        st.sidebar.error("Please upload a PDF, or make sure saved embeddings exist.")
    else:
        with st.spinner("Reading report and building embeddings..."):
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

            if uploaded_file:
                reader = PdfReader(uploaded_file)
                raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)

                if not raw_text.strip():
                    st.sidebar.error("Couldn't extract any text from this PDF. Is it a scanned image?")
                    st.stop()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_text(raw_text)
                vectorstore = FAISS.from_texts(chunks, embeddings)
                full_text = raw_text
            else:
                vectorstore = FAISS.load_local(
                    FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
                )
                full_text = get_full_text_from_vectorstore(vectorstore)

            retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

            llm = ChatOpenAI(
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )

            # Chain used for follow-up Q&A
            qa_prompt = PromptTemplate(
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
                | qa_prompt
                | llm
                | StrOutputParser()
            )

            # Auto-generate a full explanation of the report before any question is asked
            summary_prompt = PromptTemplate(
                template="""You are a medical report assistant. Read the following medical report and explain it to the patient in simple, clear language. Cover:
- What the report is about
- Each test/value and whether it's normal or abnormal
- An overall health summary
- Practical lifestyle tips

Report:
{report}

Explanation:""",
                input_variables=["report"],
            )
            summary_chain = summary_prompt | llm | StrOutputParser()
            intro_text = summary_chain.invoke({"report": full_text[:15000]})

            st.session_state.messages = [{"role": "assistant", "content": intro_text}]
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
    st.info("👈 Enter your Groq API key, optionally upload a PDF, and click 'Load Report & Start'.")
