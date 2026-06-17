import streamlit as st
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="MedReport AI", page_icon="🩺", layout="wide")
st.title("🩺 Medical Report Intelligence Assistant")
st.divider()

# Path to the FAISS index folder (index.faiss + index.pkl) that was already
# built and saved separately. Point this at wherever your embeddings live:
#   - "medical_faiss_index"                                  -> same folder as this script
#   - "/content/drive/MyDrive/medical_faiss_index"            -> Google Drive (in Colab)
FAISS_INDEX_PATH = "medical_faiss_index"

# Try to get the key from Streamlit secrets first; only show the input box if it's missing
groq_key = st.secrets.get("GROQ_API_KEY", None)

with st.sidebar:
    st.header("⚙️ Setup")
    if not groq_key:
        groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.caption(f"Loading embeddings from: `{FAISS_INDEX_PATH}`")
    load_btn = st.button("⚡ Load Report & Start", use_container_width=True)
    st.warning("⚠️ For informational purposes only. Always consult a doctor.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if load_btn:
    if not groq_key:
        st.sidebar.error("Please provide your Groq API key.")
    elif not os.path.exists(FAISS_INDEX_PATH):
        st.sidebar.error(f"No embeddings found at '{FAISS_INDEX_PATH}'. Build them first.")
    else:
        with st.spinner("Loading embeddings..."):
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
            vectorstore = FAISS.load_local(
                MEDICAL_FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
            )
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
    st.info("👈 Enter your Groq API key and click 'Load Report & Start'.")
