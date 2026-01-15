import streamlit as st
import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
# PERUBAHAN DI SINI:
from langchain.chains.retrieval_qa.base import RetrievalQA 
from langchain_core.documents import Document

# 1. SETUP API KEYS (Ambil dari Secrets)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("API Key tidak dijumpai di Streamlit Secrets!")
    st.stop()

st.set_page_config(page_title="Chatbot RAG", layout="wide")
st.title("🤖 Chatbot RAG (DeepSeek + Excel/PDF)")

# 2. SIDEBAR - UPLOAD FAIL
with st.sidebar:
    st.header("Dokumen")
    uploaded_file = st.file_uploader("Upload PDF atau Excel/CSV", type=["pdf", "xlsx", "csv"])

# 3. PROSES FAIL
if uploaded_file:
    docs = []
    with st.spinner("Sedang membaca fail..."):
        try:
            if uploaded_file.name.endswith(".pdf"):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader("temp.pdf")
                docs = loader.load()
            elif uploaded_file.name.endswith((".xlsx", ".csv")):
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                text_data = f"Data dari {uploaded_file.name}:\n\n" + df.to_string()
                docs = [Document(page_content=text_data)]

            # Bina Vector Store
            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.success("Fail berjaya diproses!")

                # 4. CHAT INTERFACE
                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])

                if prompt := st.chat_input("Tanya sesuatu tentang fail anda..."):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)

                    with st.chat_message("assistant"):
                        llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")
                        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
                        res = qa.invoke(prompt) # Guna invoke sebagai ganti run
                        st.markdown(res["result"])
                        st.session_state.messages.append({"role": "assistant", "content": res["result"]})
        except Exception as e:
            st.error(f"Ralat semasa memproses fail: {e}")
else:
    st.info("Sila upload fail untuk mula.")
