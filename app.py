import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
# GUNA CARA INI UNTUK VERSI TERBARU:
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# 1. SETUP API KEYS (Ambil dari Secrets)
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(page_title="Chatbot RAG", layout="wide")
st.title("🤖 Chatbot RAG (DeepSeek + Excel/PDF)")

# ... (Bahagian upload & pemprosesan fail kekal sama sehingga bahagian "vectorstore") ...

if uploaded_file:
    # (Kod pemprosesan fail anda di sini...)
    # Katakan anda sudah ada 'vectorstore'
    
    # 4. CHAT INTERFACE (BAHAGIAN BARU UNTUK LANGCHAIN v1.x)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("Tanya sesuatu tentang fail anda..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="deepseek-r1-distill-llama-70b")
            
            # Bina Chain Baru (Cara LangChain v1.x)
            system_prompt = (
                "Anda adalah pembantu untuk tugas menjawab soalan. "
                "Gunakan cebisan konteks berikut untuk menjawab soalan. "
                "Jika anda tidak tahu jawapannya, katakan anda tidak tahu. "
                "\n\n"
                "{context}"
            )
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
            retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
            
            # Jalankan
            response = retrieval_chain.invoke({"input": prompt})
            answer = response["answer"]
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
