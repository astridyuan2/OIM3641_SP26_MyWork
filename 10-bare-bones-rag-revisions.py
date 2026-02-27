from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
import os
import streamlit as st

st.set_page_config(page_title="Babson Handbook RAG System", layout="centered")
load_dotenv()
DATA_DIR = 'data/handbook'

Settings.llm = GoogleGenAI(model= "gemini-2.5-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name= "BAAI/bge-small-en-v1.5")

# --- CORE LOGIC ---
@st.cache_resource
def get_query_engine():
    if not os.getenv("GEMINI_API_KEY"):
        st.error("❌ GEMINI_API_KEY not found")
        st.stop()
    st.info("🗂️ Loading documents and creating index")
    documents = SimpleDirectoryReader('data/handbook').load_data()
    index = VectorStoreIndex.from_documents(documents)
    st.success("😁 RAG Indexing complete!")
    return index.as_query_engine()



# --- STREAMLIT UI ---
st.title("Bare Bones RAG Chatbot")
query_engine = get_query_engine()
prompt = st.chat_input("Ask me  a question...")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    # chat_message "user"/"assistant" comes with default emojicon
    # this can be customized by providing an avatar= argument
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            response = query_engine.query(prompt)
            bot_response = response.response
        st.markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": "bot_response"})

