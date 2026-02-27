from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
import streamlit as st

# --- SETUP ---
load_dotenv()
DATA_DIR = 'In Class Activity/data/handbook'
# rate limits for Gemini changed in 2025, which is causing the wall with
# gemini-embedding-001
Settings.llm = GoogleGenAI(model= 'gemini-2.5-flash', api_key=api_key)
Settings.embed_model = GoogleGenAIEmbedding(model_name= 'models/gemini-embedding-001', api_key= api_key)

# --- CORE LOGIC ---
def get_query_engine():
    docs = SimpleDirectoryReader('In Class Activity/data/handbook').load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine()


# --- STREAMLIT UI ---
st.title("Bare Bones RAG Chatbot")
query_engine = get_query_engine()
prompt = st.chat_input("Ask me a question...")

if prompt:
    st.write(f"User: {prompt}")
    # TO DO get response from LLM
    response = query_engine.query(prompt)
    bot_response = response.response
    st.write(f"Chatbot response: {response}")