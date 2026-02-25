from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
import os
import streamlit as st

Settings.llm = GoogleGenAI(model= 'gemini-2.5-flash')
Settings.embed_model = GoogleGenAIEmbedding(model_name= 'models/gemini-embedding-001')

# --- SETUP ---
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("Missing GEMINI_API_KEY in your .env file")

# --- CORE LOGIC ---
def get_query_engine():
    doc = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine(similarity_top_k=5)


# --- STREAMLIT UI ---
st.title("Bare Bones Rag Chatbot")
prompt = streamlit.chat_input("Ask me a question about Babson student handbook...")
query_engine = get_query_engine()

if prompt:
    st.write(f"User: {prompt}")
    # TO DO get response from LLM
    st.write(f"Chatbot response: {response}")