from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
import os
import streamlit as st

# --- SETUP ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("Missing GEMINI_API_KEY in your .env file")

Settings.llm = GoogleGenAI(model= 'gemini-2.5-flash', api_key=api_key)
Settings.embed_model = GoogleGenAIEmbedding(model_name= 'models/gemini-embedding-001', api_key= api_key)

# --- CORE LOGIC ---
def get_query_engine():
    docs = SimpleDirectoryReader("../data").load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine(similarity_top_k=5)


# --- STREAMLIT UI ---
st.title("Bare Bones Rag Chatbot")
prompt = st.chat_input("Ask me a question about Babson student handbook...")
query_engine = get_query_engine()

if prompt:
    st.write(f"User: {prompt}")
    # TO DO get response from LLM
    response = query_engine.query(prompt)
    st.write(f"Chatbot response: {response}")