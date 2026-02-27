from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path
import os
import streamlit as st
import time

# --- SETUP ---
st.set_page_config(page_title="Babson Handbook RAG System", layout="centered")
load_dotenv()
DATA_DIR = 'data/handbook'
API_KEY_ENV = "GEMINI_API_KEY"

Settings.llm = GoogleGenAI(model="gemini-2.5-flash")

embed = GoogleGenAIEmbedding(model_name="models/gemini-embedding-001")
if hasattr(embed, "embed_batch_size"):
    embed.embed_batch_size = 1
Settings.embed_model = embed
Settings.node_parser = SentenceSplitter(chunk_size=4096, chunk_overlap=200)

def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e)
    return ("429" in msg) or ("Too Many Requests" in msg) or ("RESOURCE_EXHAUSTED" in msg)


def is_quota_exhausted(e: Exception) -> bool:
    msg = str(e)
    return ("RESOURCE_EXHAUSTED" in msg) or ("exceeded your current quota" in msg.lower())

# --- CORE LOGIC ---
def validate_config():
    # 1) API key check (required by your assignment)
    if not os.getenv(API_KEY_ENV):
        st.error(f"❌ Missing API key: {API_KEY_ENV}. Add it to your .env file and restart Streamlit.")
        st.stop()

    # 2) Data dir checks
    data_dir = Path(DATA_DIR)

    if not data_dir.exists():
        st.error(f"❌ Data directory not found: {data_dir}")
        st.stop()

    if not data_dir.is_dir():
        st.error(f"❌ Expected a directory but found a file: {data_dir}")
        st.stop()

    # 3) Ensure directory has at least one file
    files = [p for p in data_dir.rglob("*") if p.is_file()]
    if not files:
        st.error(f"❌ No files found in {data_dir}. Add at least one document.")
        st.stop()

@st.cache_data(show_spinner=False)
def load_docs(data_dir_str: str):
    """Cache document loading to reduce latency on reruns."""
    return SimpleDirectoryReader(data_dir_str).load_data()

@st.cache_resource(show_spinner=False)
def get_query_engine():
    validate_config()

    st.info("🗂️ Loading documents and creating index...")
    docs = load_docs(str(Path(DATA_DIR)))

    # Retry index build on 429 with exponential backoff
    max_attempts = 5
    wait = 2  # seconds

    for attempt in range(1, max_attempts + 1):
        try:
            index = VectorStoreIndex.from_documents(docs)
            st.success("✅ RAG Indexing complete!")
            return index.as_query_engine()

        except Exception as e:
            # If Gemini embedding quota is exhausted, fall back to local embeddings
            if is_quota_exhausted(e):
                st.warning("⚠️ Gemini embedding quota exhausted. Falling back to local embeddings (HuggingFace).")

                try:
                    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
                    index = VectorStoreIndex.from_documents(docs)
                    st.success("✅ RAG Indexing complete (local embeddings)!")
                    return index.as_query_engine()

                except Exception as e2:
                    st.error(
                        "❌ Quota exhausted AND local embedding fallback failed.\n\n"
                        f"Details: {e2}\n\n"
                        "Fix: install local embedding deps:\n"
                        "pip install llama-index-embeddings-huggingface sentence-transformers"
                    )
                    st.stop()

            # Normal (non-quota) failure
            st.error(f"❌ Failed to initialize the RAG engine: {e}")
            st.stop()


# --- STREAMLIT UI ---
st.title("Bare Bones RAG Chatbot")
query_engine = get_query_engine()
prompt = st.chat_input("Ask me a question...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                response = query_engine.query(prompt)
                bot_response = getattr(response, "response", str(response))
                st.markdown(bot_response)

            except Exception as e:
                if is_rate_limit_error(e):
                    bot_response = (
                        "⚠️ I hit a Gemini rate limit (429) while answering.\n\n"
                        "Try:\n"
                        "- Wait a bit and ask again\n"
                        "- Ask a shorter question\n"
                        "- Clear cache and rerun"
                    )
                    st.error("⚠️ Rate limit hit.")
                    st.markdown(bot_response)
                else:
                    bot_response = (
                        "Sorry — I hit an error while querying the RAG engine.\n\n"
                        f"**Details:** {e}\n\n"
                        "Check your API key and that your data files are readable, then try again."
                    )
                    st.error("⚠️ Query failed.")
                    st.markdown(bot_response)

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
