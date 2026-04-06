"""
In-class Activity 11 - Production Ready Code
Updates added for robustness:
1) Configuration validation (API key + data directory checks)
2) Directory/file presence checks (fast fail with helpful messages)
3) Cached document loading to reduce latency (Streamlit cache)
4) Try/except around RAG initialization and querying (fast fail + graceful errors)
5) Fixed chat history bug (was saving literal "bot_response")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI


# -----------------------------
# UI / APP SETUP
# -----------------------------
st.set_page_config(page_title="Babson Handbook RAG System", layout="centered")
st.title("Production-Ready RAG Chatbot")

load_dotenv()

# Centralized configuration (easy to refactor later)
DATA_DIR = Path("data/handbook")
API_KEY_ENV = "GEMINI_API_KEY"

# Model config (keep in one place)
LLM_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "models/gemini-embedding-001"


# -----------------------------
# CONFIG VALIDATION (fast fail)
# -----------------------------
@dataclass(frozen=True)
class AppConfig:
    api_key_env: str = API_KEY_ENV
    data_dir: Path = DATA_DIR


def validate_config(cfg: AppConfig) -> None:
    """Prevent common failures BEFORE any expensive work starts."""
    # 1) API key presence
    api_key = os.getenv(cfg.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key. Set environment variable {cfg.api_key_env} "
            f"(e.g., in a .env file) and restart the app."
        )

    # 2) Data directory existence
    if not cfg.data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {cfg.data_dir.as_posix()}. "
            "Create the directory and add your handbook files."
        )
    if not cfg.data_dir.is_dir():
        raise NotADirectoryError(
            f"Expected a directory but found a file: {cfg.data_dir.as_posix()}."
        )

    # 3) Files in directory (at least one)
    #    (You can tune this to require specific extensions if desired.)
    files = [p for p in cfg.data_dir.rglob("*") if p.is_file()]
    if not files:
        raise FileNotFoundError(
            f"No files found in {cfg.data_dir.as_posix()}. "
            "Add at least one document (e.g., .pdf/.txt/.md) before running."
        )


# -----------------------------
# CACHING (reduce latency)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_documents(data_dir_str: str):
    """Cache document load results to avoid re-reading on every rerun."""
    reader = SimpleDirectoryReader(data_dir_str)
    return reader.load_data()


@st.cache_resource(show_spinner=False)
def build_query_engine(data_dir_str: str):
    """Create index + query engine once per session (fast, stable UX)."""
    docs = load_documents(data_dir_str)
    index = VectorStoreIndex.from_documents(docs)
    return index.as_query_engine()


def init_rag_engine(cfg: AppConfig):
    """
    Initialize Settings + query engine.
    Wrapped in try/except to fast fail and avoid cascading errors.
    """
    try:
        validate_config(cfg)

        # Configure LlamaIndex (only after config validates)
        Settings.llm = GoogleGenAI(model=LLM_MODEL)
        Settings.embed_model = GoogleGenAIEmbedding(model_name=EMBED_MODEL)

        st.info("🗂️ Loading documents and creating index (cached after first run)...")
        engine = build_query_engine(cfg.data_dir.as_posix())
        st.success("✅ RAG engine ready!")
        return engine

    except Exception as e:
        st.error(f"❌ App failed to initialize: {e}")
        st.stop()  # Fast fail: don't run the main loop if core components are broken


# -----------------------------
# MAIN APP
# -----------------------------
config = AppConfig()
query_engine = init_rag_engine(config)

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
prompt = st.chat_input("Ask me a question...")

if prompt:
    # Store + render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query with runtime protection
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            try:
                response = query_engine.query(prompt)
                bot_response = getattr(response, "response", str(response))
            except Exception as e:
                bot_response = (
                    "Sorry — I hit an error while querying the RAG engine.\n\n"
                    f"**Details:** {e}\n\n"
                    "Try again, or re-check your API key / data files."
                )
                st.error("⚠️ Query failed. See details in the response.")

        st.markdown(bot_response)

    # IMPORTANT: store the actual response (not the literal string)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
