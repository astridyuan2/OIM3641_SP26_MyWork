import streamlit as st
import requests
from bs4 import BeautifulSoup

st.title("URL Content Extractor")

url = st.text_input("Paste a URL to extract content")

if st.button("Extract Content"):

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text()

        words = text.split()
        word_count = len(words)

        preview = text[:200]

        st.subheader("Content Preview")
        st.write(preview)

        st.subheader("Word Count")
        st.write(word_count)

    except Exception as e:
        st.error("Failed to fetch the URL. Please check the link and try again.")