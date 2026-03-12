# URL Content Extractor

This project is a minimal Streamlit application that extracts text content from a user-provided URL.

## Features

- Input a URL
- Fetch webpage content
- Display the first 200 characters of the extracted text
- Display the total word count
- Error handling for invalid or unreachable URLs

## Technologies Used

- Python
- Streamlit
- Requests
- BeautifulSoup

## How to Run the App

1. Clone the repository
2. Install required dependencies

pip install streamlit requests beautifulsoup4

3. Run the Streamlit app

streamlit run app.py

4. Open the local URL provided in your browser.

## Example Use Case

Paste a URL into the input field and click **Extract Content** to view the preview and word count of the webpage.