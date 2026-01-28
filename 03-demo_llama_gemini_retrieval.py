"""
pip install llama-index-llms-google-genai
pip install llama-index-embeddings-google-genai

"""

import dotenv

# 1. Load environment variables
load_dotenv()
llama_cloud_key = os.getenv("LLAMA_CLOUD_API_KEY")
org_id = os.getenv("ORGANIZATION_ID")
google_api_key = os.getenv("GEMINI_API_KEY") # Ensure this is in your .env

if not llama_cloud_key or not org_id or not google_api_key:
    raise ValueError("Missing required environment variables.")

# 2. Configure LlamaIndex to use Google GenAI
# Using Gemini 2.5 Flash for fast, low-cost reasoning
Settings.llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
    api_key=google_api_key
)

# Using Google's modern text-embedding model
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/text-embedding-004",
    api_key=google_api_key
)

# 3. Initialize the retriever to fetch nodes from your Llama Cloud index
retriever = LlamaCloudRetriever(
    name="AI_Proposal",
    project_name="Default",
    api_key=llama_cloud_key,
    organization_id=org_id
)

# 4. Create a query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
)

# 5. Run the query
query = "What is this document about?"
response = query_engine.query(query)

# 6. Print the response
print("Query response:")
print(response)
