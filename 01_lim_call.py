from dotenv import load_dotenv 
from google import genai
import os
import re


load_dotenv()

client = genai.Gemini()

