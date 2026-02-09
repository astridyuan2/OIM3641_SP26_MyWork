from dotenv import load_dotenv
from google import genai
import os
import re

load_dotenv()
client = genai.Client()

def generate_prompt():
    return f"""write the python code to calculate
a loan payment with the following inputs: interest,
term, present value. return code only wrapped in a Markdown code block
(triple backticks). Do not add any extra text or explanation outside the
code block."""

response = client.models.generate_content(
    model = "gemini-2.5-flash",
    contents = generate_prompt()
)

match = re.search(r"```.*?([\w\W]*?)```", response.text, re.DOTALL)
code_content = match.group(1).strip()
print("---Extracted code ---")

print(code_content)
with open("loan_calc.py", "w")as f:
    f.write(code_content)
    f.close()