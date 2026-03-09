from transformers import pipeline

model = pipeline("text-classification",model="yiyanghkust/finbert-tone")
text = "The stock market rally continued, suggesting strong long-term growth."
result = model(text)
print(result)
