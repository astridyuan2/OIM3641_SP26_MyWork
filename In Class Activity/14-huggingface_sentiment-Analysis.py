from transformers import pipeline

model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The latest press release details the company's new policy on remote work, including guidelines for team communication and hardware allocation for employees worldwide."

labels = ["Employee Relations", "Financial News", "Product Announcement", "Technical Support", "Sales", "HR Policy", "Legal Compliance"]

result = model(text, labels)

print("Zero-Shot Classification Results")
print(f"Input Text: {result['sequence']}")
print("Classification Scores:")

for i, (label, score) in enumerate(zip(result['labels'], result['scores'])):
    print(f"  {i+1}. {label}: {round(score * 100, 2)}%")
