import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "./models"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}

def predict_sentiment(texts):
    enc = tokenizer(texts, truncation=True, max_length=512,
                    padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    results = [id2label[p] for p in probs.argmax(axis=1)]
    return list(zip(texts, results, probs.max(axis=1)))

if __name__ == "__main__":
    reviews = [
        "The hotel was amazing and staff were super friendly!",
        "Terrible stay, the room was dirty and noisy."
    ]
    outputs = predict_sentiment(reviews)
    for text, label, conf in outputs:
        print(f"Review: {text}\nPredicted: {label} (Confidence: {conf:.2f})\n")
