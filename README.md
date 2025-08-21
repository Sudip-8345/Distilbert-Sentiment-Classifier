# Distilbert-Sentiment-Classifier

### Fine-tuned DistilBERT model for classifying sentiment of TripAdvisor hotel reviews as positive or negative experiences, achieving 93%+ accuracy.
***
## 📌 Project Overview

This project fine-tunes DistilBERT
 on the TripAdvisor Hotel Reviews Dataset (20K+ reviews) to perform sentiment classification:

- Positive: Ratings > 3

- Negative: Ratings ≤ 3
***
## ⚙️ Tech Stack

- Python

- Transformers (Hugging Face)

- PyTorch

- Scikit-learn

- Pandas / NumPy
***
## 🚀 Setup & Usage
### 1️⃣ Install dependencies
- pip install -r requirements.txt

### 2️⃣ Run prediction script
- python app.py

### 3️⃣ Example output
- Review: The hotel was amazing and staff were super friendly!
- Predicted: POSITIVE (Confidence: 0.97)

- Review: Terrible stay, the room was dirty and noisy.
- Predicted: NEGATIVE (Confidence: 0.95)
