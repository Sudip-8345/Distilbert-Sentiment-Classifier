# Distilbert-Sentiment-Classifier

### Fine-tuned DistilBERT model for classifying sentiment of TripAdvisor hotel reviews as positive or negative experiences, achieving 93%+ accuracy.
***
## Why I Chose This Dataset

- I selected the TripAdvisor Hotel Reviews dataset because it contains 20,000+ real-world reviews with ratings, making it ideal for training and evaluating a sentiment analysis model. Reviews are diverse, covering both positive and negative customer experiences, which helps to build a balanced and practical classifier model.
***
## Preprocessing Steps Before Training

### Cleaning:

- Removed special characters, numbers, and HTML tags

- Converted all text to lowercase

- Removed stopwords (common words like “the”, “is”)

- Applied stemming to reduce words to root form (“running” → “run”)

### Labeling:

- Converted ratings > 3 → Positive

- Ratings ≤ 3 → Negative

### Tokenization:

- Used Hugging Face DistilBERT tokenizer

- Split text into subword tokens

- Applied padding & truncation to ensure uniform sequence length (max 512 tokens)

- These steps ensured that the text was clean, consistent, and ready for model training.
***
## Rationale & Output Differences

### Direct Instruction: 
- Simple and efficient
- outputs just “Positive” or “Negative.”

### Few-Shot Prompting: 
- Adds examples, improves consistency and accuracy on tricky reviews.

### Chain-of-Thought: 
- outputs are longer but more interpretable.
***
### Issue:

- When I directly evaluated raw reviews on DistilBERT without proper fine-tuning, the model performed poorly (low accuracy) due to zero-shot evaluation.

- I also attempted to train a much larger model (LLaMA) on only ~500 samples, but training was very slow and unstable (loss stuck around 4, poor predictions) due less gpu capabilibity.
### Proposed Solutions:

1. Use lightweight models (DistilBERT, BERT-mini) that train faster and perform well with fewer resources.
2. Increase training data (at least a few thousand samples) or apply data augmentation to improve generalization.
3. Optimize training – lower learning rate, fewer epochs, gradient accumulation for large models.
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
