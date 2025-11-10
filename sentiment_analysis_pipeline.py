from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

sent_pipeline = pipeline('sentiment-analysis')

print(sent_pipeline('This cheese is so good'))