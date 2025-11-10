from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import pandas as pd


MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

reviews_df = pd.read_csv("archive//Reviews.csv").head(500)

def get_sentiment(text):
    encoded_text = tokenizer(text, truncation = True, return_tensors="pt", max_length=512)
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {'roberta_neg': scores[0],
                   'roberta_neu': scores[1],
                   'roberta_pos': scores[2]}
    return scores_dict

sentiment_result = []

for idx, text in zip(reviews_df['Id'], reviews_df['Text']):
    sentiment_result.append(get_sentiment(text))


print(sentiment_result)





