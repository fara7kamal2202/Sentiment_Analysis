import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
import seaborn as sns
import nltk

def plot_graph(df):
    plt.style.use('ggplot')
    df['Score'].value_counts().sort_index().plot(kind='bar', title = "Food Reviews by Stars")
    plt.xticks(rotation=0)
    plt.xlabel('Stars')
    plt.ylabel('Review Count')
    plt.show()


reviews_df = pd.read_csv("archive/Reviews.csv").head(500)
plot_graph(reviews_df)


example = reviews_df['Text'].iloc[50]
tokens = nltk.word_tokenize(example)
pos_tag = nltk.pos_tag(tokens)
chunks = nltk.ne_chunk(pos_tag)

sentiment_analyzer = SentimentIntensityAnalyzer()

result = {}
for idx, review in zip(reviews_df["Id"], reviews_df["Text"]):
    result.update({idx: sentiment_analyzer.polarity_scores(review)})

result_df = pd.DataFrame(result).T
result_df = result_df.reset_index().rename(columns={"index": "Id"})

result_df = result_df.merge(reviews_df, how='left')


ax = sns.barplot(data = result_df, x="Score", y="compound")
ax.set_title("Sentiment Analysis Results")
plt.show()

fix, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
sns.barplot(data = result_df, x="Score", y="neg", ax=axs[0])
sns.barplot(data = result_df, x="Score", y="neu", ax=axs[1])
sns.barplot(data = result_df, x="Score", y="pos", ax=axs[2])
axs[0].set_title("Negative")
axs[1].set_title("Neutral")
axs[2].set_title("Positive")
plt.show()