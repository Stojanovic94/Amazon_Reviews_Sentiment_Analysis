import pandas as pd
import kagglehub
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# Step 1: Load the dataset
path = kagglehub.dataset_download("dongrelaxman/amazon-reviews-dataset")
print("Dataset path:", path)

amazon_csv = os.path.join(path, "Amazon_Reviews.csv")
df = pd.read_csv(
    amazon_csv,
    encoding='utf-8',
    sep=',',
    quotechar='"',
    on_bad_lines='skip',
    engine='python'
)

# Clean and check
df = df[df['Review Text'].notna()].copy()
print(df[['Review Text', 'Rating']].head())

# Step 2: Sentiment Analysis with VADER
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
df.columns = df.columns.str.strip().str.lower()

df['sentiment_score'] = df['review text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

def classify_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_label'] = df['sentiment_score'].apply(classify_sentiment)

# Distribution
print("Review count by sentiment:")
print(df['sentiment_label'].value_counts())
print("\nSentiment percentages:")
print(df['sentiment_label'].value_counts(normalize=True) * 100)

# Step 3: Most positive/negative reviews
most_positive = df.loc[df['sentiment_score'].idxmax()]
most_negative = df.loc[df['sentiment_score'].idxmin()]

print("\nMost Positive Review:\n", most_positive['review text'])
print("\nMost Negative Review:\n", most_negative['review text'])

# Step 4: Sentiment vs. Rating
df['rating_clean'] = df['rating'].str.extract(r'(\d+)').astype(float)
df = df[df['rating_clean'].notna()]

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='rating_clean', y='sentiment_score')
plt.title("Sentiment Score vs. Star Rating")
plt.xlabel("Star Rating")
plt.ylabel("Compound Sentiment Score")
plt.show()

misaligned = df[((df['sentiment_label'] == 'Positive') & (df['rating_clean'] <= 2)) |
                ((df['sentiment_label'] == 'Negative') & (df['rating_clean'] >= 4))]
print(f"\nNumber of mismatched reviews (sentiment vs rating): {len(misaligned)}")

# Step 5: WordClouds
positive_text = " ".join(df[df['sentiment_label'] == 'Positive']['review text'].dropna())
negative_text = " ".join(df[df['sentiment_label'] == 'Negative']['review text'].dropna())

# Positive WordCloud
wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off')
plt.title("Frequent Words in Positive Reviews")
plt.show()

# Negative WordCloud
wordcloud_neg = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title("Frequent Words in Negative Reviews")
plt.show()