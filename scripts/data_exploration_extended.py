import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob

# Load the dataset
file_path = "data/42hacks/train.csv"
data = pd.read_csv(file_path)

# Fill NaN values in 'text' with empty strings
data['text'] = data['text'].fillna('')

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#', '', text)        # Remove hashtag symbol
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    return text

data['clean_text'] = data['text'].apply(clean_text)

### 1. Number of Unique Words

# Combine all clean tweets into one string
all_words = ' '.join(data['clean_text'].values)
unique_words = set(all_words.split())
print(f"Total unique words: {len(unique_words)}")

### 2. Hashtag Analysis

# Function to extract hashtags
def extract_hashtags(text):
    return re.findall(r"#(\w+)", text)

data['hashtags'] = data['text'].apply(extract_hashtags)
all_hashtags = sum(data['hashtags'], [])
print(f"Total hashtags: {len(all_hashtags)}")
print(f"Unique hashtags: {len(set(all_hashtags))}")

# Top 10 hashtags
hashtag_counts = Counter(all_hashtags)
top_hashtags = hashtag_counts.most_common(10)
print("\nTop 10 Hashtags:")
for tag, count in top_hashtags:
    print(f"#{tag}: {count}")

# Visualize top hashtags
tags, counts = zip(*top_hashtags)
plt.figure(figsize=(12,6))
sns.barplot(x=list(tags), y=list(counts), palette="magma")
plt.title("Top 10 Hashtags", fontsize=14)
plt.xlabel("Hashtags", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45)
plt.show()

### 3. Mention (@username) Analysis

# Function to extract mentions
def extract_mentions(text):
    return re.findall(r"@(\w+)", text)

data['mentions'] = data['text'].apply(extract_mentions)
all_mentions = sum(data['mentions'], [])
print(f"Total mentions: {len(all_mentions)}")
print(f"Unique mentions: {len(set(all_mentions))}")

# Top 10 mentions
mention_counts = Counter(all_mentions)
top_mentions = mention_counts.most_common(10)
print("\nTop 10 Mentions:")
for mention, count in top_mentions:
    print(f"@{mention}: {count}")

# Visualize top mentions
mentions, counts = zip(*top_mentions)
plt.figure(figsize=(12,6))
sns.barplot(x=list(mentions), y=list(counts), palette="cividis")
plt.title("Top 10 Mentions", fontsize=14)
plt.xlabel("Mentions", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45)
plt.show()

### 4. URL Analysis

# Function to check for URLs
def has_url(text):
    return 1 if re.search(r'http\S+', text) else 0

data['has_url'] = data['text'].apply(has_url)
url_count = data['has_url'].sum()
print(f"Total tweets with URLs: {url_count}")

# Visualize URL presence by class
url_class_counts = data.groupby('target')['has_url'].sum().reset_index()
plt.figure(figsize=(8,5))
sns.barplot(x=['Non-Disaster', 'Disaster'], y=url_class_counts['has_url'], palette="pastel")
plt.title("Number of Tweets with URLs by Class", fontsize=14)
plt.xlabel("Class", fontsize=12)
plt.ylabel("Number of Tweets with URLs", fontsize=12)
plt.show()

### 5. Average Number of Hashtags/Mentions/URLs per Tweet

avg_hashtags = data['hashtags'].apply(len).mean()
avg_mentions = data['mentions'].apply(len).mean()
avg_urls = data['has_url'].mean()

print(f"\nAverage number of hashtags per tweet: {avg_hashtags:.2f}")
print(f"Average number of mentions per tweet: {avg_mentions:.2f}")
print(f"Average number of URLs per tweet: {avg_urls:.2f}")

### 6. Sentiment Analysis

# Function to compute sentiment polarity
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

data['sentiment'] = data['clean_text'].apply(get_polarity)

# Average sentiment per class
sentiment_class = data.groupby('target')['sentiment'].mean().reset_index()
print("\nAverage Sentiment Polarity by Class:")
print(sentiment_class)

# Visualize sentiment distribution
plt.figure(figsize=(10,6))
sns.histplot(data, x='sentiment', hue='target', bins=30, kde=True, palette=['blue','red'], alpha=0.6)
plt.title("Sentiment Polarity Distribution by Class", fontsize=14)
plt.xlabel("Sentiment Polarity", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(labels=['Non-Disaster', 'Disaster'])
plt.show()

### 7. Word Cloud Visualization

# Word cloud for all tweets
wordcloud_all = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(all_words)

plt.figure(figsize=(15,7.5))
plt.imshow(wordcloud_all, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for All Tweets", fontsize=20)
plt.show()

# Word cloud for disaster tweets
disaster_words = ' '.join(data[data['target']==1]['clean_text'].values)
wordcloud_disaster = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(disaster_words)

plt.figure(figsize=(15,7.5))
plt.imshow(wordcloud_disaster, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Disaster Tweets", fontsize=20)
plt.show()

# Word cloud for non-disaster tweets
non_disaster_words = ' '.join(data[data['target']==0]['clean_text'].values)
wordcloud_non_disaster = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(non_disaster_words)

plt.figure(figsize=(15,7.5))
plt.imshow(wordcloud_non_disaster, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for Non-Disaster Tweets", fontsize=20)
plt.show()

### 8. N-gram Analysis (Bigrams)

from sklearn.feature_extraction.text import CountVectorizer

# Function to plot top n-grams
def plot_top_ngrams(corpus, ngram_range=(2,2), n=10, title='Top N-grams'):
    vec = CountVectorizer(stop_words='english', ngram_range=ngram_range).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    top_n_grams = words_freq[:n]
    words, counts = zip(*top_n_grams)
    plt.figure(figsize=(12,6))
    sns.barplot(x=list(words), y=list(counts), palette="viridis")
    plt.title(title, fontsize=14)
    plt.xlabel("N-grams", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45)
    plt.show()

# Plot top bigrams for disaster tweets
disaster_corpus = data[data['target']==1]['clean_text']
plot_top_ngrams(disaster_corpus, ngram_range=(2,2), n=10, title='Top Bigrams in Disaster Tweets')

# Plot top bigrams for non-disaster tweets
non_disaster_corpus = data[data['target']==0]['clean_text']
plot_top_ngrams(non_disaster_corpus, ngram_range=(2,2), n=10, title='Top Bigrams in Non-Disaster Tweets')

### 9. Comparison of Top Words in Each Class

# Function to get top words
def get_top_words(corpus, n=10):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key = lambda x: x[1], reverse=True)[:n]

# Top words in disaster tweets
top_disaster_words = get_top_words(disaster_corpus)
print("\nTop Words in Disaster Tweets:")
for word, count in top_disaster_words:
    print(f"{word}: {count}")

# Top words in non-disaster tweets
top_non_disaster_words = get_top_words(non_disaster_corpus)
print("\nTop Words in Non-Disaster Tweets:")
for word, count in top_non_disaster_words:
    print(f"{word}: {count}")

### 10. Tweet Length Distribution

# Add a column for tweet length
data['tweet_length'] = data['text'].apply(len)

# Visualize tweet length distribution by class
plt.figure(figsize=(10,6))
sns.histplot(data, x='tweet_length', hue='target', bins=30, kde=True, palette=['blue','red'], alpha=0.6)
plt.title("Tweet Length Distribution by Class", fontsize=14)
plt.xlabel("Tweet Length (characters)", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.legend(labels=['Non-Disaster', 'Disaster'])
plt.show()

### 11. Punctuation Usage

# Count number of exclamation marks
data['exclamations'] = data['text'].apply(lambda x: x.count('!'))
avg_exclamations = data['exclamations'].mean()
print(f"\nAverage number of exclamation marks per tweet: {avg_exclamations:.2f}")

# Visualize exclamation marks by class
plt.figure(figsize=(8,5))
sns.barplot(x=['Non-Disaster', 'Disaster'], y=data.groupby('target')['exclamations'].mean(), palette="coolwarm")
plt.title("Average Number of Exclamation Marks by Class", fontsize=14)
plt.xlabel("Class", fontsize=12)
plt.ylabel("Average Number of Exclamation Marks", fontsize=12)
plt.show()

### 12. Emoji Analysis

# Function to extract emojis
def extract_emojis(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.findall(text)

data['emojis'] = data['text'].apply(extract_emojis)
all_emojis = sum(data['emojis'], [])
print(f"Total emojis: {len(all_emojis)}")
print(f"Unique emojis: {len(set(all_emojis))}")

# Top emojis
emoji_counts = Counter(all_emojis)
top_emojis = emoji_counts.most_common(10)
print("\nTop Emojis:")
for emoji, count in top_emojis:
    print(f"{emoji}: {count}")
