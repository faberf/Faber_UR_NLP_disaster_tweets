import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from nltk.corpus import stopwords

# Load the train and test datasets
train_file = "data/42hacks/train.csv"
test_file = "data/42hacks/test.csv"
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Fill NaN values in 'text' with empty strings
train_data['text'] = train_data['text'].fillna('')
test_data['text'] = test_data['text'].fillna('')

# Define helper functions
def count_stopwords(text):
    stop_words = set(stopwords.words("english"))
    return sum(1 for word in text.split() if word.lower() in stop_words)

def count_urls(text):
    return len(re.findall(r'http\S+', text))

def count_punctuations(text):
    return len(re.findall(r'[^\w\s]', text))

def count_hashtags(text):
    return len(re.findall(r'#\w+', text))

def count_mentions(text):
    return len(re.findall(r'@\w+', text))

def mean_word_length(text):
    words = text.split()
    return np.mean([len(word) for word in words]) if words else 0

# Compute features for train and test data
for df, label in zip([train_data, test_data], ["train", "test"]):
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['unique_word_count'] = df['text'].apply(lambda x: len(set(x.split())))
    df['stop_word_count'] = df['text'].apply(count_stopwords)
    df['url_count'] = df['text'].apply(count_urls)
    df['char_count'] = df['text'].apply(len)
    df['punctuation_count'] = df['text'].apply(count_punctuations)
    df['hashtag_count'] = df['text'].apply(count_hashtags)
    df['mention_count'] = df['text'].apply(count_mentions)
    df['mean_word_length'] = df['text'].apply(mean_word_length)

# Create directory for saving plots
plot_dir = "plots/distribution_identity"
os.makedirs(plot_dir, exist_ok=True)

# Features to compare
features = [
    'word_count', 'unique_word_count', 'stop_word_count',
    'url_count', 'mean_word_length', 'char_count',
    'punctuation_count', 'hashtag_count', 'mention_count'
]

# Plot overlapping histograms and save figures
for feature in features:
    plt.figure(figsize=(10, 6))
    plt.hist(train_data[feature], bins=30, alpha=0.6, label='Train', color='blue', density=True)
    plt.hist(test_data[feature], bins=30, alpha=0.6, label='Test', color='orange', density=True)
    plt.title(f"Distribution of {feature.capitalize()} in Train vs Test", fontsize=14)
    plt.xlabel(feature.capitalize(), fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    # Save plot to file
    plot_path = os.path.join(plot_dir, f"{feature}_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

print(f"Plots saved in '{plot_dir}' directory.")
