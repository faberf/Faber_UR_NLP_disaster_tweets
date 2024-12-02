import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Load the dataset
file_path = "data/42hacks/train.csv"
data = pd.read_csv(file_path)

# Data Overview
print(f"Dataset shape: {data.shape}")
print(data.head())

# Analyze class distribution
class_counts = data['target'].value_counts()
print("\nClass Distribution:")
print(class_counts)

# Visualize class distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.title("Class Distribution (Disaster vs Non-Disaster)", fontsize=14)
plt.xticks([0, 1], ['Non-Disaster', 'Disaster'], fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Class', fontsize=12)
plt.show()

# Add basic text-based features
data['word_count'] = data['text'].fillna("").apply(lambda x: len(x.split()))
data['char_count'] = data['text'].fillna("").apply(len)

# Basic Statistics
average_word_count = data['word_count'].mean()
average_char_count = data['char_count'].mean()
print("\nBasic Text Statistics:")
print(f"Average word count per tweet: {average_word_count:.2f}")
print(f"Average character count per tweet: {average_char_count:.2f}")

# Visualize word count distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['word_count'], kde=True, bins=30, color='blue')
plt.title("Word Count Distribution", fontsize=14)
plt.xlabel("Word Count", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

# Visualize character count distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['char_count'], kde=True, bins=30, color='green')
plt.title("Character Count Distribution", fontsize=14)
plt.xlabel("Character Count", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.show()

# Word frequency analysis
all_text = " ".join(data['text'].dropna())
word_freq = Counter(all_text.split())
most_common_words = word_freq.most_common(10)

# Display most common words
print("\nMost Common Words:")
for word, freq in most_common_words:
    print(f"{word}: {freq}")

# Visualize most common words
words, counts = zip(*most_common_words)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(words), y=list(counts), palette="coolwarm")
plt.title("Top 10 Most Common Words", fontsize=14)
plt.xlabel("Words", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.show()
