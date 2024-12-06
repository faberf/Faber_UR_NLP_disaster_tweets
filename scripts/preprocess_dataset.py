import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# Load the CSV file into a pandas DataFrame
file_path = "data/42hacks/train.csv"  # Replace with your actual path
df = pd.read_csv(file_path)

# Select the required columns for X (text) and y (target)
df = df[['text', 'target']].dropna()  # Drop rows with missing text

# Train-test split (80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert pandas DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Create a DatasetDict for compatibility with Hugging Face's Trainer
dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Save the DatasetDict for later use (optional)
dataset.save_to_disk("data/42hacks/hf_dataset")
