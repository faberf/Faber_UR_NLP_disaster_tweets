import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import requests
import tempfile
import mysql.connector
import uuid

# Download files
def download_csv_from_drive(url, temp_dir):
    file_id = url.split('/d/')[1].split('/view')[0]
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    response = requests.get(download_url)
    response.raise_for_status()
    temp_path = temp_dir / f"{file_id}.csv"
    with open(temp_path, 'wb') as f:
        f.write(response.content)
    return temp_path

# Setup MySQL connection
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
DB_NAME = "Faber_UR_NLP_disaster_tweets/augmentation"

# Connect to MySQL
conn = mysql.connector.connect(
    host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, database = DB_NAME
)
cursor = conn.cursor()


# Ensure the tables exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS tweets (
    id CHAR(36) PRIMARY KEY,
    text TEXT NOT NULL,
    label INT DEFAULT NULL
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS tags (
    id CHAR(36) PRIMARY KEY,
    tweet CHAR(36) NOT NULL,
    tag_name VARCHAR(255) NOT NULL,
    FOREIGN KEY (tweet) REFERENCES tweets(id)
);
""")

conn.commit()

# Temporary directory for downloads
with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir_path = Path(temp_dir)

    # Download and preprocess train data
    train_path = download_csv_from_drive(
        "https://drive.google.com/file/d/10CjxrcgukTCvYmKMTREPy4f7vdZKZz1s/view?usp=sharing",
        temp_dir_path
    )
    train_df = pd.read_csv(train_path)
    train_df = train_df[['text', 'target']].dropna()

    # Convert target to integers
    train_df['target'] = train_df['target'].astype(int)

    # Split into train/eval
    train_data, eval_data = train_test_split(train_df, test_size=0.2, random_state=42)

    # Insert train and eval data into the tweets table
    for _, row in train_data.iterrows():
        tweet_id = str(uuid.uuid4())
        cursor.execute("INSERT INTO tweets (id, text, label) VALUES (%s, %s, %s)", (tweet_id, row['text'], row['target']))
        cursor.execute("INSERT INTO tags (id, tweet, tag_name) VALUES (%s, %s, %s)", (str(uuid.uuid4()), tweet_id, 'original'))
        cursor.execute("INSERT INTO tags (id, tweet, tag_name) VALUES (%s, %s, %s)", (str(uuid.uuid4()), tweet_id, 'original_split_train'))

    for _, row in eval_data.iterrows():
        tweet_id = str(uuid.uuid4())
        cursor.execute("INSERT INTO tweets (id, text, label) VALUES (%s, %s, %s)", (tweet_id, row['text'], row['target']))
        cursor.execute("INSERT INTO tags (id, tweet, tag_name) VALUES (%s, %s, %s)", (str(uuid.uuid4()), tweet_id, 'original'))
        cursor.execute("INSERT INTO tags (id, tweet, tag_name) VALUES (%s, %s, %s)", (str(uuid.uuid4()), tweet_id, 'original_split_eval'))

    # Download and preprocess test data
    test_path = download_csv_from_drive(
        "https://drive.google.com/file/d/1dCXmP3x56a6Jm5lrB0Mu9a4xcz6M8jZK/view?usp=sharing",
        temp_dir_path
    )
    test_df = pd.read_csv(test_path)
    test_df = test_df[['text']].dropna()  # Test has only text field

    for _, row in test_df.iterrows():
        tweet_id = str(uuid.uuid4())
        cursor.execute("INSERT INTO tweets (id, text) VALUES (%s, %s)", (tweet_id, row['text']))
        cursor.execute("INSERT INTO tags (id, tweet, tag_name) VALUES (%s, %s, %s)", (str(uuid.uuid4()), tweet_id, 'original'))
        cursor.execute("INSERT INTO tags (id, tweet, tag_name) VALUES (%s, %s, %s)", (str(uuid.uuid4()), tweet_id, 'original_test'))

    conn.commit()

cursor.close()
conn.close()
