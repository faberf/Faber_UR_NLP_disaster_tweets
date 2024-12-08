import pandas as pd
from sklearn.model_selection import train_test_split

from pathlib import Path
import requests
import tempfile
from sqlalchemy import create_engine, text


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
DB_NAME = "Faber_UR_NLP_disaster_tweets"  # New database name

engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/")

# Create a new database if it doesn't exist
with engine.connect() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};"))

# Connect to the new database
engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{DB_NAME}")

# Ensure the database and tables exist
with engine.connect() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS train_eval_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        text TEXT NOT NULL,
        target INT NOT NULL,  -- Change target type to INT
        split ENUM('train', 'eval') NOT NULL
    );
    """))
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS test_data (
        id INT AUTO_INCREMENT PRIMARY KEY,
        text TEXT NOT NULL
    );
    """))

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

    # Add split column
    train_df['split'] = 'train'

    # Split into train/eval
    train_data, eval_data = train_test_split(train_df, test_size=0.2, random_state=42)
    train_data['split'] = 'train'
    eval_data['split'] = 'eval'

    # Concatenate and save train/eval to MySQL
    combined_df = pd.concat([train_data, eval_data])
    combined_df.to_sql('train_eval_data', con=engine, if_exists='append', index=False)

    # Download and preprocess test data
    test_path = download_csv_from_drive(
        "https://drive.google.com/file/d/1dCXmP3x56a6Jm5lrB0Mu9a4xcz6M8jZK/view?usp=sharing", 
        temp_dir_path
    )
    test_df = pd.read_csv(test_path)
    test_df = test_df[['text']].dropna()  # Test has only text field
    test_df.to_sql('test_data', con=engine, if_exists='append', index=False)
