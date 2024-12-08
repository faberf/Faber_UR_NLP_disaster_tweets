from datasets import Dataset, DatasetDict
from sqlalchemy import create_engine
import pandas as pd

# MySQL connection details
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_DB = "Faber_UR_NLP_disaster_tweets"

def load_dataset_from_sql():
    # Establish connection
    engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}")

    # Query the train and eval data
    with engine.connect() as conn:
        train_df = pd.read_sql("SELECT text, target FROM train_eval_data WHERE split='train'", conn)
        eval_df = pd.read_sql("SELECT text, target FROM train_eval_data WHERE split='eval'", conn)

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # Return as DatasetDict
    return DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset
    })