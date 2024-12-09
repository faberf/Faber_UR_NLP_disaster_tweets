from datasets import Dataset, DatasetDict
import mysql.connector
import pandas as pd

# Setup MySQL connection
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
DB_NAME = "Faber_UR_NLP_disaster_tweets"

def load_dataset_from_sql(train_tags, eval_tags):
    """
    Load dataset from MySQL based on lists of train and eval tags.
    
    Parameters:
    - train_tags (list): A list of tags for the training set.
    - eval_tags (list): A list of tags for the evaluation set.
    
    Returns:
    A DatasetDict containing "train" and "eval" splits as Hugging Face Datasets.
    """
    # Connect to MySQL
    conn = mysql.connector.connect(
        host=MYSQL_HOST, user=MYSQL_USER, password=MYSQL_PASSWORD, database=DB_NAME
    )
    cursor = conn.cursor(dictionary=True)

    # Construct placeholders for SQL IN clause
    train_placeholders = ', '.join(['%s'] * len(train_tags))
    eval_placeholders = ', '.join(['%s'] * len(eval_tags))

    # Queries for train and eval
    # Using DISTINCT ensures if a tweet is tagged multiple times, we only see it once.
    train_query = f"""
    SELECT DISTINCT t.id, t.text, t.label
    FROM tweets t
    JOIN tags tg ON t.id = tg.tweet
    WHERE tg.tag_name IN ({train_placeholders});
    """
    eval_query = f"""
    SELECT DISTINCT t.id, t.text, t.label
    FROM tweets t
    JOIN tags tg ON t.id = tg.tweet
    WHERE tg.tag_name IN ({eval_placeholders});
    """

    # Execute queries
    cursor.execute(train_query, tuple(train_tags))
    train_data = cursor.fetchall()

    cursor.execute(eval_query, tuple(eval_tags))
    eval_data = cursor.fetchall()

    # Convert the results to DataFrames
    train_df = pd.DataFrame(train_data)
    eval_df = pd.DataFrame(eval_data)

    # Check for overlapping tweet IDs between train and eval
    if not train_df.empty and not eval_df.empty:
        train_ids = set(train_df['id'])
        eval_ids = set(eval_df['id'])
        overlap = train_ids & eval_ids
        if overlap:
            print(f"Warning: The following tweet IDs are present in both train and eval splits: {overlap}")

    # Drop the 'id' column since it's not needed for the dataset
    if 'id' in train_df.columns:
        train_df = train_df.drop(columns=['id'])
    if 'id' in eval_df.columns:
        eval_df = eval_df.drop(columns=['id'])

    # Convert DataFrames to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df) if not train_df.empty else Dataset.from_dict({})
    eval_dataset = Dataset.from_pandas(eval_df) if not eval_df.empty else Dataset.from_dict({})

    # Close cursor and connection
    cursor.close()
    conn.close()

    # Return as DatasetDict
    return DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset
    })
