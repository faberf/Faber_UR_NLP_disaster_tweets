import os
import random
import torch
import pandas as pd
import uuid
from datasets import DatasetDict
from transformers import AutoTokenizer, BertConfig
from load_data import load_dataset_from_sql
from model import FinetunedBert  # Import your custom model
from langchain_ollama import ChatOllama
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import mysql.connector


class Augmenter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.false_positives = None
        self.false_negatives = None
        self.llm = None
        self.output_parser = None

    def setup(self, dataset_dict):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        config = BertConfig.from_pretrained(self.model_path)
        self.model = FinetunedBert.from_pretrained(self.model_path, config=config)

        # Tokenize the dataset
        train_dataset = dataset_dict["train"]

        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

        train_dataset_enc = train_dataset.map(preprocess_function, batched=True)
        train_dataset_enc = train_dataset_enc.remove_columns(["text"]).with_format("torch")

        # Predict using the custom model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        train_dataloader = torch.utils.data.DataLoader(train_dataset_enc, batch_size=32)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        # Convert predictions to pandas
        train_df = train_dataset.to_pandas()
        train_df["pred"] = all_preds
        train_df["label"] = all_labels

        # Identify misclassifications
        self.false_positives = train_df[(train_df["label"] == 0) & (train_df["pred"] == 1)]
        self.false_negatives = train_df[(train_df["label"] == 1) & (train_df["pred"] == 0)]

    def generate_tweets(self):
        # Construct the few-shot prompt
        few_shot_prompt = (
            "We are analyzing tweets to help detect actual disasters quickly.\n\n"
            "Below are examples of tweets that are difficult to classify. Some describe real disasters, while others do not.\n\n"
            "=== Real disasters ===\n"
        )
        for _, row in self.false_negatives.sample(min(10, len(self.false_negatives))).iterrows():
            few_shot_prompt += f"- {row['text']}\n"

        few_shot_prompt += "\n=== Not disasters ===\n"
        for _, row in self.false_positives.sample(min(10, len(self.false_positives))).iterrows():
            few_shot_prompt += f"- {row['text']}\n"

        few_shot_prompt += (
            "\nUsing the above examples as guidance, please produce two tweets that are similarly difficult to classify:"
            "\n1. A tweet that announces a real, non-metaphorical disaster event (disaster_tweet)."
            "\n2. A tweet that does not describe a real disaster (non_disaster_tweet)."
            "\nDo not use `BREAKING` or `just` in your tweets. Your output will help in timely detection and response to real disasters. "
            "\n The tweets should share some similarities with each other, even though they are opposites. For example, they might both contain a specific keyword or phrase but the non-disaster uses it metaphorically."
        )

        # Initialize the LLM and parser if not already done
        if not self.llm:
            self.llm = ChatOllama(
                model="llama3.2",
                temperature=0.9,
                num_predict=256
            )
        if not self.output_parser:
            response_schemas = [
                ResponseSchema(
                    name="disaster_tweet_explanation",
                    description="An explanation why the disaster tweet might seem like a non disaster tweet at first, but actually is a disaster tweet."
                ),
                ResponseSchema(
                    name="non_disaster_tweet_explanation",
                    description="An explanation why the non disaster tweet might seem like a disaster tweet at first, but actually is a non disaster tweet."
                ),
                ResponseSchema(
                    name="disaster_tweet",
                    description="A single tweet that announces a real, non-metaphorical disaster event."
                ),
                ResponseSchema(
                    name="non_disaster_tweet",
                    description="A single tweet that does not describe a real disaster."
                )
            ]
            self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        format_instructions = self.output_parser.get_format_instructions()
        final_prompt = f"{few_shot_prompt}\n\n{format_instructions}"

        messages = [
            ("system", "You are a skilled communicator who creates structured and concise responses for disaster detection tasks."),
            ("human", final_prompt)
        ]

        success = False
        while not success:
            try:
                output_dict = (self.llm | self.output_parser).invoke(messages)
                success = True
            except Exception as e:
                print(f"Error occurred: {e}. Retrying...")

        # Return generated tweets
        return output_dict["disaster_tweet"], output_dict["non_disaster_tweet"]


if __name__ == "__main__":
    model_path = "runs/run_6e46caef/checkpoint-1910"
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "Faber_UR_NLP_disaster_tweets",
    }

    # Initialize augmenter
    augmenter = Augmenter(model_path=model_path)

    # Load the dataset
    dataset_dict = load_dataset_from_sql(
        train_tags=["original_split_train"],
        eval_tags=["original_split_train"]  # Placeholder, eval not used
    )

    # Setup augmenter (computes false positives and negatives)
    augmenter.setup(dataset_dict)

    # Connect to the database
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Generate 2000 positive and negative tweets live
    for _ in range(2000):
        pos_tweet, neg_tweet = augmenter.generate_tweets()
        print(f"Positive tweet: {pos_tweet}")
        print(f"Negative tweet: {neg_tweet}")

        # Insert positive tweet
        pos_uuid = str(uuid.uuid4())
        cursor.execute("INSERT INTO tweets (id, text, label) VALUES (%s, %s, %s)", (pos_uuid, pos_tweet, 1))
        tag_uuid = str(uuid.uuid4())
        cursor.execute("INSERT INTO tags (id, tweet, tag_name) VALUES (%s, %s, %s)", (tag_uuid, pos_uuid, "augmented"))

        # Insert negative tweet
        neg_uuid = str(uuid.uuid4())
        cursor.execute("INSERT INTO tweets (id, text, label) VALUES (%s, %s, %s)", (neg_uuid, neg_tweet, 0))
        tag_uuid = str(uuid.uuid4())
        cursor.execute("INSERT INTO tags (id, tweet, tag_name) VALUES (%s, %s, %s)", (tag_uuid, neg_uuid, "augmented"))

        # Commit after each insertion
        conn.commit()

    cursor.close()
    conn.close()

    print("Inserted 2000 positive and 2000 negative tweets into the database.")
