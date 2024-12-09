import os
import json
import hashlib
from copy import deepcopy
from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments
)
from accelerate import Accelerator
from load_data import load_dataset_from_sql
from model import FinetunedBert
from sql_tracker import MySQLTracker, create_trainer


def get_deterministic_dir(hps, base_output_dir):
    # Serialize the hyperparameters dictionary to a string
    hps_copy = deepcopy(hps)
    # Remove keys that shouldn't affect hashing
    for key in ["run_directory", "eval_every", "eval_batch_size", "num_train_epochs", "metrics_to_log", "sample_limit"]:
        if key in hps_copy:
            del hps_copy[key]
    hps_string = json.dumps(hps_copy, sort_keys=True)
    # Generate a unique hash from the string
    hps_hash = hashlib.md5(hps_string.encode('utf-8')).hexdigest()[:8]
    # Combine base directory with hash
    return os.path.join(base_output_dir, f"run_{hps_hash}")


def train(hps):
    output_dir = get_deterministic_dir(hps, hps["run_directory"])

    # Initialize accelerator (not using built-in trackers now)
    accelerator = Accelerator()

    # Initialize MySQL tracker
    tracker = MySQLTracker(
        host="localhost",
        user="root",
        password="",
        database="Faber_UR_NLP_disaster_tweets"
    )

    tracker.start_run(output_dir)
    tracker.log_hyperparameters(hps)

    # Load dataset with specified tags
    dataset = load_dataset_from_sql(hps["train_tag"], hps["eval_tag"])

    # Load tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(hps["model_checkpoint"], use_fast=True)
    config = AutoConfig.from_pretrained(hps["model_checkpoint"])
    config.num_labels = 2
    config.finetune_hidden_sizes = hps["finetune_hidden_sizes"]
    config.finetuned_classifier_dropout = hps["finetuned_classifier_dropout"]

    model = FinetunedBert.from_pretrained(hps["model_checkpoint"], config=config)

    # Freeze layers
    if hps["frozen_layers"] > 0:
        layers_to_freeze = [f"layer.{i}" for i in range(hps["frozen_layers"])]
        for name, param in model.bert.named_parameters():
            if any(layer in name for layer in layers_to_freeze) or "embeddings" in name:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=hps["eval_every"],                 
        save_strategy="steps",
        save_steps=hps["eval_every"],                    
        learning_rate=hps["learning_rate"],
        per_device_train_batch_size=hps["batch_size"],
        per_device_eval_batch_size=hps["eval_batch_size"],
        num_train_epochs=hps["num_train_epochs"],
        weight_decay=hps["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model=hps["metric_for_best_model"],
        logging_strategy="steps",
        logging_steps=hps["eval_every"],
        push_to_hub=False
    )

    def tokenize_function(examples):
        result = tokenizer(examples["text"], padding="max_length", truncation=True)
        result["labels"] = examples["label"]  
        return result

    encoded_dataset = dataset.map(tokenize_function, batched=True)

    trainer = create_trainer(model, args, tracker, encoded_dataset["train"], encoded_dataset["eval"], tokenizer, hps)

    resume_from_checkpoint = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save model
    trainer.save_model(output_dir)

    # End run in the tracker
    tracker.end_run(best_model_path=output_dir)
    tracker.close()

    accelerator.end_training()


if __name__ == "__main__":
    hps = {
        "model_checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
        "frozen_layers": 2,  
        "finetune_hidden_sizes": [192], 
        "finetuned_classifier_dropout": 0.1,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "num_train_epochs": 15,
        "weight_decay": 0.01,
        "metric_for_best_model": "f1",
        "run_directory": "runs",
        "eval_batch_size": 64,
        "eval_every": 120,
        "metrics_to_log": ["accuracy", "precision", "recall", "f1"],
        "sample_limit": 10,
        "train_tag": ["original_split_train", "augmented"],  # Added train tag
        "eval_tag": ["original_split_eval"]    # Added eval tag
    }
    train(hps)
