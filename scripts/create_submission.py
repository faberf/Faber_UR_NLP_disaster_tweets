import pandas as pd
from pathlib import Path
import requests
import tempfile
import torch
from transformers import AutoTokenizer, BertConfig
from model import FinetunedBert  # Ensure this matches your actual model import

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

# Preprocessing function
def preprocess_function(texts, tokenizer, max_length=128):
    return tokenizer(
        texts.tolist(), 
        truncation=True, 
        padding="max_length", 
        max_length=max_length, 
        return_tensors='pt'
    )

# Model path
model_path = "runs/run_ff753f6c/checkpoint-1320"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = BertConfig.from_pretrained(model_path)
model = FinetunedBert.from_pretrained(model_path, config=config)
model.eval()  # Set model to eval mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

batch_size = 64  # Adjust this as needed. Lower if still OOM.

with tempfile.TemporaryDirectory() as temp_dir:
    temp_dir_path = Path(temp_dir)

    # Download test data
    test_path = download_csv_from_drive(
        "https://drive.google.com/file/d/1dCXmP3x56a6Jm5lrB0Mu9a4xcz6M8jZK/view?usp=sharing",
        temp_dir_path
    )

    # Load test dataframe
    test_df = pd.read_csv(test_path, sep=',', engine='python')
    # Confirm delimiter if needed
    
    print("\nTest Data:")
    print(test_df.head())

    all_texts = test_df["text"]
    predictions = []

    # Run inference in batches
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i+batch_size]
        encoded_inputs = preprocess_function(batch_texts, tokenizer)
        for key in encoded_inputs:
            encoded_inputs[key] = encoded_inputs[key].to(device)

        with torch.no_grad():
            outputs = model(**encoded_inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(batch_preds)

    # Create submission dataframe
    submission = pd.DataFrame({"id": test_df["id"], "target": predictions})
    submission.to_csv("submission.csv", index=False)
    print("\nSubmission saved to 'submission.csv'.")
    print(submission.head())
