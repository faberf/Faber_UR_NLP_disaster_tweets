from train import train
from copy import deepcopy

default_hps = {
    "model_checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
    "frozen_layers": 0,  
    "finetune_hidden_sizes": [], 
    "finetuned_classifier_dropout": 0.1,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "metric_for_best_model": "f1",
    "run_directory": "runs",
    "eval_batch_size": 128,
    "eval_every": 150,
    "metrics_to_log": ["accuracy", "precision", "recall", "f1"],
    "sample_limit": 10,
    "train_tag": "original_split_train",  # Added train tag
    "eval_tag": "original_split_eval"    # Added eval tag
}


# sweep the classifier dropout
for classifier_dropout in [0.15, 0.2, 0.3]:
    hps = deepcopy(default_hps)
    hps["finetuned_classifier_dropout"] = classifier_dropout
    train(hps)

# sweep weight decay
for weight_decay in [0.005, 0.02, 0.03]:
    hps = deepcopy(default_hps)
    hps["weight_decay"] = weight_decay
    train(hps)