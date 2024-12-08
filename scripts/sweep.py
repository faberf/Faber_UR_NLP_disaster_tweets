from train import train
from copy import deepcopy

default_hps = {
    "model_checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
    "frozen_layers": 0,  
    "finetune_hidden_sizes": [], 
    "finetuned_classifier_dropout": 0.1,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "num_train_epochs": 5,
    "weight_decay": 0.01,
    "metric_for_best_model": "f1",
    "run_directory": "runs",
    "eval_batch_size": 64,
    "eval_every": 90,
    "metrics_to_log": ["accuracy", "precision", "recall", "f1"],
    "sample_limit": 10
}

# try larger base model
hps = deepcopy(default_hps)
hps["model_checkpoint"] = "sentence-transformers/all-MiniLM-L12-v2"
train(hps)

# sweep the number of frozen layers
for frozen_layers in range(0, 6):
    hps = deepcopy(default_hps)
    hps["frozen_layers"] = frozen_layers
    train(hps)

# sweep the hidden sizes
for hidden_sizes in [[384], [192], [384, 192]]:
    hps = deepcopy(default_hps)
    hps["finetune_hidden_sizes"] = hidden_sizes
    train(hps)

