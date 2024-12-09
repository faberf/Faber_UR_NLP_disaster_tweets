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

# # try larger base model
# hps = deepcopy(default_hps)
# hps["model_checkpoint"] = "sentence-transformers/all-MiniLM-L12-v2"
# train(hps)

# # sweep the number of frozen layers
# for frozen_layers in range(1, 3):
#     hps = deepcopy(default_hps)
#     hps["frozen_layers"] = frozen_layers
#     train(hps)

# sweep the hidden sizes
for hidden_sizes in [[192, 96]]: # [384], [192], [96], 
    hps = deepcopy(default_hps)
    hps["finetune_hidden_sizes"] = hidden_sizes
    train(hps)

