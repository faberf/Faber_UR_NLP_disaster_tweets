from transformers import (
    AutoTokenizer,
    AutoConfig,
    TrainingArguments,
    Trainer,
    BertModel,
    BertPreTrainedModel,
    TrainerCallback
)
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import load_from_disk, load_metric
import numpy as np
from accelerate import Accelerator

import torch
from typing import Optional, Tuple, Union
import json
import hashlib
import os
from copy import deepcopy

class FinetunedBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)

        # Fine-tuning classifier architecture based on `finetune_hidden_sizes`
        finetune_hidden_sizes = getattr(config, "finetune_hidden_sizes", [256])
        classifier_dropout = getattr(config, "finetuned_classifier_dropout", 0.1)

        layers = []
        input_size = config.hidden_size
        for hidden_size in finetune_hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(classifier_dropout))
            input_size = hidden_size
        layers.append(nn.Linear(input_size, config.num_labels))
        self.classifier = nn.Sequential(*layers)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Define all hyperparameters in a single dictionary, including freeze_pretrained
hps = {
    "model_checkpoint": "sentence-transformers/all-MiniLM-L6-v2",
    "frozen_layers": 5,  
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
}

def get_deterministic_dir(hps, base_output_dir):
    # Serialize the hyperparameters dictionary to a string
    hps_string = json.dumps(hps, sort_keys=True)
    # Generate a unique hash from the string
    hps_hash = hashlib.md5(hps_string.encode('utf-8')).hexdigest()[:8]  # Use first 8 characters for brevity
    # Combine base directory with hash
    return os.path.join(base_output_dir, f"run_{hps_hash}")

output_dir = get_deterministic_dir(hps, hps["run_directory"])

accelerator = Accelerator(log_with="all", project_dir = ".")
accelerator.init_trackers("project", config=hps)

# Load preprocessed dataset
dataset = load_from_disk("data/42hacks/hf_dataset")

# Load the tokenizer and model checkpoint
tokenizer = AutoTokenizer.from_pretrained(hps["model_checkpoint"], use_fast=True)

# Load a pre-trained model and specify the number of labels
config = AutoConfig.from_pretrained(hps["model_checkpoint"])
config.num_labels = 2
config.finetune_hidden_sizes = hps["finetune_hidden_sizes"]
config.finetuned_classifier_dropout = hps["finetuned_classifier_dropout"]

model = FinetunedBert.from_pretrained(
    hps["model_checkpoint"],
    config=config
)

# Freeze layers
if hps["frozen_layers"] > 0:
    layers_to_freeze = [f"layer.{i}" for i in range(hps["frozen_layers"])]
    for name, param in model.bert.named_parameters():
        if any(layer in name for layer in layers_to_freeze) or "embeddings" in name:
            param.requires_grad = False

# Print trainable parameters for verification
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

# Define training arguments using the hps dict
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
    push_to_hub=False
)


def tokenize_function(examples):
    result = tokenizer(examples["text"], padding="max_length", truncation=True)
    result["labels"] = examples["target"]  
    return result

encoded_dataset = dataset.map(tokenize_function, batched=True)

metric = load_metric("f1")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


class EvaluateTrainCallback(TrainerCallback):
    def __init__(self, trainer, train_dataset):
        super().__init__()
        self._trainer = trainer
        self._train_dataset = train_dataset

    def on_evaluate(self, args, state, control, **kwargs):
        control = deepcopy(control)

        # Get predictions and loss on the training dataset without triggering `on_evaluate`
        train_predictions = self._trainer.predict(self._train_dataset)

        # Compute training metrics (F1, accuracy, etc.)
        train_metrics = self._trainer.compute_metrics(
            (train_predictions.predictions, train_predictions.label_ids)
        )

        # Add training loss to the metrics
        train_metrics["loss"] = train_predictions.metrics["test_loss"] if "test_loss" in train_predictions.metrics else train_predictions.loss

        # Log all training metrics with a 'train_' prefix
        for key, value in train_metrics.items():
            self._trainer.log({f"train_{key}": value})

        return control


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.add_callback(EvaluateTrainCallback(trainer, train_dataset=encoded_dataset["train"]))

trainer.train(resume_from_checkpoint=False)
trainer.save_model(output_dir)

accelerator.end_training()
