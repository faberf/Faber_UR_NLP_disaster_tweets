import mysql.connector
import time
import json
import numpy as np
from transformers import TrainerCallback, Trainer
import evaluate
import uuid

def create_trainer(model, args, tracker, train_dataset, eval_dataset, tokenizer, hyperparams):
    metrics_to_log = hyperparams["metrics_to_log"]
    classification_metrics = evaluate.combine(metrics_to_log)
    sample_limit = hyperparams["sample_limit"]
    
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=1)
        result = classification_metrics.compute(predictions=preds, references=labels)
        return result
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    logging_callback = MySQLLoggingCallback(
        tracker=tracker,
        model_trainer=trainer,
        sample_limit=sample_limit
    )
    trainer.add_callback(logging_callback)
    
    return trainer

class MySQLTracker:
    """
    Tracks runs, metrics, and artifacts in a MySQL database using UUIDs.
    """
    def __init__(self, host, user, password, database):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.conn.cursor()
        self._create_tables()
        self.run_id = None
        
    def _create_tables(self):
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id CHAR(36) PRIMARY KEY,
            start_time DATETIME,
            end_time DATETIME,
            output_dir VARCHAR(255),
            best_model_path VARCHAR(255),
            duration_seconds FLOAT
        )
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS hyperparameters (
            id CHAR(36) PRIMARY KEY,
            run_id CHAR(36),
            param_name VARCHAR(255),
            param_value TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        )
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS metrics (
            id CHAR(36) PRIMARY KEY,
            run_id CHAR(36),
            step INT,
            metric_name VARCHAR(255),
            metric_value TEXT,
            phase VARCHAR(50),
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        )
        """)

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_checkpoints (
            id CHAR(36) PRIMARY KEY,
            run_id CHAR(36),
            checkpoint_path VARCHAR(255),
            step INT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
        )
        """)

        self.conn.commit()

    def start_run(self, output_dir):
        self.run_id = str(uuid.uuid4())  # Generate a UUID for the run
        start_time = time.strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute(
            "INSERT INTO runs (run_id, start_time, output_dir) VALUES (%s, %s, %s)",
            (self.run_id, start_time, output_dir)
        )
        self.conn.commit()

    def end_run(self, best_model_path):
        end_time = time.strftime('%Y-%m-%d %H:%M:%S')
        self.cursor.execute("SELECT start_time FROM runs WHERE run_id=%s", (self.run_id,))
        start_time = self.cursor.fetchone()[0]
        duration = (time.mktime(time.strptime(end_time, '%Y-%m-%d %H:%M:%S')) - 
                    time.mktime(time.strptime(str(start_time), '%Y-%m-%d %H:%M:%S')))
        self.cursor.execute("""
            UPDATE runs 
            SET end_time=%s, best_model_path=%s, duration_seconds=%s 
            WHERE run_id=%s
        """, (end_time, best_model_path, float(duration), self.run_id))
        self.conn.commit()

    def log_hyperparameters(self, hps):
        for k, v in hps.items():
            param_id = str(uuid.uuid4())  # Generate a UUID for each hyperparameter
            self.cursor.execute(
                "INSERT INTO hyperparameters (id, run_id, param_name, param_value) VALUES (%s, %s, %s, %s)",
                (param_id, self.run_id, k, json.dumps(v))
            )
        self.conn.commit()

    def log_metric(self, step, metric_name, metric_value, phase="train"):
        metric_id = str(uuid.uuid4())  # Generate a UUID for each metric
        metric_value = json.dumps(metric_value)  # Store metric as JSON
        self.cursor.execute(
            "INSERT INTO metrics (id, run_id, step, metric_name, metric_value, phase) VALUES (%s, %s, %s, %s, %s, %s)",
            (metric_id, self.run_id, step, metric_name, metric_value, phase)
        )
        self.conn.commit()

    def log_checkpoint(self, checkpoint_path, step):
        checkpoint_id = str(uuid.uuid4())  # Generate a UUID for each checkpoint
        self.cursor.execute(
            "INSERT INTO model_checkpoints (id, run_id, checkpoint_path, step) VALUES (%s, %s, %s, %s)",
            (checkpoint_id, self.run_id, checkpoint_path, step)
        )
        self.conn.commit()

    def close(self):
        self.cursor.close()
        self.conn.close()



class MySQLLoggingCallback(TrainerCallback):
    def __init__(self, tracker, model_trainer, sample_limit=10):
        super().__init__()
        self.tracker = tracker
        self.model_trainer = model_trainer
        self.sample_limit = sample_limit

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        step = state.global_step

        train_preds = self.model_trainer.predict(self.model_trainer.train_dataset)
        for metric, value in train_preds.metrics.items():
            if metric.startswith("test_"):
                metric = metric[5:]
            self.tracker.log_metric(step, metric, value, phase="train")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        for k, v in logs.items():
            if not k.startswith("eval_"):  
                self.tracker.log_metric(step, k, v, phase="train")
            else:
                self.tracker.log_metric(step, k[5:], v, phase="eval")

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        checkpoint_path = args.output_dir
        checkpoint_dir = f"{checkpoint_path}/checkpoint-{step}"
        self.tracker.log_checkpoint(checkpoint_dir, step)