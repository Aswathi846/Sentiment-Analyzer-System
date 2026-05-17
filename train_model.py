import numpy as np
import os
import mlflow
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
# --- NEW: Import scikit-learn metrics ---
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# ----------------------------------------

def compute_metrics(eval_pred):
    """
    Computes production metrics for imbalanced slang datasets.
    Logs Accuracy, Precision, Recall, and F1-Score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, 
        predictions, 
        average="binary", 
        zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    
    # Return dictionary matching HF expectations
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def train():
    root_path = Path(os.getcwd())
    data_path = root_path / "data" / "slang_reviews.csv"
    tracking_path = root_path / "mlruns"
    
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_path.as_uri())
    
    experiment_name = "Internet_Slang_Sentiment"
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="DistilBERT_ONNX_Ready_V2"):
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data file at {data_path}")
            
        df = pd.read_csv(data_path)
        dataset = Dataset.from_pandas(df).train_test_split(test_size=0.2)
        
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        tokenized_datasets = dataset.map(
            lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=128), 
            batched=True
        )
        
        training_args = TrainingArguments(
            output_dir="./models/checkpoints",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3, # Multi-epoch run to track curve in UI
            logging_steps=10,
            report_to="mlflow", # Automatically pushes Hugging Face logs to MLflow
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            compute_metrics=compute_metrics, # --- NEW: Injected advanced metrics ---
        )
        
        print("🚀 Launching Training Run with Advanced Metrics Tracking...")
        trainer.train()
        
        save_path = root_path / "models" / "sentiment_model"
        save_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(save_path))
        
        print(f"✅ Finished! Run 'mlflow ui' to check performance metrics.")

if __name__ == "__main__":
    train()