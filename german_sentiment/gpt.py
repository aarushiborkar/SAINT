import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2Tokenizer, GPT2ForSequenceClassification,
    Trainer, TrainingArguments
)
import evaluate
from sklearn.model_selection import train_test_split

DATA_PATH = "data/train.csv"
OUTPUT_DIR = "models/gpt2-sentiment-output"
OUTPUT_CSV = "outputs/test_predictions_gpt2.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

df = pd.read_csv(DATA_PATH)

if "text" not in df.columns and "displayed_text" in df.columns:
    df.rename(columns={"displayed_text": "text"}, inplace=True)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_texts = train_df["text"].tolist()
train_labels = train_df["label_id"].tolist()

val_texts = val_df["text"].tolist()
val_labels = val_df["label_id"].tolist()
val_instance_ids = val_df["instance_id"].tolist()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)

model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.to(DEVICE)

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting fine-tuning...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

predictions_output = trainer.predict(val_dataset)
preds = torch.argmax(torch.tensor(predictions_output.predictions), dim=-1).tolist()

output_df = pd.DataFrame({
    "instance_id": val_instance_ids,
    "text": val_texts,
    "true_label": val_labels,
    "pred_label": preds
})

os.makedirs("outputs", exist_ok=True)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}")
