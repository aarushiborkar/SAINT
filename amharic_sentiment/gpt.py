import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

MODEL_NAME = "Davlan/afro-xlmr-base"
BATCH_SIZE = 8
EPOCHS = 5
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
TEST_PATH = "data/test.csv"
MODEL_SAVE_PATH = "models/afroxlmr_amharic_sentiment"
OUTPUT_CSV = "outputs/test_predictions_gpt_amharic.csv"

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data(path):
    df = pd.read_csv(path)
    return df["displayed_text"].tolist(), df["label_id"].tolist(), df["instance_id"].tolist()

train_texts, train_labels, _ = load_data(TRAIN_PATH)
val_texts, val_labels, _ = load_data(VAL_PATH)
test_texts, test_labels, test_ids = load_data(TEST_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model.to(DEVICE)

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

optimizer = AdamW(model.parameters(), lr=1e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def eval_model(model, data_loader):
    model.eval()
    preds, trues, probs_all = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)

            preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
            probs_all.extend(probs.cpu().numpy().tolist())
    return preds, trues, probs_all

for epoch in range(EPOCHS):
    loss = train_epoch(model, train_loader, optimizer, scheduler)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss:.4f}")

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")

preds, true_labels, probs = eval_model(model, test_loader)

acc = accuracy_score(true_labels, preds)
print(f"Test Accuracy: {acc:.4f}")
print("Classification Report:\n", classification_report(true_labels, preds, digits=4))

output_df = pd.DataFrame({
    "instance_id": test_ids,
    "text": test_texts,
    "true_label": true_labels,
    "pred_label": preds,
    "prob_negative": [p[0] for p in probs],
    "prob_neutral": [p[1] for p in probs],
    "prob_positive": [p[2] for p in probs],
})
os.makedirs("outputs", exist_ok=True)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")
