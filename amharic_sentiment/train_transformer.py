import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

MODEL_NAME = "Davlan/afro-xlmr-base"

BATCH_SIZE = 8
EPOCHS = 3
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/val.csv"
TEST_PATH = "data/test.csv"
OUTPUT_CSV = "outputs/test_predictions_amharic.csv"
MODEL_SAVE_PATH = "models/afroxlmr_amharic_sentiment"

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
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=True,
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data(path):
    df = pd.read_csv(path)
    return df["displayed_text"].tolist(), df["label_id"].tolist(), df["instance_id"].tolist()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_texts, train_labels, _ = load_data(TRAIN_PATH)
val_texts, val_labels, _ = load_data(VAL_PATH)
test_texts, test_labels, test_ids = load_data(TEST_PATH)

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss()

def train_epoch(model, data_loader, optimizer, scheduler):
    model.train()
    losses = []
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)

def eval_model(model, data_loader):
    model.eval()
    preds, trues, probs_all = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())
    return preds, trues, probs_all

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, optimizer, scheduler)
    print(f"Epoch {epoch+1}/{EPOCHS} — Train Loss: {train_loss:.4f}")

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}")

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
