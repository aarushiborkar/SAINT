import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd
import ast

MODEL_NAME = "Davlan/afro-xlmr-base"
BATCH_SIZE = 8
EPOCHS = 3
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_LIST = [
    "O", "B-negative", "I-negative", "B-positive", "I-positive", 
    "B-negative object", "I-negative object", "B-positive object", "I-positive object"
]
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

class TokenSentimentDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.samples = []

        for _, row in df.iterrows():
            tokens = ast.literal_eval(row["tokens"])
            tags = ast.literal_eval(row["tags"])
            text = row["text"]
            instance_id = row["instance_id"]

            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_offsets_mapping=True,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()
            offset_mapping = encoding["offset_mapping"].squeeze()

            labels = [LABEL2ID["O"]] * MAX_LEN
            token_idx = 0

            for i in range(1, len(offset_mapping)-1):
                if token_idx >= len(tags):
                    break
                labels[i] = LABEL2ID.get(tags[token_idx], LABEL2ID["O"])
                token_idx += 1

            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": torch.tensor(labels, dtype=torch.long),
                "instance_id": instance_id,
                "tokens": tokens,
                "true_tags": tags,
                "text": text
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def custom_collate(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    instance_ids = [item["instance_id"] for item in batch]
    tokens = [item["tokens"] for item in batch]
    texts = [item["text"] for item in batch]
    true_tags = [item["true_tags"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "instance_id": instance_ids,
        "tokens": tokens,
        "text": texts,
        "true_tags": true_tags,
    }

train_dataset = TokenSentimentDataset("data/token_train.csv")
val_dataset = TokenSentimentDataset("data/token_val.csv")
test_dataset = TokenSentimentDataset("data/token_test.csv")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate)

model = BertForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=LABEL2ID["O"])

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_model(model, loader):
    model.eval()
    all_preds, all_trues = [], []
    analysis_rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=2)

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                pred_labels = []
                true_labels = []

                for j in range(MAX_LEN):
                    if attention_mask[i][j] == 0:
                        continue
                    pred_label = ID2LABEL[preds[i][j].item()]
                    true_label = ID2LABEL[labels[i][j].item()]
                    pred_labels.append(pred_label)
                    true_labels.append(true_label)

                all_preds.extend(pred_labels)
                all_trues.extend(true_labels)

                analysis_rows.append({
                    "instance_id": batch["instance_id"][i],
                    "text": batch["text"][i],
                    "tokens": batch["tokens"][i],
                    "true_tags": batch["true_tags"][i],
                    "pred_tags": pred_labels[:len(batch["true_tags"][i])]
                })

    return all_trues, all_preds, analysis_rows

for epoch in range(EPOCHS):
    loss = train_epoch(model, train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {loss:.4f}")

os.makedirs("models/bert_token_sentiment", exist_ok=True)
model.save_pretrained("models/bert_token_sentiment")
tokenizer.save_pretrained("models/bert_token_sentiment")
print("Model saved!")

true_tags, pred_tags, output_rows = eval_model(model, test_loader)
print(classification_report(true_tags, pred_tags, digits=4))

os.makedirs("outputs", exist_ok=True)
df_out = pd.DataFrame(output_rows)
df_out.to_csv("outputs/token_predictions.csv", index=False)
print("Token-level predictions saved to outputs/token_predictions.csv")
