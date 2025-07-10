import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
from tqdm import tqdm

MODEL_PATH = "models/afroxlmr_amharic_sentiment"
TEST_PATH = "data/test.csv"
OUTPUT_ATTR_CSV = "outputs/test_token_attributions_amharic.csv"
MAX_LEN = 128
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

class TestDataset(Dataset):
    def __init__(self, texts, labels, ids, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.ids = ids
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
            add_special_tokens=True
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "instance_id": self.ids[idx],
            "text": text
        }

df = pd.read_csv(TEST_PATH)
texts = df["displayed_text"].tolist()
labels = df["label_id"].tolist()
ids = df["instance_id"].tolist()

dataset = TestDataset(texts, labels, ids, tokenizer, MAX_LEN)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

def forward_func(inputs_embeds, attention_mask):
    return model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask
    ).logits


ig = IntegratedGradients(forward_func)
all_records = []

for batch in tqdm(loader, desc="Explaining"):
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    instance_id = batch["instance_id"][0]
    text = batch["text"][0]

    embeddings = model.base_model.embeddings(input_ids)
    pred_logits = forward_func(embeddings, attention_mask)
    pred_class = torch.argmax(pred_logits, dim=1)

    attributions, delta = ig.attribute(
        inputs=embeddings,
        additional_forward_args=(attention_mask,),
        target=pred_class,
        return_convergence_delta=True
    )

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attr_scores = attributions.sum(dim=2).squeeze(0)

    sep_idx = tokens.index(tokenizer.sep_token) if tokenizer.sep_token in tokens else len(tokens)

    for tok, score in zip(tokens[1:sep_idx], attr_scores[1:sep_idx]):
        all_records.append({
            "instance_id": instance_id,
            "token": tok,
            "attribution_score": score.item()
        })

df_out = pd.DataFrame(all_records)
os.makedirs("outputs", exist_ok=True)
df_out.to_csv(OUTPUT_ATTR_CSV, index=False)
print(f"Token attributions saved to {OUTPUT_ATTR_CSV}")
