import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification
from captum.attr import IntegratedGradients
from tqdm import tqdm

MODEL_PATH = "models/bert_token_sentiment" 
TEST_PATH = "data/token_test.csv"           
OUTPUT_ATTR_CSV = "outputs/token_attributions.csv"
MAX_LEN = 128
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_LIST = [
    "O", "B-negative", "I-negative", "B-positive", "I-positive",
    "B-negative object", "I-negative object", "B-positive object", "I-positive object"
]
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

model = BertForTokenClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)

class TestDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = eval(row["tokens"])     
        tags = eval(row["tags"]) 
        instance_id = row["instance_id"]

        encoding = self.tokenizer(tokens,
                                  is_split_into_words=True,
                                  truncation=True,
                                  padding="max_length",
                                  max_length=self.max_len,
                                  return_offsets_mapping=True,
                                  return_tensors="pt")

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "offset_mapping": encoding["offset_mapping"].squeeze(),
            "tokens": tokens,
            "tags": tags,
            "instance_id": instance_id,
            "text": text
        }

df_test = pd.read_csv(TEST_PATH)

test_dataset = TestDataset(df_test, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

def forward_func(inputs_embeds, attention_mask, target_token_idx, target_label_idx):
    outputs = model.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    sequence_output = outputs.last_hidden_state  
    logits = model.classifier(sequence_output)   
    selected_logit = logits[0, target_token_idx, target_label_idx]
    return selected_logit.unsqueeze(0)

ig = IntegratedGradients(forward_func)
all_records = []

for batch in tqdm(test_loader, desc="Explaining tokens"):
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    tokens = batch["tokens"][0]
    tags = batch["tags"][0]
    instance_id = batch["instance_id"][0]
    text = batch["text"][0]

    embeddings = model.bert.embeddings(input_ids)
    pred_logits = forward_func(embeddings, attention_mask, 0, 0)  
    pred_labels = torch.argmax(model.classifier(model.bert(input_ids=input_ids).last_hidden_state), dim=2).squeeze(0).tolist()
    bert_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    skip_tokens = {"[CLS]", "[SEP]", "[PAD]"}

    for token_idx, (bert_tok, pred_label) in enumerate(zip(bert_tokens, pred_labels)):
        if bert_tok in skip_tokens:
            continue

        attributions, delta = ig.attribute(
            inputs=embeddings,
            additional_forward_args=(attention_mask, token_idx, pred_label),
            return_convergence_delta=True
        )

        token_attr_score = attributions[0, token_idx].sum().item()

        all_records.append({
            "instance_id": instance_id,
            "token": bert_tok,
            "pred_tag": ID2LABEL[pred_label],
            "attribution_score": token_attr_score
        })

os.makedirs("outputs", exist_ok=True)
df_out = pd.DataFrame(all_records)
df_out.to_csv(OUTPUT_ATTR_CSV, index=False)

print(f"Token-level attributions saved to {OUTPUT_ATTR_CSV}")
