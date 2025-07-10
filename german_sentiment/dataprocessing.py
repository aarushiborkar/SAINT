import pandas as pd
from transformers import BertTokenizerFast
import ast
import os
from sklearn.model_selection import train_test_split

MODEL_NAME = "bert-base-german-cased"
DATA_PATH = "german_dataset.csv"
MAX_LEN = 128

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

df = pd.read_csv(DATA_PATH)
print("Dataset loaded!")

df = df[["instance_id", "displayed_text", "span_annotation:::negative", "span_annotation:::positive", "label"]]

def combine_spans(row):
    combined = []
    for col in ["span_annotation:::negative", "span_annotation:::positive"]:
        val = row.get(col, "[]")
        try:
            spans = ast.literal_eval(val) if isinstance(val, str) else []
            if isinstance(spans, list):
                combined.extend(spans)
        except Exception as e:
            print(f"Error parsing span: {val} — {e}")
            continue
    return combined

df["all_spans"] = df.apply(combine_spans, axis=1)

def tokenize_and_tag(text, spans):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=MAX_LEN)
    offset_mapping = token_ids['offset_mapping']
    
    tags = ["O"] * len(offset_mapping)

    for span in spans:
        if not isinstance(span, dict) or "start" not in span or "end" not in span or "annotation" not in span:
            continue
        start = span["start"]
        end = span["end"]
        label = span["annotation"]

        for i, (tok_start, tok_end) in enumerate(offset_mapping):
            if tok_start >= end or tok_end <= start:
                continue
            if tok_start >= start and tok_end <= end:
                prefix = "B-" if tags[i] == "O" else "I-"
                tags[i] = f"{prefix}{label}"
    return tokens, tags

records = []
for idx, row in df.iterrows():
    text = row["displayed_text"]
    spans = row["all_spans"]
    instance_id = row["instance_id"]

    if not isinstance(text, str) or not text.strip():
        print(f"Skipping empty text at row {idx}")
        continue

    try:
        tokens, tags = tokenize_and_tag(text, spans)
        if len(tokens) > 0 and len(tags) == len(tokenizer(text, truncation=True, max_length=MAX_LEN)['input_ids']):
            records.append({
                "instance_id": instance_id,
                "text": text,
                "tokens": tokens,
                "tags": tags
            })
        else:
            print(f"Skipping due to token-tag mismatch at row {idx}")
    except Exception as e:
        print(f"Error processing row {idx}: {e}")

processed_df = pd.DataFrame(records)
print("Finished processing spans!")
print("Total usable rows:", len(processed_df))

if len(processed_df) == 0:
    print("No valid rows to split. Exiting.")
    exit()

train_df, temp_df = train_test_split(processed_df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

os.makedirs("data", exist_ok=True)
train_df.to_csv("data/token_train.csv", index=False)
val_df.to_csv("data/token_val.csv", index=False)
test_df.to_csv("data/token_test.csv", index=False)

print("Token-level BIO-tagged data saved:")

