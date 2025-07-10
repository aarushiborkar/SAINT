import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

MODEL_NAME = "google/flan-t5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

TEST_PATH = "data/test.csv"
OUTPUT_CSV = "outputs/test_predictions_flan.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

df = pd.read_csv(TEST_PATH)
texts = df["displayed_text"].tolist()
true_labels = df["label_id"].tolist()
instance_ids = df["instance_id"].tolist()

def make_prompt(text):
    return f"Classify the sentiment of the following text as negative, neutral, or positive:\n\"{text}\""

def extract_label_from_text(text):
    text = text.lower().strip()
    if "negative" in text:
        return 0
    elif "neutral" in text:
        return 1
    elif "positive" in text:
        return 2
    else:
        return -1

preds, responses = [], []
batch_size = 8

for i in tqdm(range(0, len(texts), batch_size), desc="Classifying with FLAN-T5"):
    batch_texts = texts[i:i+batch_size]
    batch_prompts = [make_prompt(t) for t in batch_texts]

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False
        )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    for response in decoded:
        label = extract_label_from_text(response)
        preds.append(label)
        responses.append(response.strip())

output_df = pd.DataFrame({
    "instance_id": instance_ids,
    "text": texts,
    "true_label": true_labels,
    "pred_label": preds,
    "flan_response": responses
})

os.makedirs("outputs", exist_ok=True)
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved predictions to {OUTPUT_CSV}")
