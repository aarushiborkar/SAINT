import pandas as pd
import ast
from transformers import BertTokenizerFast

MODEL_NAME = "bert-base-german-cased"
tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def extract_spans(tokens, tags, text):
    spans = []
    current_span = []
    current_label = None

    encoding = tokenizer(text, return_offsets_mapping=True, is_split_into_words=False, truncation=True)
    offset_mapping = encoding["offset_mapping"]
    bert_tokens = tokenizer.tokenize(text)

    tok_idx = 0
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag == "O":
            if current_span:
                start_token = current_span[0]
                end_token = current_span[-1]
                start_char = offset_mapping[start_token][0]
                end_char = offset_mapping[end_token][1]
                span_text = text[start_char:end_char]
                spans.append((span_text, start_char, end_char, current_label))
                current_span = []
                current_label = None
        elif tag.startswith("B-"):
            if current_span:
                start_token = current_span[0]
                end_token = current_span[-1]
                start_char = offset_mapping[start_token][0]
                end_char = offset_mapping[end_token][1]
                span_text = text[start_char:end_char]
                spans.append((span_text, start_char, end_char, current_label))
            current_span = [tok_idx]
            current_label = tag[2:]
        elif tag.startswith("I-") and current_label == tag[2:]:
            current_span.append(tok_idx)
        else:
            if current_span:
                start_token = current_span[0]
                end_token = current_span[-1]
                start_char = offset_mapping[start_token][0]
                end_char = offset_mapping[end_token][1]
                span_text = text[start_char:end_char]
                spans.append((span_text, start_char, end_char, current_label))
            current_span = []
            current_label = None
        tok_idx += 1

    if current_span:
        start_token = current_span[0]
        end_token = current_span[-1]
        start_char = offset_mapping[start_token][0]
        end_char = offset_mapping[end_token][1]
        span_text = text[start_char:end_char]
        spans.append((span_text, start_char, end_char, current_label))

    return spans

df = pd.read_csv("outputs/token_predictions.csv")

rows = []
for idx, row in df.iterrows():
    instance_id = row["instance_id"]
    text = row["text"]
    tokens = ast.literal_eval(row["tokens"])
    pred_tags = ast.literal_eval(row["pred_tags"])

    spans = extract_spans(tokens, pred_tags, text)
    for span_text, start, end, label in spans:
        rows.append({
            "instance_id": instance_id,
            "text": text,
            "span_text": span_text,
            "start_char": start,
            "end_char": end,
            "pred_label": label
        })

output_df = pd.DataFrame(rows)
output_df.to_csv("outputs/span_predictions.csv", index=False)
print("Span-level predictions saved to outputs/span_predictions.csv")
