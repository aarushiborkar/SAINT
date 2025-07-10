import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

df_bert = pd.read_csv("outputs/test_predictions.csv")
df_gpt = pd.read_csv("outputs/test_predictions_gpt2.csv", quotechar='"', quoting=1)
df_flan = pd.read_csv("outputs/test_predictions_flan.csv", quotechar='"', quoting=1)
df_tokens = pd.read_csv("outputs/test_token_attributions.csv")
df_span_preds = pd.read_csv("outputs/token_predictions.csv")

for df in [df_bert, df_gpt, df_flan]:
    df["instance_id"] = df["instance_id"].astype(int)
df_tokens["instance_id"] = df_tokens["instance_id"].astype(str).str.extract(r"(\d+)").astype(int)

token_map = defaultdict(list)
for _, row in df_tokens.iterrows():
    token_map[row["instance_id"]].append((row["token"], row["attribution_score"]))

def format_tokens_colored(tokens):
    if not tokens:
        return ""
    scores = [abs(score) for _, score in tokens]
    max_score = max(scores) or 1e-5
    html = ""
    for token, score in tokens:
        intensity = int(min(abs(score) / max_score, 1.0) * 255)
        color = f"rgb(255, {255 - intensity}, {255 - intensity})"
        html += f'<span style="background-color:{color};padding:1px 2px;margin:1px;border-radius:3px">{token}</span> '
    return html

def fix_label(val):
    try:
        if pd.isna(val): return np.nan
        val_str = str(val).strip().lower()
        if val_str in ["0", "negative", "0.0"]: return 0
        elif val_str in ["1", "neutral", "1.0"]: return 1
        elif val_str in ["2", "positive", "2.0"]: return 2
        elif val_str.isdigit(): return int(val_str)
        elif '.' in val_str: return int(float(val_str))
    except Exception as e:
        print(f"fix_label error for value {val}: {e}")
    return np.nan

df_bert["true_label"] = df_bert["true_label"].apply(fix_label)
df_bert["pred_label"] = df_bert["pred_label"].apply(fix_label)
df_gpt["pred_label"] = df_gpt["pred_label"].apply(fix_label)
df_flan["pred_label"] = df_flan["pred_label"].apply(fix_label)

df = df_bert[["instance_id", "text", "true_label", "pred_label"]].copy()
df = df.rename(columns={"pred_label": "bert_pred"})
df["captum"] = df["instance_id"].map(lambda x: format_tokens_colored(token_map.get(x, [])))

df = df.merge(
    df_gpt[["instance_id", "pred_label"]].rename(columns={"pred_label": "gpt_pred"}),
    on="instance_id", how="left"
)

df = df.merge(
    df_flan[["instance_id", "pred_label", "flan_response"]].rename(columns={"pred_label": "flan_pred"}),
    on="instance_id", how="left"
)

df["flan_pred"] = df["flan_pred"].apply(fix_label)

def make_expander(txt):
    if pd.isna(txt): return ""
    return f'<details><summary>Show full prompt</summary><pre>{txt}</pre></details>'

df["flan_response"] = df["flan_response"].apply(make_expander)

display_df = df[[
    "instance_id", "text", "true_label",
    "bert_pred", "captum",
    "gpt_pred",
    "flan_pred", "flan_response"
]].copy()

display_df.columns = [
    "Instance ID", "Text", "True Label",
    "BERT Prediction", "Captum Attribution",
    "GPT2 Prediction",
    "FLAN Prediction", "FLAN Response"
]

def highlight(row):
    def col(pred_val):
        try:
            if pd.isna(pred_val): return "background-color:#f8d7da"
            if int(pred_val) == int(row["True Label"]): return "background-color:#d4edda"
            return "background-color:#fff3cd"
        except:
            return ""
    return [
        "", "", "", col(row["BERT Prediction"]),
        "", col(row["GPT2 Prediction"]),
        col(row["FLAN Prediction"]), ""
    ]
Path("outputs").mkdir(exist_ok=True)
styled = display_df.style.apply(highlight, axis=1)
styled.to_html("outputs/full_model_comparison.html", escape=False)

print("HTML report saved: outputs/full_model_comparison.html")
