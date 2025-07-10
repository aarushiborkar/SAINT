import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

df_bert = pd.read_csv("outputs/test_predictions_amharic.csv")
df_gpt = pd.read_csv("outputs/test_predictions_gpt_amharic.csv", quotechar='"', quoting=1)
df_flan = pd.read_csv("outputs/test_predictions_flan.csv", quotechar='"', quoting=1)
df_tokens = pd.read_csv("outputs/test_token_attributions_amharic.csv")

for name, df in zip(["BERT", "GPT", "FLAN"], [df_bert, df_gpt, df_flan]):
    df["instance_id"] = pd.to_numeric(df["instance_id"], errors="coerce")  
    missing = df[df["instance_id"].isna()]
    if not missing.empty:
        print(f"\n{name} has missing instance_id rows:\n", missing)
    df.dropna(subset=["instance_id"], inplace=True)
    df["instance_id"] = df["instance_id"].astype(int)

df_tokens["instance_id"] = df_tokens["instance_id"].astype(str).str.extract(r"(\d+)").astype(float)
df_tokens.dropna(subset=["instance_id"], inplace=True)
df_tokens["instance_id"] = df_tokens["instance_id"].astype(int)

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
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip().lower()
        if val_str in ["0", "negative", "0.0"]:
            return 0
        elif val_str in ["1", "neutral", "1.0"]:
            return 1
        elif val_str in ["2", "positive", "2.0"]:
            return 2
        elif val_str.isdigit():
            return int(val_str)
        elif '.' in val_str:
            return int(float(val_str))
    except Exception as e:
        print(f"fix_label error for value {val}: {e}")
    return np.nan

df_bert["true_label"] = df_bert["true_label"].apply(fix_label)
df_bert["pred_label"] = df_bert["pred_label"].apply(fix_label)
df_gpt["pred_label"] = df_gpt["pred_label"].apply(fix_label)
df_flan["pred_label"] = df_flan["pred_label"].apply(fix_label)

df = df_bert[[
    "instance_id", "text", "true_label", "pred_label",
    "prob_negative", "prob_neutral", "prob_positive"
]].copy()
df = df.rename(columns={"pred_label": "bert_pred"})
df["captum"] = df["instance_id"].map(lambda x: format_tokens_colored(token_map.get(x, [])))

df = df.merge(
    df_gpt[[
        "instance_id", "pred_label",
        "prob_negative", "prob_neutral", "prob_positive"
    ]].rename(columns={
        "pred_label": "gpt_pred",
        "prob_negative": "gpt_neg",
        "prob_neutral": "gpt_neu",
        "prob_positive": "gpt_pos"
    }),
    on="instance_id", how="left"
)

def update_gpt_pred(row):
    probs = {
        0: row["gpt_neg"] if not pd.isna(row["gpt_neg"]) else -1,
        1: row["gpt_neu"] if not pd.isna(row["gpt_neu"]) else -1,
        2: row["gpt_pos"] if not pd.isna(row["gpt_pos"]) else -1,
    }
    return max(probs, key=probs.get)

df["gpt_pred"] = df.apply(update_gpt_pred, axis=1)

df = df.merge(
    df_flan[["instance_id", "pred_label", "flan_response"]]
    .rename(columns={"pred_label": "flan_pred"}),
    on="instance_id", how="left"
)

df["true_label"] = df["true_label"].apply(fix_label)
df["bert_pred"] = df["bert_pred"].apply(fix_label)
df["gpt_pred"] = df["gpt_pred"].apply(fix_label)
df["flan_pred"] = df["flan_pred"].apply(fix_label)

def make_expander(txt):
    if pd.isna(txt):
        return ""
    return f'<details><summary>Show FLAN output</summary><pre>{txt}</pre></details>'

df["flan_response"] = df["flan_response"].apply(make_expander)

display_df = df[[
    "instance_id", "text", "true_label",
    "bert_pred", "captum",
    "gpt_pred", "gpt_neg", "gpt_neu", "gpt_pos",
    "flan_pred", "flan_response"
]].copy()

display_df.columns = [
    "Instance ID", "Text", "True Label",
    "BERT Prediction", "Captum Attribution",
    "GPT2 Prediction", "GPT2 Neg", "GPT2 Neu", "GPT2 Pos",
    "FLAN Prediction", "FLAN Response"
]

def highlight(row):
    def col(pred_val):
        try:
            if pd.isna(pred_val):
                return "background-color:#f8d7da"  
            if int(pred_val) == int(row["True Label"]):
                return "background-color:#d4edda"  
            return "background-color:#fff3cd"      
        except:
            return ""
    return [
        "", "", "",                      
        col(row["BERT Prediction"]),
        "",                             
        col(row["GPT2 Prediction"]),
        "", "", "",                      
        col(row["FLAN Prediction"]),
        ""                                
    ]

styled = display_df.style.apply(highlight, axis=1)

Path("outputs").mkdir(exist_ok=True)
styled.to_html("outputs/amharic_error_analysis.html", escape=False)

print("\nHTML report saved: outputs/amharic_error_analysis.html")
