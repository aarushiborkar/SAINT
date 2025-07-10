import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("german_dataset.csv", sep=",")

df = df[["instance_id", "displayed_text", "label"]]

label_map = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}
df["label_id"] = df["label"].map(label_map)

df_train, df_temp = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["label_id"]
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.5, random_state=42, stratify=df_temp["label_id"]
)

df_train.to_csv("data/train.csv", index=False)
df_val.to_csv("data/val.csv", index=False)
df_test.to_csv("data/test.csv", index=False)

print("Data preparation complete!")
