import os
import pandas as pd
import matplotlib.pyplot as plt

SOURCE = "./output/clean_en_pop_metal.csv"
OUT_DIR = "./output/splits"

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(SOURCE)

# compute class distribution and visualize it as a pie chart
class_counts = df["Genre"].value_counts()
print("Class distribution:")
print(class_counts)

plt.figure(figsize=(6, 6))
plt.pie(class_counts, labels=class_counts.index, autopct="%1.1f%%", startangle=140)
plt.title("Genre Distribution in Cleaned Dataset")
plt.axis("equal")
plt.savefig(os.path.join(OUT_DIR, "genre_distribution.png"))

# do a train validation test split of 80/10/10
train = df.sample(frac=0.8, random_state=42)
temp = df.drop(train.index)
val = temp.sample(frac=0.5, random_state=42)
test = temp.drop(val.index)
train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False, encoding="utf-8")
val.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False, encoding="utf-8")
test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False, encoding="utf-8")
print(f"Saved train/val/test splits to {OUT_DIR}")