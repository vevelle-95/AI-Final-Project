from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "data" / "reviews.csv")
print(df["image_paths"].head())
print(repr(df["image_paths"].iloc[0]))