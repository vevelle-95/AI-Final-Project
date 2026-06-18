import yaml
import pandas as pd
from pathlib import Path

from training.train import main as train_main

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    with open(BASE_DIR / "configs" / "config.yaml") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(BASE_DIR / "data" / "training.csv")

    train_main(df, config, BASE_DIR)
