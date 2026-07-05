import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import src.helper_functions as hf
from src.parameters import DATA_DIR, ROOT_DIR
import pandas as pd
import numpy as np

def run_cross_validation(train_path, years, features, spline_features):
    df_train = pd.read_parquet(train_path)
    df_train['log_num_races_prev'] = np.log1p(df_train['num_races_prev'])
    df_train['num_peers_prev20'] = df_train['num_peers_prev'].clip(upper=20)

    if "C(birth_season)" in features:
        df_train = df_train.dropna(subset=["birth_season", "date_of_birth"])
    else:
        df_train = df_train.drop(columns=["birth_season", "date_of_birth"])

    _, _, yearly_metrics = hf.district_cross_validation_ns(df_train, years, features, spline_features, plot="no")
    return yearly_metrics

if __name__ == "__main__":
    train_path = Path(f"{DATA_DIR}/processed/modelling/youth_dataset_train.parquet")
    years = range(2016, 2026)
    features = ["age_at_start",
                "performance_prev",
                "newbie",
                "age",
                "coach_ratio_prev",
                "num_peers_prev20"]
    spline_features = ["age",
                       "performance_prev"]    

    yearly_metrics = run_cross_validation(train_path, years, features, spline_features)
    print(yearly_metrics)