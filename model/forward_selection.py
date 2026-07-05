import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import src.helper_functions as hf
from src.parameters import DATA_DIR
import pandas as pd
import numpy as np

def run_forward_selection(years, candidate_features, spline_candidates, train_path):
    df_train = pd.read_parquet(train_path)

    if "C(birth_season)" in candidate_features:
        df_train = df_train.dropna(subset=["birth_season", "date_of_birth"])
    else:
        df_train = df_train.drop(columns=["birth_season", "date_of_birth"])

    df_train['log_num_races_prev'] = np.log1p(df_train['num_races_prev'])
    df_train['num_peers_prev20'] = df_train['num_peers_prev'].clip(upper=20)
    
    features, spline_features, prauc, history = hf.forward_feature_selection_ns(df_train, years, candidate_features, spline_candidates)
    
    return features, spline_features, prauc, history


if __name__ == "__main__":
    train_path = Path(f"{DATA_DIR}/processed/modelling/youth_dataset_train.parquet")
    test_path = Path(f"{DATA_DIR}/processed/modelling/youth_dataset_test.parquet")

    years = range(2016, 2026)

    candidate_features = ["log_num_races_prev",
                          "age",
                          "performance_prev",
                          "age_at_start",
                          "newbie",
                          "coach_ratio_prev",
                          "num_peers_prev20"]

    spline_candidates = ["age",
                         "age_at_start",
                         "performance_prev"]
    
    features, spline_features, prauc, history = run_forward_selection(years, candidate_features, spline_candidates, train_path)
    print(history)