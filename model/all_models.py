import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import src.helper_functions as hf
from src.parameters import DATA_DIR, ROOT_DIR
from cross_validation import run_cross_validation
import pandas as pd
import numpy as np

def fit_and_evaluate_all_models(train_path, test_path, years, features, spline_features, save_dir=None, name=None):
    # load the data
    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)
    df_train['log_num_races_prev'] = np.log1p(df_train['num_races_prev'])
    df_test['log_num_races_prev'] = np.log1p(df_test['num_races_prev'])
    df_train['num_peers_prev20'] = df_train['num_peers_prev'].clip(upper=20)
    df_test['num_peers_prev20'] = df_test['num_peers_prev'].clip(upper=20)

    # paths
    figure_dir = save_dir / "figures"
    table_dir = save_dir / "tables"

    # drop empty columns
    if "C(birth_season)" in features:
        df_train = df_train.dropna(subset=["birth_season", "date_of_birth"])
        df_test = df_test.dropna(subset=["birth_season", "date_of_birth"])
    else:
        df_train = df_train.drop(columns=["birth_season", "date_of_birth"])
        df_test = df_test.drop(columns=["birth_season", "date_of_birth"])

    # find thresholds using cross-validation
    cv_metrics = run_cross_validation(train_path, years, features, spline_features)
    cv_thresholds = cv_metrics["Threshold"].drop("Global (Weighted)").to_dict()
    print("Learned thresholds from cross-validation:")
    print(cv_thresholds)

    # fit the models using training data
    models = hf.fit_all_models_ns(df_train, years, features, spline_features)
    ensemble_train = hf.ensemble_prediction_ns(models, df_train)
    ensemble_test = hf.ensemble_prediction_ns(models, df_test)

    metric_train = hf.calculate_metrics_by_year(ensemble_train, given_thresholds=cv_thresholds,)

    # compute metrics on the test data
    metric_test = hf.calculate_metrics_by_year(ensemble_test, given_thresholds=cv_thresholds, save_dir=table_dir, file_name=name)
    metric_test = metric_test.drop(columns="Threshold")
    print("\nFinal test metrics:")
    print(metric_test)

    hf.plot_pr_curve_by_year(ensemble_test, save_dir=figure_dir, file_name=name)

    df_coef = hf.get_coefficients_table(models)
    hf.plot_coefficient_evolution(df_coef, save_dir=figure_dir, file_name=name)

    if len(spline_features)>0:
        for s in spline_features:
            hf.plot_spline_effect(models, df_test, s, save_dir=figure_dir, file_name=name)

    return metric_train, metric_test

if __name__ == "__main__":
    train_path = Path(f"{DATA_DIR}/processed/modelling/youth_dataset_train.parquet")
    test_path = Path(f"{DATA_DIR}/processed/modelling/youth_dataset_test.parquet")
    years = range(2016, 2026)
    save_dir = Path(f"{ROOT_DIR}/results")
    name = "peers_spl"

    features = ["num_peers_prev"]
    spline_features = ["num_peers_prev"] 
    metric_train, metric_test = fit_and_evaluate_all_models(train_path, 
                                                            test_path, 
                                                            years, 
                                                            features, 
                                                            spline_features,
                                                            save_dir,
                                                            name)