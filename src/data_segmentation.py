import pandas as pd
from pathlib import Path
import numpy as np
from parameters import DATA_DIR


def divide_data(df_model=None):
    if df_model is None:
        print("Warning: Dataset not given!")
        df_model = pd.read_parquet(PARQUET_DIR)


    # -------------- DIVIDE INTO TRAIN + TEST DATA --------------

    test_districts = [4, 12, 15, 5]

    df_test = df_model[df_model["home_district"].isin(test_districts)]
    df_train = df_model[~df_model["home_district"].isin(test_districts)]

    # check the amount od data in each dataset
    print("Training rows:", len(df_train), ", and percentage of the whole dataset:", round(len(df_train)/len(df_model)*100,2), "%")
    print("Test rows:", len(df_test), ", and percentage of the whole dataset:", round(len(df_test)/len(df_model)*100,2), "%")

    out_dir = Path(f"{DATA_DIR}/processed/modelling") 
    file_path_train = out_dir / "youth_dataset_train.parquet"
    file_path_test = out_dir / "youth_dataset_test.parquet"

    df_train.to_parquet(file_path_train, index=False)
    df_test.to_parquet(file_path_test, index=False)

    return df_train, df_test

if __name__ == "__main__":
    PARQUET_DIR = Path(f"{DATA_DIR}/processed/modelling/youth_dataset_all.parquet")
    df_tr, df_te = divide_data()

