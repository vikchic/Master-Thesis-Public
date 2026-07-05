from __future__ import annotations
from pathlib import Path
import pandas as pd
from parameters import SRC_DIR, DATA_DIR

# This code was created by Gemini (generative AI tool) based on provided data

# paths
DATASET_PATH = Path(f"{DATA_DIR}/processed/semi_clean_dataset.parquet")
MAPPING_PARQUET = Path(f"{DATA_DIR}/processed/club_standardization/club_mapping.parquet")
MAPPING_CSV = Path(f"{DATA_DIR}/processed/club_standardization/club_mapping_list.csv")
OUTPUT_PATH = Path(f"{DATA_DIR}/processed/clean_dataset.parquet")

DATASET_CLUB_COLUMN = "home_club"
DATASET_DISTRICT_COLUMN = "home_district"

# helper functions
def load_mapping() -> pd.DataFrame:
    if MAPPING_PARQUET.exists():
        try:
            return pd.read_parquet(MAPPING_PARQUET)
        except Exception as e:
            print(f"Could not read parquet mapping ({e}). Falling back to CSV.")

    if MAPPING_CSV.exists():
        return pd.read_csv(MAPPING_CSV)

    raise FileNotFoundError(
        f"Could not find mapping file at either:\n"
        f"  {MAPPING_PARQUET}\n"
        f"  {MAPPING_CSV}"
    )

# mapping fucntion
def map_clubs(df=None):
    if df is None:
        print("Loading dataset from file...")
        df = pd.read_parquet(DATASET_PATH)
    else:
        print("Using provided DataFrame...")

    if DATASET_CLUB_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{DATASET_CLUB_COLUMN}' not found in {DATASET_PATH.name}. "
            f"Available columns: {list(df.columns)}"
        )

    print("Loading club mapping...")
    mapping = load_mapping()

    required_mapping_cols = {
        "home_club",
        "normalized_club",
        "canonical_club",
        "canonical_district",
        "canonical_district_name",
        "status",
        "match_method",
        "match_score",
    }
    missing = required_mapping_cols - set(mapping.columns)
    if missing:
        raise ValueError(
            f"Mapping file is missing columns: {sorted(missing)}. "
            f"Available columns: {list(mapping.columns)}"
        )

    mapping_small = mapping[
        [
            "home_club",
            "normalized_club",
            "canonical_club",
            "canonical_district",
            "canonical_district_name",
            "status",
            "match_method",
            "match_score",
        ]
    ].copy()

    mapping_small = mapping_small.rename(
        columns={
            "home_club": "home_club_map_key",
            "normalized_club": "home_club_normalized",
            "canonical_club": "home_club_canonical",
            "canonical_district": "home_district_canonical",
            "canonical_district_name": "home_district_name_canonical",
            "status": "home_club_match_status",
            "match_method": "home_club_match_method",
            "match_score": "home_club_match_score",
        }
    )

    print("Merging mapping into dataset...")
    df = df.rename(columns={DATASET_CLUB_COLUMN: "home_club_raw"})

    df = df.merge(
        mapping_small,
        how="left",
        left_on="home_club_raw",
        right_on="home_club_map_key",
    )

    if DATASET_DISTRICT_COLUMN in df.columns:
        df = df.rename(columns={DATASET_DISTRICT_COLUMN: "home_district_raw"})

    df["home_club"] = df["home_club_canonical"].fillna(df["home_club_raw"])

    if "home_district_raw" in df.columns:
        df["home_district"] = df["home_district_canonical"].fillna(df["home_district_raw"])
    else:
        df["home_district"] = df["home_district_canonical"]

    df["home_club_was_mapped"] = df["home_club_match_status"].isin(["matched", "klubblos"])

    df = df.drop(columns=["home_club_map_key"])

    # report results
    print("Match status counts:")
    print(df["home_club_match_status"].value_counts(dropna=False).to_string())

    # final cleaning based on the results
    df = df[df["home_club_match_status"].isin(["matched", "klubblos"])]
    df = df[df["home_club_canonical"]!="SWE"]
    
    # perform the majority club selection again to avoid the confusion with relay combined teams
    club_idx = df.groupby(["person_id", "home_club_canonical"]).size().reset_index(name="count").groupby("person_id")["count"].idxmax()
    majority_clubs = df.groupby(["person_id", "home_club_canonical"]).size().reset_index(name="count").loc[club_idx, ["person_id", "home_club_canonical"]]
    df = df.merge(majority_clubs.rename(columns={"home_club_canonical": "home_club_fixed"}), on="person_id", how="left")

    # double check the districts as well
    dist_idx = df.groupby(["person_id", "home_district_canonical"]).size().reset_index(name="count").groupby("person_id")["count"].idxmax()
    majority_districts = df.groupby(["person_id", "home_district_canonical"]).size().reset_index(name="count").loc[dist_idx, ["person_id", "home_district_canonical"]]
    df = df.merge(majority_districts.rename(columns={"home_district_canonical": "home_district_fixed"}), on="person_id", how="left")


    df = df[["person_id", 
             "event_year", 
             "year_of_birth",
             "date_of_birth",
             "sex",
             "home_club_fixed",
             "home_district_fixed",
             "num_races",
             "performance"]]    
    
    df = df.rename(columns={"home_club_fixed": "home_club", 
                                    "home_district_fixed": "home_district"})


    print("Saving cleaned dataset...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    
    return df

if __name__ == "__main__":
    map_clubs()
    
