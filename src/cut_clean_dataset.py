from pathlib import Path
from parameters import DATA_DIR
import pandas as pd

def cut_clean_data(df_clean0=None):
    if df_clean0 is None:
        in_dir = Path(f"{DATA_DIR}/processed")
        df_clean0 = pd.read_parquet(in_dir/"clean_dataset_club_fixed.parquet")

    df_clean0 = df_clean0[df_clean0["home_club_match_status"].isin(["matched", "klubblos"])]
    df_clean0 = df_clean0[df_clean0["home_club_canonical"]!="SWE"]
    
    # perform the majority club selection again to avoid the confusion with relay combined teams
    club_idx = df_clean0.groupby(["person_id", "home_club_canonical"]).size().reset_index(name="count").groupby("person_id")["count"].idxmax()
    majority_clubs = df_clean0.groupby(["person_id", "home_club_canonical"]).size().reset_index(name="count").loc[club_idx, ["person_id", "home_club_canonical"]]
    df_clean0 = df_clean0.merge(majority_clubs.rename(columns={"home_club_canonical": "home_club_fixed"}), on="person_id", how="left")

    # double check the districts as well
    dist_idx = df_clean0.groupby(["person_id", "home_district_canonical"]).size().reset_index(name="count").groupby("person_id")["count"].idxmax()
    majority_districts = df_clean0.groupby(["person_id", "home_district_canonical"]).size().reset_index(name="count").loc[dist_idx, ["person_id", "home_district_canonical"]]
    df_clean0 = df_clean0.merge(majority_districts.rename(columns={"home_district_canonical": "home_district_fixed"}), on="person_id", how="left")


    df_clean0 = df_clean0[["person_id", 
                           "event_year", 
                           "year_of_birth",
                           "date_of_birth", 
                           "sex",
                           "home_club_fixed",
                           "home_district_fixed",
                           "num_races",
                           "performance"]]    
    
    df_clean0 = df_clean0.rename(columns={"home_club_fixed": "home_club", 
                                    "home_district_fixed": "home_district"})
    
    out_dir = Path(f"{DATA_DIR}/processed")
    file_path = out_dir / "clean_dataset_club_fixed_cut.parquet"
    df_clean0.to_parquet(file_path, index=False)
    return df_clean0

if __name__ == "__main__":
    cut_clean_data()