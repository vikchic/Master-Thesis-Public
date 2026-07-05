import pandas as pd
from pathlib import Path
import numpy as np
from parameters import DATA_DIR

def concat_raw_files():
    dfs = []
    raw_data = Path(f"{DATA_DIR}/raw/parquets")
    for file in raw_data.glob("*.parquet"):
        print("Loading:", file.name)
        dfs.append(pd.read_parquet(file))

    df = pd.concat(dfs, ignore_index=True)
    return df

def clean_raw_data(df):
    # ------------- PERFORMANCE -------------

    # remove rows where the competitor didn't even start or the competition didn't happen
    df = df[~df["result_status"].isin(["notStarted", "notActivated", "started", "sportWithdr", None])].copy()

    # add relative position at a competition to the dataset
    df["relative_position"] = 1.0
    mask_competitive = (df["position"] >= 1) & (df["competitors"] > 1)
    df.loc[mask_competitive, "relative_position"] = (df.loc[mask_competitive, "position"] / df.loc[mask_competitive, "competitors"])

    # make sure the performance is within (0,1]
    df["relative_position"] = df["relative_position"].clip(lower=1e-6, upper=1.0)

    # add extra binary variable in case we need it in the end
    df["competitive"] = mask_competitive.astype(int)
    
    print("Performance created.")


    # ------------- CLUBS AND DISTRICTS -------------

    # pick out all results except those where district_id is 1 (mostly USM) - because we don't want this to be the home district for anyone
    df_valid = df[df["district_id"] != 1]

    # and find the majority and make that their home district/club
    dist_idx = df_valid.groupby(["person_id", "district_id"]).size().reset_index(name="count").groupby("person_id")["count"].idxmax()
    majority_districts = df_valid.groupby(["person_id", "district_id"]).size().reset_index(name="count").loc[dist_idx, ["person_id", "district_id"]]
    club_idx = df_valid.groupby(["person_id", "club_name"]).size().reset_index(name="count").groupby("person_id")["count"].idxmax()
    majority_clubs = df_valid.groupby(["person_id", "club_name"]).size().reset_index(name="count").loc[club_idx, ["person_id", "club_name"]]

    # add the home district/club do the dataset
    df_home = df.merge(majority_districts.rename(columns={"district_id": "home_district"}), on="person_id", how="left")
    df_home = df_home.merge(majority_clubs.rename(columns={"club_name": "home_club"}), on="person_id", how="left")

    # delete international people
    df_home = df_home[df_home["home_district"] != 650]
    # delete people without club
    df_home = df_home[df_home["home_district"] != 0]
    df_home = df_home.dropna(subset=["home_district", "home_club"])

    print("Clubs and districts partially cleaned.")


    # ------------- BIRTH AND SEX INCONSISTENCIES --------------

    # some people need to have their year of birth unified
    df_filtered = df_home.dropna(subset=["year_of_birth"])
    birthyear_counts = df_filtered.groupby(["person_id", "year_of_birth"]).size().reset_index(name="count")
    idx = birthyear_counts.groupby("person_id")["count"].idxmax()
    majority_birthyears = birthyear_counts.loc[idx, ["person_id", "year_of_birth"]]

    # replace the the old inconsistent values with a fixed version
    birthyear_map = majority_birthyears.set_index("person_id")["year_of_birth"]
    df_home["year_of_birth"] = df_home["person_id"].map(birthyear_map)
    df_home = df_home.dropna(subset=["year_of_birth"])

    # fix birthdates for those who have them, and add the column if it doesn't exist
    if "date_of_birth" not in df_home.columns:
        df_home["date_of_birth"] = pd.NaT
        
    # convert to datetime
    df_home["date_of_birth"] = pd.to_datetime(df_home["date_of_birth"], errors="coerce")

    # fill missing dates of birth using the person's other valid rows
    df_filtered = df_home.dropna(subset=["date_of_birth"])
    if not df_filtered.empty:
        birthday_counts = df_filtered.groupby(["person_id", "date_of_birth"]).size().reset_index(name="count")
        idx = birthday_counts.groupby("person_id")["count"].idxmax()
        majority_birthdays = birthday_counts.loc[idx, ["person_id", "date_of_birth"]]
        
        birthday_map = majority_birthdays.set_index("person_id")["date_of_birth"]
        
        # fill in missing rows only for people who have a known date of birth, and leave the rest as NaT
        df_home["date_of_birth"] = df_home["person_id"].map(birthday_map).combine_first(df_home["date_of_birth"])

    # some people need to have their sex unified
    df_filtered = df_home.dropna(subset=["sex"])
    sex_counts = df_filtered.groupby(["person_id", "sex"]).size().reset_index(name="count")
    idx = sex_counts.groupby("person_id")["count"].idxmax()
    majority_sex = sex_counts.loc[idx, ["person_id", "sex"]]

    # replace the the old inconsistent values with a fixed version
    sex_map = majority_sex.set_index("person_id")["sex"]
    df_home["sex"] = df_home["person_id"].map(sex_map)
    df_home = df_home.dropna(subset=["sex"])

    print("Sex and birth dates cleaned.")


    # ------------- CLEAN DATASET -------------

    # group by person and year, add number of races in a particular year
    df_home["event_year"] = df_home["event_date"].dt.year 
    
    # added date_of_birth
    df_races = df_home.groupby(
        ["person_id", "event_year", "year_of_birth", "date_of_birth", "sex", "home_club", "home_district"],
        dropna=False
    ).size().reset_index(name="num_races")

    # add the yearly performance - mean relative position
    df_home['performance_score'] = 1 - df_home['relative_position']
    performance_yearly = df_home.groupby(["person_id", "event_year"])["performance_score"].mean().reset_index(name="performance")

    # define a grid with every person and every year
    all_years = list(range(2012, 2026))
    all_people = df_races["person_id"].unique()
    full_index = pd.MultiIndex.from_product([all_people, all_years], names=["person_id", "event_year"])

    # construct a data set
    df_clean = pd.DataFrame(index=full_index).reset_index()

    # merge the available data into the data set
    df_clean = df_clean.merge(df_races, on=["person_id", "event_year"], how="left")
    df_clean = df_clean.merge(performance_yearly, on=["person_id", "event_year"], how="left")

    # if there are races missing, replace NaN with 0 (for performance, no competitions = no performance)
    df_clean["num_races"] = df_clean["num_races"].fillna(0)
    df_clean["performance"] = df_clean["performance"].fillna(0)

    # fill in the "permanent info" for each person as well, because right now it is missing in the rows which were added
    static_cols = ["year_of_birth", "date_of_birth", "sex", "home_club", "home_district"]
    for col in static_cols:
        df_clean[col] = df_clean.groupby("person_id")[col].ffill()
        df_clean[col] = df_clean.groupby("person_id")[col].bfill()

    df_clean = df_clean.sort_values(["person_id", "event_year"])

    # save the new clean dataset permanently
    out_dir = Path(f"{DATA_DIR}/processed")
    df_clean.to_parquet(out_dir / "semi_clean_dataset.parquet", index=False)

    print("Dataset saved permanently!")

    return df_clean

def load_and_clean_data():
    df = concat_raw_files()
    df_clean = clean_raw_data(df)
    return df_clean

if __name__ == "__main__":
    df0 = load_and_clean_data()
    print(df0)