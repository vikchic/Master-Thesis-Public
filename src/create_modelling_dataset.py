import pandas as pd
from pathlib import Path
import numpy as np
from parameters import DATA_DIR


def create_modelling_data(df_clean=None, dropout_cutoff=3, covid_cutoff=None):
    if df_clean is None:
        print("Warning: Dataset not given!")    
        df_clean = pd.read_parquet(PARQUET_DIR)
    
    # ------------- ADD AGE + COUNT YOUTH/ADULTS -------------
    
    # include one's age
    df_clean["age"] = df_clean["event_year"] - df_clean["year_of_birth"]
    print("Age added.")

    # count "parents"
    adult_counts = df_clean[df_clean["age"].between(30, 50)].groupby(
        ["home_club", "event_year"]
    )["person_id"].nunique().reset_index(name="num_adults")
    print("Adult coaches counted.")

    # count all youths
    youth_counts = df_clean[df_clean["age"].between(8, 20)].groupby(
        ["home_club", "event_year"]
    )["person_id"].nunique().reset_index(name="num_youths")
    print("Youths counted.")

    # choose only youths to make the code run faster - later we make a tighter selection
    df_youth = df_clean[df_clean["age"].between(0, 20)].copy()

    # ------------- BIRTH SEASON -------------
    
    # add birth season (1: months 1-4, 2: months 5-8, 3: months 9-12)
    df_youth["birth_season"] = pd.cut(
        df_youth["date_of_birth"].dt.month, 
        bins=[0, 4, 8, 12], 
        labels=[1, 2, 3]
    ).astype(float)
    
    print("Birth season added.")

    # ------------- YOUTH + ADULTS --------------

    # there are a few duplictes - delete them!
    df_youth = df_youth.drop_duplicates(subset=["person_id", "event_year"], keep="first").copy()
    df_youth = df_youth.sort_values(["person_id", "event_year"])

    # merge in adults
    df_youth = df_youth.merge(adult_counts, on=["home_club", "event_year"], how="left")
    df_youth["num_adults"] = df_youth["num_adults"].fillna(0) # if no adults found, it's 0
    df_youth.loc[df_youth["home_district"] == 0, "num_adults"] = 0 # "Klubblös" have no club coaches
    df_youth["num_adults_prev"] = df_youth.groupby("person_id")["num_adults"].shift(1)

    # merge in youths
    df_youth = df_youth.merge(youth_counts, on=["home_club", "event_year"], how="left")
    df_youth["num_youths"] = df_youth["num_youths"].fillna(0)
    df_youth.loc[df_youth["home_district"] == 0, "num_youths"] = 1
    df_youth["num_youths_prev"] = df_youth.groupby("person_id")["num_youths"].shift(1)

    # adult to kid ratio
    df_youth["coach_ratio_prev"] = df_youth["num_adults_prev"]/df_youth["num_youths_prev"]

    # ------------- NUMBER OF RACES + PERFORMANE LAST YEAR -------------

    # find the first year each person had a number of races > 0
    first_race_year = df_youth[df_youth["num_races"] > 0].groupby("person_id")["event_year"].min().reset_index(name="first_year")
    df_youth = df_youth.merge(first_race_year, on="person_id", how="left")
    df_youth["age_at_start"] = df_youth["first_year"] - df_youth["year_of_birth"]

    # find the number of races in the previous year by shifting the number of current races 1 step into the future
    df_youth["num_races_prev"] = df_youth.groupby("person_id")["num_races"].shift(1)
    
    # number of races 2 years ago
    df_youth["num_races_prev2"] = df_youth.groupby("person_id")["num_races"].shift(2)

    # at race difference
    df_youth["race_difference"] = df_youth["num_races_prev"] - df_youth["num_races_prev2"]
    
    print("Number of races added.")

    # add performance from last year
    df_youth["performance_prev"] = df_youth.groupby("person_id")["performance"].shift(1)
    print("Performance added.")

    # ------------- NEWBIE STATUS -------------

    # find how many years have passed since the person started - this can be negative, but those values will be deleted
    df_youth["years_since_start"] = df_youth["event_year"] - df_youth["first_year"]

    # newbies are people who have only been competeing up to 2 years
    df_youth["newbie"] = np.where((df_youth["years_since_start"] == 1), 1, 0)

    # if their first year was 2012 or 2013, we can't be sure they were truly "new" 
    df_youth.loc[df_youth["first_year"] == 2012, "newbie"] = np.nan
    
    print("Newbie status added.")

    # ------------- ACTIVE / DROPOUT -------------

    # create the response variable "active" - 1 when number of races is >=3 and 0 otherwise
    df_youth["active"] = (df_youth["num_races"] > dropout_cutoff).astype(int)
    df_youth["dropout"] = (df_youth["num_races"] <= dropout_cutoff).astype(int)

    # special rule for covid years
    if covid_cutoff is not None:
        covid_mask = df_youth["event_year"].isin([2020])
        df_youth.loc[covid_mask, "active"] = (df_youth.loc[covid_mask, "num_races"] > covid_cutoff).astype(int)
        df_youth.loc[covid_mask, "dropout"] = (df_youth.loc[covid_mask, "num_races"] <= covid_cutoff).astype(int)

    df_youth["active_prev"] = df_youth.groupby("person_id")["active"].shift(1)

    print("Active/dropout status added.")

    # ------------- PEERS -------------

    # find the number of active club peers
    # how many people are there with the same year of birth and the same club in a certain year?
    peer_counts = df_youth.groupby(["home_club", "event_year", "year_of_birth"])["active"].sum().reset_index(name="peer_count")

    p = peer_counts.copy() # count kids who are of the same age/birth year
    p_plus1 = peer_counts.copy() # also count kids 1 year younger
    p_plus1["year_of_birth"] -= 1
    p_minus1 = peer_counts.copy() # also count kids 1 year older
    p_minus1["year_of_birth"] += 1
    # sum all of them
    total_peers = pd.concat([p, p_plus1, p_minus1]).groupby(["home_club", "event_year", "year_of_birth"])["peer_count"].sum().reset_index(name="total_active_peer_group")
    df_youth = df_youth.merge(total_peers, on=["home_club", "event_year", "year_of_birth"], how="left")

    # if there is a value missing, use 0
    # also if a kid is active subtract 1 from their peer group because they are counted towards it
    df_youth["num_peers"] = (df_youth["total_active_peer_group"].fillna(0) - 1).clip(lower=0) 
    df_youth.loc[df_youth["home_district"] == 0, "num_peers"] = 0 # if one is klubblös, one doesn't have any club peers

    # shift by one year into the future to get the number of peers last year
    df_youth["num_peers_prev"] = df_youth.groupby("person_id")["num_peers"].shift(1)
    # two years ago
    df_youth["num_peers_prev2"] = df_youth.groupby("person_id")["num_peers"].shift(2)
    # peer difference
    df_youth["peer_difference"] = df_youth["num_peers_prev"] - df_youth["num_peers_prev2"]
    print("Number of peers added.")

    # number of peers of the same sex
    peer_counts_g = df_youth.groupby(["home_club", "event_year", "year_of_birth", "sex"])["active"].sum().reset_index(name="peer_count_g")
    pg = peer_counts_g.copy()
    pg_plus1 = peer_counts_g.copy()
    pg_plus1["year_of_birth"] -= 1
    pg_minus1 = peer_counts_g.copy()
    pg_minus1["year_of_birth"] += 1

    total_peers_g = pd.concat([pg, pg_plus1, pg_minus1]).groupby(["home_club", "event_year", "year_of_birth", "sex"])["peer_count_g"].sum().reset_index(name="total_sex_peer_group")
    df_youth = df_youth.merge(total_peers_g, on=["home_club", "event_year", "year_of_birth", "sex"], how="left")

    df_youth["num_sex_peers"] = (df_youth["total_sex_peer_group"].fillna(0) - 1).clip(lower=0) 
    df_youth.loc[df_youth["home_district"] == 0, "num_sex_peers"] = 0
    print("Number of peers added.")

    # shift by one year into the future to get the number of peers last year
    df_youth["num_sex_peers_prev"] = df_youth.groupby("person_id")["num_sex_peers"].shift(1)


    # ------------- FULL MODELLING DATASET -------------

    # delete the very young kids - they are not of interest
    df_youth = df_youth[df_youth["age"].between(8, 20)].copy()

    # remove the first years for each person from the data set - i.e. previous year values are missing
    df_model = df_youth.dropna(subset=["num_races_prev", 
                                       "num_peers_prev", 
                                       "performance_prev", 
                                       "num_sex_peers_prev", 
                                       "num_races_prev2",
                                       "num_peers_prev2",
                                       "newbie",
                                       "active_prev",
                                       "num_adults_prev",
                                       "num_youths_prev"])
    
    # only model on people who were active in the previous year
    df_model = df_model[df_model["active_prev"] == 1].copy()
    
    # drop helper columns
    df_model = df_model.drop(columns=["first_year", 
                                      "total_active_peer_group",
                                      "total_sex_peer_group",
                                      "years_since_start",
                                      "active_prev",
                                      "num_races_prev2",
                                      "num_peers_prev2",
                                      "num_peers",
                                      "num_sex_peers",
                                      "num_adults",
                                      "num_youths"])

    # turn sex into a dummy variable
    df_model["sex_male"] = (df_model["sex"] == "M").astype(int)

    print("Number of rows in the modelling data set:", len(df_model))

    # save the data set
    out_dir = Path(f"{DATA_DIR}/processed/modelling")
    file_path = out_dir / "youth_dataset_all.parquet"
    df_model.to_parquet(file_path, index=False)

    return df_model

if __name__ == "__main__":
    PARQUET_DIR = Path(f"{DATA_DIR}/processed/clean_dataset_club_fixed_cut.parquet")
    df_m = create_modelling_data()
