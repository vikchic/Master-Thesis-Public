import pandas as pd
from pathlib import Path

# data paths
RAW_DIR = Path("data/raw/excels")
OUT_DIR = Path("data/raw/parquets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# rename our columns
COLUMN_MAP = {
    "personId": "person_id",
    "sex": "sex",
    "dateOfBirth": "date_of_birth", # some years have a full date, some have only the year
    "yearOfBirth": "year_of_birth",
    "nationalityId": "nationality_id",
    "nationalityName": "nationality_name",
    "organisationId": "club_id",
    "organisationName": "club_name",
    "districtId": "district_id",
    "districtName": "district_name",
    "organisationCountryId": "club_country_id",
    "organisationCountryName": "club_country_name",
    "eventId": "event_id",
    "eventName": "event_name",
    "eventDate": "event_date",
    "eventClassification": "event_classification",
    "motionsorientering": "motion",
    "className": "race_class",
    "baseClassId": "base_class_id",
    "resultStatus": "result_status",
    "position": "position",
    "numberOfStarts": "competitors",
    "time": "time",
    "timeBehind": "time_loss",
}

# columns used in the first analysis
CORE_COLUMNS = {
    "person_id",
    "sex",
    "date_of_birth",
    "year_of_birth",
    "event_id",
    "event_date",
    "club_id",
    "club_name",
    "district_id",
    "result_status",
    "position",
    "competitors",
}

# converting excel to parquet
for excel_file in RAW_DIR.glob("*.xlsx"):
    print("Processing:", excel_file.name)

    try:
        # Read the file directly (defaults to the first sheet)
        df = pd.read_excel(excel_file)

        # Process the dataframe
        df = df.rename(columns=COLUMN_MAP)
        df = df[[c for c in CORE_COLUMNS if c in df.columns]]

        df["club_id"] = pd.to_numeric(df["club_id"], errors="coerce")
        df["person_id"] = pd.to_numeric(df["person_id"], errors="coerce")
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")

        if "date_of_birth" in df.columns:
            df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")
            df["year_of_birth"] = df["date_of_birth"].dt.year

        df["year_of_birth"] = pd.to_numeric(df["year_of_birth"], errors="coerce")

        if "club_name" in df.columns:
            df["club_name"] = df["club_name"].astype(str)

        mask_good = (df["club_id"] == 0) & (df["person_id"] != 0)
        df = df.loc[mask_good]

        # Save file
        out_file = OUT_DIR / f"{excel_file.stem}.parquet"
        df.to_parquet(out_file, index=False)
        print("Saved:", out_file)

    except Exception as e:
        print("Failed:", excel_file.name)
        print(e)

print("All files processed.")

