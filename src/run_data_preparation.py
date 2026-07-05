from create_clean_dataset import load_and_clean_data
from create_modelling_dataset import create_modelling_data
from data_segmentation import divide_data
from cut_clean_dataset import cut_clean_data
from apply_club_mapping import map_clubs
from parameters import CUTOFF, COVID_CUTOFF

if __name__ == "__main__":
    print("Starting full data preparation pipeline...")
    df_clean0 = load_and_clean_data()
    df_clean = map_clubs(df_clean0)
    df_model = create_modelling_data(df_clean, dropout_cutoff=CUTOFF, covid_cutoff=COVID_CUTOFF)
    df_train, df_test = divide_data(df_model)
    print("\nPipeline finished successfully!")
    