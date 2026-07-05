import os

SRC_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = f"{ROOT_DIR}/data"
COVID_CUTOFF = 1
CUTOFF = 3

if __name__ == "__main__":
    print(DATA_DIR)