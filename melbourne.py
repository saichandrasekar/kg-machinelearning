import pandas as pd


def run_melbourne():
    melbourne_file_path = "S:/sai-py/workspace/kg-machinelearning/datasets/archive/melb_data.csv"
    melbourne_data = pd.read_csv(melbourne_file_path)
    print(melbourne_data.describe())
