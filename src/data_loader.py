import pandas as pd
import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/shrikant-temburwar/Loan-Prediction-Dataset/master/train.csv"
RAW_DATA_PATH = os.path.join("data", "raw", "loan_data.csv")

def load_data(filepath=RAW_DATA_PATH):
    """
    Loads data from CSV. Downloads it if not present.
    """
    if not os.path.exists(filepath):
        print(f"File not found at {filepath}. Downloading from {DATA_URL}...")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        try:
            urllib.request.urlretrieve(DATA_URL, filepath)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise

    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    load_data()
