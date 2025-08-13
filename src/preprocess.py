import pandas as pd
import os

RAW_DATA_PATH = "data/raw/german_credit_data.csv"
PROCESSED_DATA_PATH = "data/processed/cleand_credit_data.csv"

def load_raw_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df

def clean_data(df):
    #Handing nulls is cols Saving accounts and Checking account
    df['Saving accounts'] = df['Saving accounts'].fillna('no account')
    df['Checking account'] = df['Checking account'].fillna('no account')
    return df
    

def save_processed_data(df):
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index = False)


    if __name__ == "__main__":
        df_raw = load_raw_data()
        df_clean = clean_data(df_raw)
        save_processed_data(df_clean)

        print(f"Processed data saved to {PROCESSED_DATA_PATH}")