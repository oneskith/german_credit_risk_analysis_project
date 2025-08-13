from src import preprocess, model

def main():
    print("==STEP 1 : Loading raw data ==")
    df_raw = preprocess.load_raw_data()

    print("==STEP 2: Cleaning data ==")
    df_clean = preprocess.clean_data(df_raw)

    print("==STEP 3: Saving processed data ==")
    preprocess.save_processed_data(df_clean)

    print("==STEP 4: Training Random Forest Model ==")
    model.train_model()

    print("=== Pipeline complete ==")


if __name__ == "__main__":
    main()

    