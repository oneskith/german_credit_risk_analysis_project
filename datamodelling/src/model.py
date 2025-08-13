import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
import joblib
import os

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed/cleaned_credit_data.csv")

MODEL_PATH = "../webapp/credit_model.pkl"


def train_model():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X = df.drop(["Unnamed: 0","Risk"], axis = 1)
    y = df["Risk"]

    #Encoding features
    encoders = {}
    for col in X.columns:
        if X[col].dtype == "object":
            enc = LabelEncoder()
            X[col] = enc.fit_transform(X[col])
            encoders[col] = enc

    #Encode target
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    #Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # #Applying SMOTE to trainig set
    # smote = SMOTE(random_state = 42)
    # X_train, y_train = smote.fit_resample(X_train, y_train)

    #Training the model
    model = RandomForestClassifier(n_estimators = 200, max_depth = None, random_state=42, class_weight='balanced')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Clssification report:")
    print(classification_report(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok = True)
    joblib.dump({
        "model": model,
        "feature_encoders": encoders,
        "target_encoder": target_encoder
    }, MODEL_PATH)


if __name__ == "__main__":
    train_model()
    