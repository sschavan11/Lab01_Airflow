import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "/opt/airflow/data/Titanic-Dataset.csv"
OUTPUT_DIR = "/opt/airflow/output"

MODEL_PATH = os.path.join(OUTPUT_DIR, "titanic_model.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.txt")


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df.to_json()


def data_preprocessing(data_json):
    df = pd.read_json(data_json)

    selected_cols = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Survived",
    ]

    df = df[selected_cols].copy()
    df = df.dropna(subset=["Survived"])

    return df.to_json()


def train_model(data_json):
    df = pd.read_json(data_json)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    numeric_features = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(report)

    return {
        "accuracy": float(accuracy),
        "model_path": MODEL_PATH,
        "metrics_path": METRICS_PATH,
    }


def evaluate_model(result):
    return (
        f"Model saved at {result['model_path']} | "
        f"Metrics saved at {result['metrics_path']} | "
        f"Accuracy = {result['accuracy']:.4f}"
    )