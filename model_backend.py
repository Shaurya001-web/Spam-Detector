from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class SpamModelArtifacts:
    model: RandomForestClassifier
    vectorizer: TfidfVectorizer
    accuracy: float
    row_count: int
    ham_count: int
    spam_count: int


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="latin-1")

    if df.shape[1] < 2:
        raise ValueError("Dataset must contain at least 2 columns: label and message.")

    # Keep first two columns and standardize names
    df = df.iloc[:, :2].copy()
    df.columns = ["label", "message"]

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)

    df["message"] = df["message"].fillna("").astype(str).apply(clean_text)
    df = df[df["message"].str.len() > 0].copy()

    if df.empty:
        raise ValueError("No valid rows found after cleaning labels/messages.")

    return df


def train_spam_model(csv_path: str | Path) -> SpamModelArtifacts:
    df = load_dataset(csv_path)

    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df["message"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))

    ham_count = int((df["label"] == 0).sum())
    spam_count = int((df["label"] == 1).sum())

    return SpamModelArtifacts(
        model=model,
        vectorizer=vectorizer,
        accuracy=accuracy,
        row_count=len(df),
        ham_count=ham_count,
        spam_count=spam_count,
    )


def predict_spam(text: str, artifacts: SpamModelArtifacts) -> tuple[str, float]:
    processed = clean_text(text)
    vector = artifacts.vectorizer.transform([processed])

    pred = int(artifacts.model.predict(vector)[0])

    confidence = 0.0
    if hasattr(artifacts.model, "predict_proba"):
        probabilities = artifacts.model.predict_proba(vector)[0]
        confidence = float(max(probabilities))

    label = "Spam" if pred == 1 else "Not Spam"
    return label, confidence
