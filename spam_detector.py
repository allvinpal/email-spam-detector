"""Train and evaluate a text-only spam classifier on the bundled SMS dataset."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DEFAULT_DATASET = Path(__file__).resolve().with_name("spam.csv")


def load_dataset(path=DEFAULT_DATASET):
    """Validate the input and remove repeated messages before splitting."""
    data = pd.read_csv(path)
    if not {"Category", "Message"}.issubset(data.columns):
        raise ValueError("Dataset must contain Category and Message columns.")
    data = data[["Category", "Message"]].copy()
    if data.isnull().any().any():
        raise ValueError("Category and Message must not contain missing values.")
    data["Category"] = data["Category"].astype(str).str.strip().str.lower()
    data["Message"] = data["Message"].astype(str).str.strip()
    if not data["Category"].isin(["ham", "spam"]).all():
        raise ValueError("Category values must be ham or spam.")
    if data["Message"].eq("").any():
        raise ValueError("Messages must not be blank.")
    if data.groupby("Message")["Category"].nunique().gt(1).any():
        raise ValueError("Identical messages have conflicting labels.")
    data = data.drop_duplicates(subset="Message").reset_index(drop=True)
    counts = data["Category"].value_counts()
    if len(counts) != 2 or counts.min() < 5:
        raise ValueError("Dataset needs at least five unique messages per class.")
    return data


def build_model():
    return Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", MultinomialNB()),
    ])


def train_and_evaluate(data, random_state=42):
    """Fit vocabulary on training data only; keep a reproducible holdout."""
    x_train, x_test, y_train, y_test = train_test_split(
        data["Message"], data["Category"], test_size=0.2,
        random_state=random_state, stratify=data["Category"],
    )
    model = build_model()
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = {
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "accuracy": accuracy_score(y_test, predicted),
        "classification_report": classification_report(
            y_test, predicted, labels=["ham", "spam"],
            output_dict=True, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predicted, labels=["ham", "spam"],
        ).tolist(),
        "label_order": ["ham", "spam"],
    }
    return model, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--message", help="Classify a message after evaluation.")
    args = parser.parse_args()
    if args.message is not None and not args.message.strip():
        parser.error("--message must not be blank")
    try:
        data = load_dataset(args.data)
        model, metrics = train_and_evaluate(data)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps(metrics, indent=2))
    if args.message is not None:
        # Evaluation stays held out; prediction uses all validated data.
        model.fit(data["Message"], data["Category"])
        print(f"Prediction: {model.predict([args.message])[0]}")


if __name__ == "__main__":
    main()
