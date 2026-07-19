from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import train_model

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "Loan.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X, y = train_model.transform_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for name, model in [
        ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
        ("random_forest", RandomForestClassifier(n_estimators=200, random_state=42)),
        ("logistic_regression", LogisticRegression(max_iter=5000, random_state=42)),
    ]:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(name, "test_acc", round(acc, 4))


if __name__ == "__main__":
    main()
