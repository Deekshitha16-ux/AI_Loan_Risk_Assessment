import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "Loan.csv"
MODEL_PATH = Path(__file__).resolve().parent / "loan_model3.pkl"

FEATURE_ORDER = [
    "Age",
    "AnnualIncome",
    "CreditScore",
    "EducationLevel",
    "Experience",
    "LoanAmount",
    "LoanDuration",
    "NumberOfDependents",
    "MonthlyDebtPayments",
    "CreditCardUtilizationRate",
    "NumberOfOpenCreditLines",
    "NumberOfCreditInquiries",
    "DebtToIncomeRatio",
    "BankruptcyHistory",
    "PreviousLoanDefaults",
    "PaymentHistory",
    "LengthOfCreditHistory",
    "SavingsAccountBalance",
    "CheckingAccountBalance",
    "TotalAssets",
    "TotalLiabilities",
    "UtilityBillsPaymentHistory",
    "JobTenure",
    "EmploymentStatus_Self-Employed",
    "EmploymentStatus_Unemployed",
    "MaritalStatus_Married",
    "MaritalStatus_Single",
    "MaritalStatus_Widowed",
    "HomeOwnershipStatus_Other",
    "HomeOwnershipStatus_Own",
    "HomeOwnershipStatus_Rent",
    "LoanPurpose_Debt Consolidation",
    "LoanPurpose_Education",
    "LoanPurpose_Home",
    "LoanPurpose_Other",
]
RAW_FEATURES = [
    "Age",
    "AnnualIncome",
    "CreditScore",
    "EducationLevel",
    "Experience",
    "LoanAmount",
    "LoanDuration",
    "NumberOfDependents",
    "MonthlyDebtPayments",
    "CreditCardUtilizationRate",
    "NumberOfOpenCreditLines",
    "NumberOfCreditInquiries",
    "DebtToIncomeRatio",
    "BankruptcyHistory",
    "PreviousLoanDefaults",
    "PaymentHistory",
    "LengthOfCreditHistory",
    "SavingsAccountBalance",
    "CheckingAccountBalance",
    "TotalAssets",
    "TotalLiabilities",
    "UtilityBillsPaymentHistory",
    "JobTenure",
    "EmploymentStatus",
    "MaritalStatus",
    "HomeOwnershipStatus",
    "LoanPurpose",
]
CATEGORICAL_COLUMNS = ["EmploymentStatus", "MaritalStatus", "HomeOwnershipStatus", "LoanPurpose"]
EDUCATION_MAP = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}


def transform_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for _, row in df.iterrows():
        input_data = {col: row[col] for col in RAW_FEATURES}
        input_data["EducationLevel"] = EDUCATION_MAP[input_data["EducationLevel"]]
        raw_input = pd.DataFrame([input_data], columns=RAW_FEATURES)
        encoded = pd.get_dummies(raw_input, columns=CATEGORICAL_COLUMNS, drop_first=True).astype(float)
        encoded = encoded.reindex(columns=FEATURE_ORDER, fill_value=0.0)
        rows.append(encoded.iloc[0])

    X = pd.DataFrame(rows)
    y = df["LoanApproved"].astype(int)
    return X, y


def evaluate_model(model_name: str, model) -> tuple[float, int, float]:
    df = pd.read_csv(DATA_PATH)
    X, y = transform_dataframe(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    sample_row = df[df["ApplicationDate"] == "2018-01-30"].iloc[0]
    sample_input = {col: sample_row[col] for col in RAW_FEATURES}
    sample_input["EducationLevel"] = EDUCATION_MAP[sample_input["EducationLevel"]]
    sample_raw = pd.DataFrame([sample_input], columns=RAW_FEATURES)
    sample_encoded = pd.get_dummies(sample_raw, columns=CATEGORICAL_COLUMNS, drop_first=True).astype(float)
    sample_encoded = sample_encoded.reindex(columns=FEATURE_ORDER, fill_value=0.0)
    prob = float(model.predict_proba(sample_encoded)[0][1])
    pred = int(model.predict(sample_encoded)[0])
    return acc, pred, prob


models = {
    "gradient_boosting": GradientBoostingClassifier(random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "logistic_regression": LogisticRegression(max_iter=5000, random_state=42),
}


def main() -> None:
    results = []
    for name, model in models.items():
        acc, pred, prob = evaluate_model(name, model)
        print(name, "acc=", round(acc, 4), "sample_pred=", pred, "sample_prob=", round(prob, 4))
        results.append((name, acc, model))

    best_name, best_acc, best_model = max(results, key=lambda item: item[1])
    df = pd.read_csv(DATA_PATH)
    X, y = transform_dataframe(df)
    best_model.fit(X, y)
    with MODEL_PATH.open("wb") as fh:
        pickle.dump(best_model, fh)

    print("Selected model:", best_name)
    print("Best accuracy:", round(best_acc, 4))
    print("Saved model to", MODEL_PATH)


if __name__ == "__main__":
    main()
