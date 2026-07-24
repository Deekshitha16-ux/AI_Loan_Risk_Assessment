# AI-Based Loan Risk Assessment and Credit Decision Support System

## 📌 Project Overview

The AI-Based Loan Risk Assessment and Credit Decision Support System is a Machine Learning application that predicts whether a loan application should be approved or rejected based on an applicant's financial, personal, and credit-related information.

The project helps banks and financial institutions make faster, consistent, and data-driven loan approval decisions while reducing the risk of loan defaults.

---

## 🎯 Objectives

- Predict loan approval using Machine Learning.
- Reduce manual loan verification.
- Improve decision-making using historical loan data.
- Compare multiple classification algorithms and select the best-performing model.

---

## 📊 Dataset

- Dataset Name: Loan.csv
- Records: 20,000
- Features: 36
- Target Variable: LoanApproved
- Problem Type: Binary Classification

---

## 🛠 Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Pickle

---

## 🔍 Data Preprocessing

- Checked missing values
- Removed duplicate records
- Converted ApplicationDate into DateTime format
- Extracted Year and Month
- Applied Ordinal Encoding
- Applied One-Hot Encoding
- Selected relevant features
- Train-Test Split (80:20)

---

## 📈 Exploratory Data Analysis (EDA)

Performed:

- Loan Approval Distribution
- Age Distribution
- Loan Amount Distribution
- Annual Income Distribution
- Employment Status Analysis
- Loan Purpose Analysis
- Correlation Heatmap
- Pair Plot

---

## 🤖 Machine Learning Models

The following models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting Classifier

### Final Model

Gradient Boosting Classifier

Reason:
- Highest prediction accuracy
- Better overall classification performance

---

## 📊 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | ~90% |
| Decision Tree | ~87% |
| Random Forest | ~90% |
| Gradient Boosting | **~92%** |

---

## 📋 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🚀 Deployment

The trained Gradient Boosting model was saved using Pickle and deployed using Streamlit.

Users can:

- Enter applicant details
- Predict loan approval
- Receive instant results

---

## 📂 Project Structure

```
Loan-Risk-Assessment/
│
├── Loan.csv
├── ai_loan.ipynb
├── app.py
├── loan_model3.pkl
├── requirements.txt
├── README.md
└── screenshots/
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Loan-Risk-Assessment.git
```

Navigate to the project

```bash
cd Loan-Risk-Assessment
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📷 Application Features

- User-friendly interface
- Real-time prediction
- Fast model inference
- Financial risk assessment
- Machine Learning-based decision support

---

## 🔮 Future Scope

- Collect larger real-world datasets
- Hyperparameter tuning
- Apply SMOTE for handling class imbalance
- Experiment with XGBoost and LightGBM
- Deploy on AWS, Azure, or Google Cloud
- Implement MLOps for continuous monitoring


## LIVE DEMO : https://ailoanriskassessment-q4xr5rnbppxyd2hvehcubi.streamlit.app/
