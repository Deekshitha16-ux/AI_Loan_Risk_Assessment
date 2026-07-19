import pickle
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parent / 'loan_model3.pkl'
print('Model file exists:', p.exists())
if not p.exists():
    raise SystemExit('Model missing')
with p.open('rb') as f:
    model=pickle.load(f)
print('Model class:', model.__class__.__name__)
FEATURE_ORDER=[
    "Age","AnnualIncome","CreditScore","EducationLevel","Experience","LoanAmount","LoanDuration","NumberOfDependents","MonthlyDebtPayments","CreditCardUtilizationRate","NumberOfOpenCreditLines","NumberOfCreditInquiries","DebtToIncomeRatio","BankruptcyHistory","PreviousLoanDefaults","PaymentHistory","LengthOfCreditHistory","SavingsAccountBalance","CheckingAccountBalance","TotalAssets","TotalLiabilities","UtilityBillsPaymentHistory","JobTenure","EmploymentStatus_Self-Employed","EmploymentStatus_Unemployed","MaritalStatus_Married","MaritalStatus_Single","MaritalStatus_Widowed","HomeOwnershipStatus_Other","HomeOwnershipStatus_Own","HomeOwnershipStatus_Rent","LoanPurpose_Debt Consolidation","LoanPurpose_Education","LoanPurpose_Home","LoanPurpose_Other",
]
X=np.zeros((1,len(FEATURE_ORDER)))
try:
    prob=model.predict_proba(X)
    print('predict_proba shape:', getattr(prob,'shape',None))
    print('predict_proba sample:', prob[0][:5])
    print('prob[0][1]=', prob[0][1])
except Exception as e:
    print('predict_proba failed:', e)
try:
    pred=model.predict(X)
    print('predict:', pred)
except Exception as e:
    print('predict failed:', e)
