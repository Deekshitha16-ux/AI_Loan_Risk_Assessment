import pandas as pd


df = pd.read_csv('Loan.csv')
row = df[df['ApplicationDate'] == '2018-01-30'].iloc[0]

raw_features = [
    'Age','AnnualIncome','CreditScore','EducationLevel','Experience','LoanAmount','LoanDuration','NumberOfDependents',
    'MonthlyDebtPayments','CreditCardUtilizationRate','NumberOfOpenCreditLines','NumberOfCreditInquiries','DebtToIncomeRatio',
    'BankruptcyHistory','PreviousLoanDefaults','PaymentHistory','LengthOfCreditHistory','SavingsAccountBalance','CheckingAccountBalance',
    'TotalAssets','TotalLiabilities','UtilityBillsPaymentHistory','JobTenure','EmploymentStatus','MaritalStatus','HomeOwnershipStatus','LoanPurpose'
]
feature_order = [
    'Age','AnnualIncome','CreditScore','EducationLevel','Experience','LoanAmount','LoanDuration','NumberOfDependents','MonthlyDebtPayments','CreditCardUtilizationRate','NumberOfOpenCreditLines','NumberOfCreditInquiries','DebtToIncomeRatio','BankruptcyHistory','PreviousLoanDefaults','PaymentHistory','LengthOfCreditHistory','SavingsAccountBalance','CheckingAccountBalance','TotalAssets','TotalLiabilities','UtilityBillsPaymentHistory','JobTenure','EmploymentStatus_Self-Employed','EmploymentStatus_Unemployed','MaritalStatus_Married','MaritalStatus_Single','MaritalStatus_Widowed','HomeOwnershipStatus_Other','HomeOwnershipStatus_Own','HomeOwnershipStatus_Rent','LoanPurpose_Debt Consolidation','LoanPurpose_Education','LoanPurpose_Home','LoanPurpose_Other'
]
categorical = ['EmploymentStatus','MaritalStatus','HomeOwnershipStatus','LoanPurpose']
edu_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}

input_data = {col: row[col] for col in raw_features}
input_data['EducationLevel'] = edu_map[input_data['EducationLevel']]
raw_input = pd.DataFrame([input_data], columns=raw_features)
encoded = pd.get_dummies(raw_input, columns=categorical, drop_first=True).astype(float).reindex(columns=feature_order, fill_value=0.0)

rows = []
labels = []
for _, r in df.iterrows():
    d = {col: r[col] for col in raw_features}
    d['EducationLevel'] = edu_map[d['EducationLevel']]
    rr = pd.DataFrame([d], columns=raw_features)
    ee = pd.get_dummies(rr, columns=categorical, drop_first=True).astype(float).reindex(columns=feature_order, fill_value=0.0)
    rows.append(ee.iloc[0].to_dict())
    labels.append(int(r['LoanApproved']))

reference_features = pd.DataFrame(rows)
matched = reference_features.eq(encoded.iloc[0]).all(axis=1)
print('row_label', int(row['LoanApproved']))
print('matched_any', bool(matched.any()))
print('matched_label', int(labels[matched.idxmax()]) if matched.any() else None)
