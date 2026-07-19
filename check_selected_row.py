import pandas as pd
import app

row = pd.read_csv('Loan.csv')
row = row[row['ApplicationDate'] == '2018-01-30'].iloc[0]

input_data = {
    'Age': int(row['Age']),
    'AnnualIncome': int(row['AnnualIncome']),
    'CreditScore': int(row['CreditScore']),
    'EducationLevel': {'High School':0,'Associate':1,'Bachelor':2,'Master':3,'Doctorate':4}[row['EducationLevel']],
    'Experience': int(row['Experience']),
    'LoanAmount': int(row['LoanAmount']),
    'LoanDuration': int(row['LoanDuration']),
    'NumberOfDependents': int(row['NumberOfDependents']),
    'MonthlyDebtPayments': int(row['MonthlyDebtPayments']),
    'CreditCardUtilizationRate': float(row['CreditCardUtilizationRate']),
    'NumberOfOpenCreditLines': int(row['NumberOfOpenCreditLines']),
    'NumberOfCreditInquiries': int(row['NumberOfCreditInquiries']),
    'DebtToIncomeRatio': float(row['DebtToIncomeRatio']),
    'BankruptcyHistory': int(row['BankruptcyHistory']),
    'PreviousLoanDefaults': int(row['PreviousLoanDefaults']),
    'PaymentHistory': int(row['PaymentHistory']),
    'LengthOfCreditHistory': int(row['LengthOfCreditHistory']),
    'SavingsAccountBalance': int(row['SavingsAccountBalance']),
    'CheckingAccountBalance': int(row['CheckingAccountBalance']),
    'TotalAssets': int(row['TotalAssets']),
    'TotalLiabilities': int(row['TotalLiabilities']),
    'UtilityBillsPaymentHistory': float(row['UtilityBillsPaymentHistory']),
    'JobTenure': int(row['JobTenure']),
    'EmploymentStatus': row['EmploymentStatus'],
    'MaritalStatus': row['MaritalStatus'],
    'HomeOwnershipStatus': row['HomeOwnershipStatus'],
    'LoanPurpose': row['LoanPurpose'],
}
raw_input = pd.DataFrame([input_data], columns=app.RAW_FEATURES)
encoded = pd.get_dummies(raw_input, columns=app.CATEGORICAL_COLUMNS, drop_first=True).astype(float)
encoded = encoded.reindex(columns=app.FEATURE_ORDER, fill_value=0.0)
reference_features, reference_labels = app.load_reference_data()
matched_rows = reference_features.eq(encoded.iloc[0]).all(axis=1)
print('row_label', int(row['LoanApproved']))
print('matched_rows_any', bool(matched_rows.any()))
if matched_rows.any():
    print('matched_label', int(reference_labels[matched_rows].iloc[0]))
