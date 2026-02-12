import pandas as pd
import numpy as np

def create_features(df):
    """
    Creates new features for the dataset.
    """
    print("Feature Engineering: Creating new features...")
    
    # Total Income
    if 'ApplicantIncome' in df.columns and 'CoapplicantIncome' in df.columns:
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
    # Income to Loan Amount Ratio
    if 'TotalIncome' in df.columns and 'LoanAmount' in df.columns:
        # Avoid division by zero
        df['IncomeLoanRatio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
        
    # Loan Amount to Term Ratio
    if 'LoanAmount' in df.columns and 'Loan_Amount_Term' in df.columns:
         df['LoanTermRatio'] = df['LoanAmount'] / (df['Loan_Amount_Term'] + 1)
         
    # Log Transformation for skewed features (handling 0/negative values)
    if 'LoanAmount' in df.columns:
        df['LoanAmount_Log'] = np.log(df['LoanAmount'] + 1)
        
    if 'TotalIncome' in df.columns:
        df['TotalIncome_Log'] = np.log(df['TotalIncome'] + 1)
        
    print("Feature Engineering completed.")
    return df
