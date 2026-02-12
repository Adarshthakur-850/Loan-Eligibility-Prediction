import pandas as pd

def perform_eda(df):
    """
    Performs Exploratory Data Analysis.
    Prints basic statistics and distributions.
    """
    print("\n" + "="*40)
    print("Exploratory Data Analysis")
    print("="*40)
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nNumerical Statistics:")
    print(df.describe())
    
    print("\nCategorical Distributions:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\nValue Counts for {col}:")
        print(df[col].value_counts())
    
    print("\nLoan Status Distribution:")
    if 'Loan_Status' in df.columns:
        print(df['Loan_Status'].value_counts(normalize=True))
