import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def handle_missing_values(df):
    """
    Handles missing values in the dataframe.
    - Categorical: Mode
    - Numerical: Median
    """
    print("Handling missing values...")
    categorical = df.select_dtypes(include=['object']).columns
    numerical = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in categorical:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    for col in numerical:
        df[col].fillna(df[col].median(), inplace=True)
        
    return df

def encode_categorical(df):
    """
    Encodes categorical variables using Label Encoding.
    Retains mapping for interpreting results.
    """
    print("Encoding categorical variables...")
    le = LabelEncoder()
    object_cols = df.select_dtypes(include=['object']).columns
    
    for col in object_cols:
        df[col] = le.fit_transform(df[col])
        
    return df

def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler.
    """
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for future use
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    
    return X_train_scaled, X_test_scaled

def split_data(df, target_col='Loan_Status', test_size=0.2, random_state=42):
    """
    Splits data into train and test sets.
    """
    print("Splitting data...")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode target if it's not numeric yet (it should be handled in encode_categorical but good to be safe)
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
