import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "plots"

def save_plot(fig, filename):
    """
    Saves a plot to the plots directory.
    """
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath)
    print(f"Plot saved to {filepath}")
    plt.close(fig)

def plot_distributions(df):
    """
    Plots distributions of numerical variables.
    """
    print("Generating distribution plots...")
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numerical_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        save_plot(fig, f'dist_{col}.png')

def plot_categorical_counts(df):
    """
    Plots counts of categorical variables.
    """
    print("Generating categorical count plots...")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(y=col, data=df, ax=ax)
        ax.set_title(f'Count of {col}')
        save_plot(fig, f'count_{col}.png')

def plot_correlation(df):
    """
    Plots correlation heatmap.
    """
    print("Generating correlation heatmap...")
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Heatmap')
        save_plot(fig, 'correlation_heatmap.png')

def plot_relationships(df):
    """
    Plots relationships between variables vs Loan Status.
    """
    print("Generating relationship plots...")
    if 'Loan_Status' in df.columns and 'ApplicantIncome' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df, ax=ax)
        ax.set_title('Applicant Income vs Loan Status')
        save_plot(fig, 'income_vs_loan_status.png')
    
    if 'Loan_Status' in df.columns and 'Credit_History' in df.columns:
         fig, ax = plt.subplots(figsize=(8, 6))
         sns.countplot(x='Credit_History', hue='Loan_Status', data=df, ax=ax)
         ax.set_title('Credit History vs Loan Status')
         save_plot(fig, 'credit_history_vs_loan_status.png')
