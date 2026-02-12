import sys
import os

# Add src to python path to ensure modules are found
sys.path.append(os.path.join(os.getcwd(), 'src'))

from src.data_loader import load_data
from src.preprocessing import handle_missing_values, encode_categorical, scale_features, split_data
from src.eda import perform_eda
from src.visualization import plot_distributions, plot_categorical_counts, plot_correlation, plot_relationships
from src.feature_engineering import create_features
from src.model_trainer import train_models
from src.evaluation import evaluate_models

def main():
    print("Starting Loan Eligibility Prediction Pipeline...")
    
    # 1. Load Data
    df = load_data()
    
    # 2. EDA (Initial)
    perform_eda(df)
    plot_distributions(df)
    plot_categorical_counts(df)
    
    # 3. Feature Engineering
    df = create_features(df)
    
    # 4. Preprocessing
    df = handle_missing_values(df)
    
    # Visualization after handling missing values
    plot_correlation(df)
    plot_relationships(df)
    
    df = encode_categorical(df)
    
    # 5. Split Data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # 6. Scaling
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # 7. Model Training
    models = train_models(X_train_scaled, y_train)
    
    # 8. Evaluation
    evaluate_models(models, X_test_scaled, y_test)
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()
