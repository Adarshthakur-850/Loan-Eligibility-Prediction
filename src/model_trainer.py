from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib
import os

MODELS_DIR = "models"

def train_models(X_train, y_train):
    """
    Trains multiple models and saves them.
    Returns a dictionary of trained models.
    """
    print("Training models...")
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Save model
        joblib.dump(model, os.path.join(MODELS_DIR, f'{name}.pkl'))
        print(f"{name} saved.")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{name} CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
    return trained_models
