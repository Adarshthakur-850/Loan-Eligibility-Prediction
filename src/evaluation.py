from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

PLOTS_DIR = "plots"

def evaluate_models(models, X_test, y_test):
    """
    Evaluates models and saves results.
    """
    print("\nEvaluating models...")
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
        
        print(f"\n{name} Results:")
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)
        fig.savefig(os.path.join(PLOTS_DIR, f'confusion_matrix_{name}.png'))
        plt.close(fig)
        
    results_df = pd.DataFrame(results)
    print("\nModel Comparison:")
    print(results_df)
    
    # Save comparison to CSV
    results_df.to_csv(os.path.join(PLOTS_DIR, 'model_comparison.csv'), index=False)
    
    return results_df
