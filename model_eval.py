import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from data_preprocessing import load_processed_data
from model_training import load_model

# Directory to store evaluation results
RESULTS_PATH = './results/'
os.makedirs(RESULTS_PATH, exist_ok=True)  # Create directory if it doesn't exist

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance and generate visualizations.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for saving results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"Evaluating {model_name}...")
    
    # Generate predictions and probability scores
    y_pred = model.predict(X_test)  # Class predictions
    y_pred_proba = model.predict_proba(X_test)  # Probability scores for each class
    
    # Calculate basic performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')  # Macro-average precision across all classes
    recall_macro = recall_score(y_test, y_pred, average='macro')  # Macro-average recall across all classes
    f1_macro = f1_score(y_test, y_pred, average='macro')  # Macro-average F1 score across all classes
    
    # Generate and save detailed classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(RESULTS_PATH, f'{model_name}_classification_report.csv'))
    
    # Create and save confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(RESULTS_PATH, f'{model_name}_confusion_matrix.png'), bbox_inches='tight')
    
    # Determine number of unique classes
    n_classes = len(np.unique(y_test))
    
    # Generate ROC curve - different approach for binary vs multiclass
    if n_classes == 2:  # Binary classification
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])  # Use probability of positive class
        roc_auc = auc(fpr, tpr)  # Area under the ROC curve
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (random classifier)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(RESULTS_PATH, f'{model_name}_roc_curve.png'))
    else:  # Multi-class classification
        # Convert multi-class labels to binary format (one-vs-rest)
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_test_bin.shape[1]
        
        # Plot ROC curve for each class (limited to first 5 classes to avoid overcrowding)
        plt.figure(figsize=(10, 8))
        
        for i in range(min(n_classes, 5)):  
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (area = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line (random classifier)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (One-vs-Rest) - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(RESULTS_PATH, f'{model_name}_roc_curve_multi.png'))
    
    # Generate precision-recall curve for binary classification
    if n_classes == 2:
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
        average_precision = average_precision_score(y_test, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                 label=f'Precision-Recall curve (AP = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(RESULTS_PATH, f'{model_name}_precision_recall_curve.png'))
    
    # Print performance metrics to console
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Return dictionary of key metrics for model comparison
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

def compare_models(model_metrics):
    """
    Compare different models based on their performance metrics.
    Creates visualizations and identifies the best model.
    
    Args:
        model_metrics: List of dictionaries containing model metrics
        
    Returns:
        Name of the best performing model based on F1 score
    """
    # Create DataFrame from list of metrics dictionaries
    metrics_df = pd.DataFrame(model_metrics)
    metrics_df.set_index('model_name', inplace=True)
    
    # Save metrics to CSV for future reference
    metrics_df.to_csv(os.path.join(RESULTS_PATH, 'model_comparison.csv'))
    
    # Create bar chart comparing model performance
    plt.figure(figsize=(12, 6))
    metrics_df.plot(kind='bar', ax=plt.gca())
    plt.title('Model Performance Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(RESULTS_PATH, 'model_comparison.png'), bbox_inches='tight')
    
    # Print comparison table to console
    print("\nModel Comparison:")
    print(metrics_df)
    
    # Identify best model based on F1 score
    b_model = metrics_df['f1_macro'].idxmax()
    print(f"\nBest model based on F1 score: {b_model}")
    return b_model

def evaluate_all_models():
    """
    Evaluate all trained models using the test dataset.
    
    Returns:
        Name of the best performing model
    """
    # Load the preprocessed data
    X_train, X_test, y_train, y_test = load_processed_data()
    
    # List of models to evaluate
    model_names = ['random_forest', 'svm', 'neural_network']
    
    # Evaluate each model and collect metrics
    metrics_list = []
    for model_name in model_names:
        try:
            # Load trained model from disk
            model = load_model(model_name)
            if model is not None:
                # Evaluate model and append metrics to list
                metrics = evaluate_model(model, X_test, y_test, model_name)
                metrics_list.append(metrics)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Compare models and return the best one
    if metrics_list:
        best_model = compare_models(metrics_list)
        return best_model
    else:
        print("No models could be evaluated.")
        return None

def label_binarize(y, classes):
    """
    Transform multi-class labels to binary labels (one-vs-rest).
    This is a custom implementation of scikit-learn's label_binarize function.
    
    Args:
        y: Multi-class labels
        classes: Unique classes
        
    Returns:
        Binary labels in one-hot encoding format
    """
    n_samples = len(y)
    n_classes = len(classes)
    classes_dict = {c: i for i, c in enumerate(classes)}  # Map each class to an index
    
    # Initialize binary label matrix with zeros
    y_bin = np.zeros((n_samples, n_classes))
    
    # Set 1 at the column corresponding to the class
    for i, label in enumerate(y):
        y_bin[i, classes_dict[label]] = 1
    
    return y_bin

# Main execution block - runs when script is executed directly
if __name__ == "__main__":
    try:
        best_model = evaluate_all_models()
        print(f"Evaluation complete. Best model: {best_model}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()