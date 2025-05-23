import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from data_preprocessing import load_processed_data

# Define paths for saving models and results
MODEL_PATH = './saved_models/'
RESULTS_PATH = './results/'
# Create directories if they don't exist
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

def train_random_forest(X_train, y_train, optimize=True):
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        optimize: Whether to perform hyperparameter optimization
        
    Returns:
        Trained model
    """
    print("Training Random Forest classifier...")
    start_time = time.time()
    
    if optimize:
        # Define hyperparameter search space for optimization
        param_grid = {
            'n_estimators': [100, 200],        # Number of trees in the forest
            'max_depth': [None, 20, 30],       # Maximum depth of each tree
            'min_samples_split': [2, 5]        # Minimum samples required to split a node
        }
        
        # Initialize Random Forest model
        model = RandomForestClassifier(random_state=42, n_jobs=-1)  # n_jobs=-1 uses all processors
        
        # Use RandomizedSearchCV for efficient hyperparameter tuning
        # This samples random combinations instead of trying all possible combinations
        grid_search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=5, n_jobs=-1, random_state=42)
        grid_search.fit(X_train, y_train)
        
        # Get the best model from hyperparameter search
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        # Use default hyperparameters if optimization is not requested
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
    
    # Calculate and display training time
    end_time = time.time()
    print(f"Random Forest training completed in {end_time - start_time:.2f} seconds")
    
    # Save the trained model to disk
    dump(model, os.path.join(MODEL_PATH, 'random_forest.joblib'))
    
    return model

def train_svm(X_train, y_train, optimize=False):
    """
    Train an SVM classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        optimize: Whether to perform hyperparameter optimization
        
    Returns:
        Trained model
    """
    print("Training SVM classifier...")
    start_time = time.time()
    
    # SVMs are computationally expensive for large datasets
    # Reduce dataset size if it's too large
    if len(X_train) > 10000:
        print("Using subset of data for SVM due to computational constraints")
        # Randomly sample 10,000 instances to make training feasible
        indices = np.random.choice(len(X_train), 10000, replace=False)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
    else:
        X_subset = X_train
        y_subset = y_train
    
    if optimize:
        # Define hyperparameter search space for optimization
        param_grid = {
            'C': [0.1, 1, 10],               # Regularization parameter
            'kernel': ['linear', 'rbf'],     # Kernel type
            'gamma': ['scale', 'auto']       # Kernel coefficient
        }
        
        # Initialize SVM model with probability estimates enabled
        model = SVC(probability=True, random_state=42)
        
        # Use RandomizedSearchCV for efficient hyperparameter tuning
        grid_search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=5, n_jobs=-1, random_state=42)
        grid_search.fit(X_subset, y_subset)
        
        # Get the best model from hyperparameter search
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        # Use default hyperparameters if optimization is not requested
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_subset, y_subset)
    
    # Calculate and display training time
    end_time = time.time()
    print(f"SVM training completed in {end_time - start_time:.2f} seconds")
    
    # Save the trained model to disk
    dump(model, os.path.join(MODEL_PATH, 'svm.joblib'))
    
    return model

def train_neural_network(X_train, y_train, optimize=False):
    """
    Train a Neural Network classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        optimize: Whether to perform hyperparameter optimization
        
    Returns:
        Trained model
    """
    print("Training Neural Network classifier...")
    start_time = time.time()
    
    if optimize:
        # Define hyperparameter search space for optimization
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # Network architecture
            'activation': ['relu', 'tanh'],                   # Activation function
            'alpha': [0.0001, 0.001, 0.01],                   # L2 regularization
            'max_iter': [200, 300]                            # Maximum iterations
        }
        
        # Initialize Neural Network model
        model = MLPClassifier(random_state=42)
        
        # Use RandomizedSearchCV for efficient hyperparameter tuning
        grid_search = RandomizedSearchCV(model, param_grid, cv=3, n_iter=5, n_jobs=-1, random_state=42)
        grid_search.fit(X_train, y_train)
        
        # Get the best model from hyperparameter search
        model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        # Use default hyperparameters if optimization is not requested
        model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=200, random_state=42)
        model.fit(X_train, y_train)
    
    # Calculate and display training time
    end_time = time.time()
    print(f"Neural Network training completed in {end_time - start_time:.2f} seconds")
    
    # Save the trained model to disk
    dump(model, os.path.join(MODEL_PATH, 'neural_network.joblib'))
    
    return model

def train_all_models(X_train, y_train, optimize=False):
    """
    Train all models and return a dictionary of them.
    
    Args:
        X_train: Training features
        y_train: Training labels
        optimize: Whether to perform hyperparameter optimization
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    # Train each model type and add to dictionary
    models['random_forest'] = train_random_forest(X_train, y_train, optimize)
    
    models['svm'] = train_svm(X_train, y_train, optimize)
    
    models['neural_network'] = train_neural_network(X_train, y_train, optimize)
    
    return models

def load_model(model_name='random_forest'):
    """
    Load a saved model.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Loaded model
    """
    try:
        # Attempt to load the specified model from saved files
        model = load(os.path.join(MODEL_PATH, f'{model_name}.joblib'))
        print(f"Model {model_name} loaded successfully.")
        return model
    except Exception as e:
        # Handle any errors during model loading
        print(f"Error loading model {model_name}: {e}")
        return None

# Main execution block
if __name__ == "__main__":
    try:
        # Load preprocessed data split into training and testing sets
        X_train, X_test, y_train, y_test = load_processed_data()
        
        # Train all models with hyperparameter optimization turned off
        # Set optimize=True to enable hyperparameter tuning (will take longer)
        models = train_all_models(X_train, y_train, optimize=False)
        
        print("All models trained successfully!")
        
    except Exception as e:
        # Error handling for any exceptions during execution
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()