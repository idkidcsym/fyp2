import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

# Specify the directory containing the CSV files
DATA_DIRECTORY = './CICIDS2017'
PROCESSED_DATA_DIR = './processed_data'
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_and_preprocess_data(visualize=True):
    """
    Load and preprocess the CICIDS2017 dataset.
    
    Args:
        visualize: Boolean flag to generate and save visualizations
        
    Returns:
        Preprocessed training and testing data
    """
    print("Loading and preprocessing data...")
    
    # Load data
    data_frames = []
    for file in os.listdir(DATA_DIRECTORY):
        if file.endswith(".csv"):
            print(f"Processing {file}...")
            file_path = os.path.join(DATA_DIRECTORY, file)
            try:
                temp_df = pd.read_csv(file_path, low_memory=False)
                temp_df.columns = temp_df.columns.str.strip()
                
                # Remove infinity values
                temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Handle missing values
                numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
                temp_df[numeric_cols] = temp_df[numeric_cols].fillna(temp_df[numeric_cols].median())
                
                # Forward fill for categorical columns
                temp_df.ffill(inplace=True)
                
                data_frames.append(temp_df)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    if not data_frames:
        raise ValueError("No valid data files found in the specified directory.")
    
    # Combine all data
    print("Combining datasets...")
    full_data = pd.concat(data_frames, ignore_index=True)
    
    # Rename the label column if it exists with a different name
    if 'Label' not in full_data.columns and ' Label' in full_data.columns:
        full_data.rename(columns={' Label': 'Label'}, inplace=True)
    
    # Convert attack labels to numeric values
    le = LabelEncoder()
    if full_data['Label'].dtype == 'object':
        print("Encoding labels...")
        full_data['Label'] = le.fit_transform(full_data['Label'])
        
        # Save encoder mapping for later reference
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        pd.DataFrame(list(label_mapping.items()), columns=['Attack', 'Code']).to_csv(
            os.path.join(PROCESSED_DATA_DIR, 'label_mapping.csv'), index=False)
    
    # Display attack distribution
    if visualize:
        print("Generating visualizations...")
        plt.figure(figsize=(12, 6))
        attack_counts = full_data['Label'].value_counts()
        sns.barplot(x=attack_counts.index, y=attack_counts.values)
        plt.title('Attack Distribution')
        plt.xticks(rotation=90)
        plt.savefig(os.path.join(PROCESSED_DATA_DIR, 'attack_distribution.png'), bbox_inches='tight')
        
    # Split features and target
    X = full_data.drop('Label', axis=1)
    y = full_data['Label']
    
    # Handle non-numeric columns
    print("Handling non-numeric data...")
    X = X.select_dtypes(include=[np.number])
    
    # Check for and remove columns with constant values
    constant_columns = [col for col in X.columns if X[col].nunique() <= 1]
    if constant_columns:
        print(f"Removing {len(constant_columns)} constant columns")
        X = X.drop(columns=constant_columns)
    
    # Fill remaining NaN values with the median of each column
    X.fillna(X.median(), inplace=True)
    
    # Check for any remaining issues
    if X.isnull().any().any():
        print("Warning: Dataset still contains NaN values. Using forward fill.")
        X.fillna(method='ffill', inplace=True)
        X.fillna(0, inplace=True)  # Replace any remaining NaNs with 0
    
    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler for future use
    from joblib import dump
    dump(scaler, os.path.join(PROCESSED_DATA_DIR, 'scaler.joblib'))
    
    # Feature importance analysis
    if visualize:
        from sklearn.ensemble import RandomForestClassifier
        
        # Use a small subset for feature importance
        sample_size = min(10000, len(X_scaled))
        X_sample = X_scaled[:sample_size]
        y_sample = y[:sample_size]
        
        print("Calculating feature importance...")
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_sample, y_sample)
        
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
        plt.title('Top 15 Features by Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(PROCESSED_DATA_DIR, 'feature_importance.png'))
    
    # Split data into training and testing sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y if len(y.unique()) < 10 else None
    )
    
    # Check class imbalance
    print("Class distribution in training set:")
    train_class_dist = Counter(y_train)
    for label, count in train_class_dist.items():
        print(f"Class {label}: {count} samples")
    
    if len(set(y_train)) > 1:  # Only apply if there's more than one class
        min_samples = min(train_class_dist.values())
        if min_samples < 100:
            print("Applying SMOTE for handling class imbalance...")
            try:
                # Get classes with fewer than 10,000 samples
                minority_classes = {cls: min(10000, count*10) for cls, count in train_class_dist.items() if count < 10000}
                
                # If no minority classes or all classes have >10,000 samples, don't apply SMOTE
                if minority_classes:
                    # Create a sampling strategy dictionary
                    sampling_strategy = minority_classes
                    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                    print("After SMOTE:", Counter(y_train))
                else:
                    print("No minority classes require SMOTE balancing")
            except Exception as e:
                print(f"SMOTE failed: {e}. Continuing without it.")
    
    # Save processed data
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    # Save feature names
    pd.DataFrame({'feature_names': X.columns}).to_csv(
        os.path.join(PROCESSED_DATA_DIR, 'feature_names.csv'), index=False
    )
    
    print("Data preprocessing complete and saved.")
    return X_train, X_test, y_train, y_test

def create_balanced_sample(X, y, max_per_class=10000):
    """
    Create a balanced sample dataset with max_per_class samples per class.
    
    Args:
        X: Features
        y: Labels
        max_per_class: Maximum number of samples per class
        
    Returns:
        X_sample, y_sample: Balanced sample dataset
    """
    unique_classes = np.unique(y)
    X_sample_list = []
    y_sample_list = []
    
    for cls in unique_classes:
        # Get indices for this class
        indices = np.where(y == cls)[0]
        
        # If we have more samples than max_per_class, randomly sample
        if len(indices) > max_per_class:
            indices = np.random.choice(indices, max_per_class, replace=False)
            
        # Add to our sample lists
        X_sample_list.append(X[indices])
        y_sample_list.append(y[indices])
    
    # Combine samples from all classes
    X_sample = np.vstack(X_sample_list)
    y_sample = np.concatenate(y_sample_list)
    
    # Shuffle the samples
    shuffle_idx = np.random.permutation(len(y_sample))
    X_sample = X_sample[shuffle_idx]
    y_sample = y_sample[shuffle_idx]
    
    print(f"Created balanced sample with {len(y_sample)} samples")
    print("Class distribution:", Counter(y_sample))
    
    return X_sample, y_sample

def load_processed_data(sample=False, max_per_class=10000):
    """Load preprocessed data if it exists."""
    try:
        X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
        X_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_test.npy'))
        y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
        y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
        print("Loaded preprocessed data from files.")
        
        # Option to create a balanced sample
        if sample:
            print("Creating balanced sample for training...")
            X_train, y_train = create_balanced_sample(X_train, y_train, max_per_class)
            
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("Preprocessed data not found. Running preprocessing...")
        return load_and_preprocess_data()

if __name__ == "__main__":
    try:
        X_train, X_test, y_train, y_test = load_processed_data()
        print("Data shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
    except Exception as e:
        print(f"An error occurred: {e}")