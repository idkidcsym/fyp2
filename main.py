import os
import argparse
import time
from data_preprocessing import load_and_preprocess_data, load_processed_data
from model_training import train_all_models, load_model
from model_eval import evaluate_all_models
from realtime_detect import start_real_time_detection

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Network Intrusion Detection System')
    
    # Main operation mode
    parser.add_argument('--mode', type=str, default='train',
                        choices=['preprocess', 'train', 'evaluate', 'realtime', 'all'],
                        help='Operation mode')
    
    # Data preprocessing options
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations during preprocessing')
    
    # Training options
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization during training')
    parser.add_argument('--models', type=str, default='all',
                        choices=['random_forest', 'svm', 'neural_network', 'all'],
                        help='Models to train')
    
    # Evaluation options
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'svm', 'neural_network', 'best'],
                        help='Model to evaluate or use in real-time detection')
    
    # Real-time detection options
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Speed multiplier for packet generation')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Alert threshold for real-time detection')
    parser.add_argument('--duration', type=int, default=60,
                        help='Duration in seconds for real-time detection')
    
    return parser.parse_args()

def main():
    """Main entry point of the application."""
    args = parse_arguments()
    
    # Create necessary directories
    os.makedirs('./processed_data', exist_ok=True)
    os.makedirs('./saved_models', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    if args.mode == 'preprocess' or args.mode == 'all':
        print("\n===== Data Preprocessing =====")
        load_and_preprocess_data(visualize=args.visualize)
    
    if args.mode == 'train' or args.mode == 'all':
        print("\n===== Model Training =====")
        # Load data
        X_train, X_test, y_train, y_test = load_processed_data()
        
        # Train selected models
        if args.models == 'all':
            from model_training import train_all_models
            train_all_models(X_train, y_train, optimize=args.optimize)
        else:
            if args.models == 'random_forest':
                from model_training import train_random_forest
                train_random_forest(X_train, y_train, optimize=args.optimize)
            elif args.models == 'svm':
                from model_training import train_svm
                train_svm(X_train, y_train, optimize=args.optimize)
            elif args.models == 'neural_network':
                from model_training import train_neural_network
                train_neural_network(X_train, y_train, optimize=args.optimize)
    
    if args.mode == 'evaluate' or args.mode == 'all':
        print("\n===== Model Evaluation =====")
        best_model = evaluate_all_models()
        print(f"Best model: {best_model}")
    
    if args.mode == 'realtime' or args.mode == 'all':
        print("\n===== Real-time Detection =====")
        # Load the best model if specified
        model_name = args.model
        if model_name == 'best':
            # Try to find the best model from previous evaluation
            try:
                import pandas as pd
                model_comparison = pd.read_csv('./results/model_comparison.csv')
                best_model = model_comparison.loc[model_comparison['f1_macro'].idxmax(), 'model_name']
                model_name = best_model
                print(f"Using best model: {model_name}")
            except:
                model_name = 'random_forest'
                print(f"Could not determine best model, using {model_name}")
        
        # Run real-time detection
        start_real_time_detection(
            model_name=model_name,
            speed=args.speed,
            alert_threshold=args.threshold,
            duration=args.duration
        )
    
    print("\nAll operations completed successfully.")

if __name__ == "__main__":
    try:
        start_time = time.time()
        main()
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()