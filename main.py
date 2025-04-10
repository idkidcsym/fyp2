import os
import argparse
import time
from data_preprocessing import load_and_preprocess_data, load_processed_data
from model_training import train_all_models, load_model
from model_eval import evaluate_all_models
from realtime_detect import start_real_time_detection

def parse_arguments():
    """
    Parse command line arguments for the Network Intrusion Detection System.
    Defines all the parameters that can be passed to control program execution.
    """
    parser = argparse.ArgumentParser(description='Network Intrusion Detection System')
    
    # Mode selection - determines which parts of the pipeline to run
    parser.add_argument('--mode', type=str, default='train',
                        choices=['preprocess', 'train', 'evaluate', 'realtime', 'all'],
                        help='Operation mode')
    
    # Data preprocessing options
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations during preprocessing')
    parser.add_argument('--sample', action='store_true',
                   help='Use a balanced sample of the dataset for faster training')
    parser.add_argument('--max_per_class', type=int, default=10000,
                   help='Maximum samples per class when using --sample')
    
    # Model training options
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization during training')
    parser.add_argument('--models', type=str, default='all',
                        choices=['random_forest', 'svm', 'neural_network', 'all'],
                        help='Models to train')
    
    # Model selection for evaluation or real-time detection
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
    """
    Main entry point of the application.
    Orchestrates the entire network intrusion detection pipeline based on command line arguments.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create necessary directories if they don't exist
    os.makedirs('./processed_data', exist_ok=True)  # For storing preprocessed data
    os.makedirs('./saved_models', exist_ok=True)    # For storing trained models
    os.makedirs('./results', exist_ok=True)         # For storing evaluation results
    os.makedirs('./logs', exist_ok=True)            # For storing application logs
    
    # Step 1: Data Preprocessing
    if args.mode == 'preprocess' or args.mode == 'all':
        print("\n===== Data Preprocessing =====")
        # Process raw CICIDS2017 data into clean, normalized format
        load_and_preprocess_data(visualize=args.visualize)
    
    # Step 2: Model Training
    if args.mode == 'train' or args.mode == 'all':  
        print("\n===== Model Training =====")
        # Load preprocessed data (or create it if it doesn't exist)
        X_train, X_test, y_train, y_test = load_processed_data(
            sample=args.sample,               # Use balanced sample if requested
            max_per_class=args.max_per_class  # Maximum samples per class
        )
        
        # Train selected models
        if args.models == 'all':
            # Train all available models
            from model_training import train_all_models
            train_all_models(X_train, y_train, optimize=args.optimize)
        else:
            # Train specific model based on user selection
            if args.models == 'random_forest':
                from model_training import train_random_forest
                train_random_forest(X_train, y_train, optimize=args.optimize)
            elif args.models == 'svm':
                from model_training import train_svm
                train_svm(X_train, y_train, optimize=args.optimize)
            elif args.models == 'neural_network':
                from model_training import train_neural_network
                train_neural_network(X_train, y_train, optimize=args.optimize)
    
    # Step 3: Model Evaluation
    if args.mode == 'evaluate' or args.mode == 'all':
        print("\n===== Model Evaluation =====")
        # Evaluate all models and determine the best performer
        b_model = evaluate_all_models()
        print(f"Best model: {b_model}")
    
    # Step 4: Real-time Detection
    if args.mode == 'realtime' or args.mode == 'all':
        print("\n===== Real-time Detection =====")
        # Determine which model to use for real-time detection
        model_name = args.model
        if model_name == 'best':
            try:
                # Automatically select the model with the highest F1 score
                import pandas as pd
                model_comparison = pd.read_csv('./results/model_comparison.csv')
                b_model = model_comparison.loc[model_comparison['f1_macro'].idxmax(), 'model_name']
                model_name = b_model
                print(f"Using best model: {model_name}")
            except:
                # Fall back to random forest if best model can't be determined
                model_name = 'random_forest'
                print(f"Could not determine best model, using {model_name}")
        
        # Start the real-time detection engine with specified parameters
        start_real_time_detection(
            model_name=model_name,             # Model to use for detection
            speed=args.speed,                  # Speed multiplier for simulation
            alert_threshold=args.threshold,    # Detection confidence threshold
            duration=args.duration             # How long to run the detection
        )
    
    print("\nAll operations completed successfully.")

if __name__ == "__main__":
    try:
        # Track and report total execution time
        start_time = time.time()
        main()
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        # Handle and display any exceptions
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()