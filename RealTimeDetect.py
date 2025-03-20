import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from datetime import datetime
import threading
import queue
import random
from data_preprocessing import load_processed_data
from model_training import load_model

# Create real-time logs directory
LOGS_DIR = './logs/'
os.makedirs(LOGS_DIR, exist_ok=True)

class PacketGenerator:
    """Simulates network packets for testing the IDS in real-time."""
    
    def __init__(self, X_test, y_test, speed=1.0):
        """
        Initialize the packet generator.
        
        Args:
            X_test: Test features
            y_test: Test labels
            speed: Speed multiplier for packet generation (higher is faster)
        """
        self.X_test = X_test
        self.y_test = y_test
        self.speed = speed
        self.running = False
        self.packet_queue = queue.Queue(maxsize=100)
        self.log_file = os.path.join(LOGS_DIR, f"packet_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Load feature names if available
        try:
            self.feature_names = pd.read_csv('./processed_data/feature_names.csv')['feature_names'].tolist()
        except:
            self.feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        
        # Initialize log file with header
        with open(self.log_file, 'w') as f:
            f.write("timestamp,packet_id,actual_label\n")
    
    def start(self):
        """Start generating packets."""
        if self.running:
            print("Packet generator is already running")
            return
        
        self.running = True
        self.generator_thread = threading.Thread(target=self._generate_packets)
        self.generator_thread.daemon = True
        self.generator_thread.start()
        print("Packet generator started")
    
    def stop(self):
        """Stop generating packets."""
        self.running = False
        if hasattr(self, 'generator_thread'):
            self.generator_thread.join(timeout=2.0)
        print("Packet generator stopped")
    
    def _generate_packets(self):
        """Generate packets in a background thread."""
        packet_id = 0
        idx = 0
        
        while self.running:
            if idx >= len(self.X_test):
                idx = 0  # Reset to beginning of test set
            
            # Get next packet
            packet_features = self.X_test[idx]
            label = self.y_test[idx]
            
            # Create packet with metadata
            packet = {
                'id': packet_id,
                'features': packet_features,
                'label': label,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            }
            
            # Log packet
            with open(self.log_file, 'a') as f:
                f.write(f"{packet['timestamp']},{packet_id},{label}\n")
            
            # Add to queue (non-blocking)
            try:
                self.packet_queue.put(packet, block=False)
            except queue.Full:
                # Queue is full, skip this packet
                pass
            
            # Increment counters
            packet_id += 1
            idx += 1
            
            # Sleep between packets (randomize to simulate real traffic patterns)
            delay = random.uniform(0.01, 0.1) / self.speed
            time.sleep(delay)
    
    def get_packet(self, timeout=1.0):
        """
        Get the next packet from the queue.
        
        Args:
            timeout: Maximum time to wait for a packet
            
        Returns:
            Packet dict or None if queue is empty
        """
        try:
            return self.packet_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class NetworkIntrusionDetector:
    """Real-time network intrusion detection system."""
    
    def __init__(self, model_name='random_forest', alert_threshold=0.8):
        """
        Initialize the detector.
        
        Args:
            model_name: Name of the model to use
            alert_threshold: Probability threshold for raising alerts
        """
        self.model = load_model(model_name)
        if self.model is None:
            raise ValueError(f"Could not load model: {model_name}")
            
        self.model_name = model_name
        self.alert_threshold = alert_threshold
        self.running = False
        self.detection_queue = queue.Queue()
        self.alerts = []
        
        # Create detection log file
        self.log_file = os.path.join(LOGS_DIR, f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(self.log_file, 'w') as f:
            f.write("timestamp,packet_id,predicted_label,actual_label,probability,is_alert\n")
            
        # Initialize performance metrics
        self.metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'processed_packets': 0,
            'alerts_raised': 0
        }
        
        # For visualization
        self.detection_times = []
        self.detection_results = []
        
    def start(self, packet_generator):
        """
        Start the detection process.
        
        Args:
            packet_generator: PacketGenerator instance
        """
        if self.running:
            print("Detector is already running")
            return
            
        self.running = True
        self.packet_generator = packet_generator
        
        # Start detection thread
        self.detector_thread = threading.Thread(target=self._detection_loop)
        self.detector_thread.daemon = True
        self.detector_thread.start()
        
        # Start visualization thread
        self.visualization_thread = threading.Thread(target=self._visualization_loop)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()
        
        print(f"Network Intrusion Detector started with {self.model_name} model")
        
    def stop(self):
        """Stop the detection process."""
        self.running = False
        if hasattr(self, 'detector_thread'):
            self.detector_thread.join(timeout=2.0)
        if hasattr(self, 'visualization_thread'):
            self.visualization_thread.join(timeout=2.0)
        
        # Print final statistics
        self._print_statistics()
        print("Network Intrusion Detector stopped")
        
    def _detection_loop(self):
        """Main detection loop that processes packets."""
        while self.running:
            # Get next packet
            packet = self.packet_generator.get_packet()
            if packet is None:
                time.sleep(0.01)  # Small pause to prevent CPU spinning
                continue
                
            # Process the packet
            features = packet['features'].reshape(1, -1)
            start_time = time.time()
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            max_prob = max(probabilities)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Determine if this is an alert
            is_alert = (prediction != 0) and (max_prob >= self.alert_threshold)
            if is_alert:
                self.alerts.append(packet)
                self.metrics['alerts_raised'] += 1
                
            # Update metrics
            self.metrics['processed_packets'] += 1
            if prediction == packet['label']:
                if prediction == 0:  # True negative
                    self.metrics['true_negatives'] += 1
                else:  # True positive
                    self.metrics['true_positives'] += 1
            else:
                if prediction == 0:  # False negative
                    self.metrics['false_negatives'] += 1
                else:  # False positive
                    self.metrics['false_positives'] += 1
                    
            # Log detection
            with open(self.log_file, 'a') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]},"
                        f"{packet['id']},{prediction},{packet['label']},{max_prob:.4f},{1 if is_alert else 0}\n")
                
            # Add to visualization data
            self.detection_times.append(time.time())
            self.detection_results.append(prediction)
            
            # Optional: add a small delay to simulate processing time
            # time.sleep(random.uniform(0.005, 0.02))
            
    def _visualization_loop(self):
        """Periodically updates the visualization."""
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.canvas.manager.set_window_title('Real-time Network Intrusion Detection')
        
        # For plotting performance over time
        accuracy_history = []
        alert_history = []
        time_points = []
        start_time = time.time()
        
        while self.running:
            # Only update every second
            time.sleep(1.0)
            
            if not self.detection_results:
                continue
                
            # Clear previous plots
            ax1.clear()
            ax2.clear()
            
            # Calculate current accuracy
            if self.metrics['processed_packets'] > 0:
                current_accuracy = (self.metrics['true_positives'] + self.metrics['true_negatives']) / self.metrics['processed_packets']
                accuracy_history.append(current_accuracy)
                alert_history.append(self.metrics['alerts_raised'] / max(1, self.metrics['processed_packets']))
                time_points.append(time.time() - start_time)
                
                # Plot accuracy and alert rate
                ax1.plot(time_points, accuracy_history, 'b-', label='Accuracy')
                ax1.plot(time_points, alert_history, 'r-', label='Alert Rate')
                ax1.set_title('Detection Performance')
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Rate')
                ax1.grid(True)
                ax1.legend()
                ax1.set_ylim(0, 1)
                
            # Plot last 100 detection results
            recent_results = self.detection_results[-100:]
            unique_labels = np.unique(recent_results)
            
            for label in unique_labels:
                count = recent_results.count(label)
                ax2.bar(str(label), count, label=f'Class {label}' if label > 0 else 'Normal')
                
            ax2.set_title('Recent Traffic Classification')
            ax2.set_xlabel('Traffic Class')
            ax2.set_ylabel('Count')
            ax2.legend()
            
            # Display metrics in the plot
            stats_text = (f"Processed: {self.metrics['processed_packets']}, "
                          f"Alerts: {self.metrics['alerts_raised']}, "
                          f"TPR: {self._calculate_tpr():.2f}, "
                          f"FPR: {self._calculate_fpr():.2f}")
            fig.suptitle(stats_text)
            
            # Refresh the plot
            fig.canvas.draw()
            fig.canvas.flush_events()
            
        plt.ioff()  # Turn off interactive mode when done
        
    def _calculate_tpr(self):
        """Calculate True Positive Rate (Recall)."""
        if self.metrics['true_positives'] + self.metrics['false_negatives'] == 0:
            return 0
        return self.metrics['true_positives'] / (self.metrics['true_positives'] + self.metrics['false_negatives'])
        
    def _calculate_fpr(self):
        """Calculate False Positive Rate."""
        if self.metrics['false_positives'] + self.metrics['true_negatives'] == 0:
            return 0
        return self.metrics['false_positives'] / (self.metrics['false_positives'] + self.metrics['true_negatives'])
        
    def _print_statistics(self):
        """Print detection statistics."""
        processed = self.metrics['processed_packets']
        if processed == 0:
            print("No packets processed")
            return
            
        # Calculate metrics
        accuracy = (self.metrics['true_positives'] + self.metrics['true_negatives']) / processed
        precision = self.metrics['true_positives'] / max(1, (self.metrics['true_positives'] + self.metrics['false_positives']))
        recall = self._calculate_tpr()
        f1 = 2 * (precision * recall) / max(0.001, (precision + recall))
        
        print("\n========== Detection Statistics ==========")
        print(f"Total packets processed: {processed}")
        print(f"Alerts raised: {self.metrics['alerts_raised']} ({self.metrics['alerts_raised']/processed:.2%})")
        print(f"True Positives: {self.metrics['true_positives']}")
        print(f"False Positives: {self.metrics['false_positives']}")
        print(f"True Negatives: {self.metrics['true_negatives']}")
        print(f"False Negatives: {self.metrics['false_negatives']}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("==========================================")
        
        # Save statistics to file
        stats_file = os.path.join(LOGS_DIR, f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(stats_file, 'w') as f:
            f.write("========== Detection Statistics ==========\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Alert Threshold: {self.alert_threshold}\n")
            f.write(f"Total packets processed: {processed}\n")
            f.write(f"Alerts raised: {self.metrics['alerts_raised']} ({self.metrics['alerts_raised']/processed:.2%})\n")
            f.write(f"True Positives: {self.metrics['true_positives']}\n")
            f.write(f"False Positives: {self.metrics['false_positives']}\n")
            f.write(f"True Negatives: {self.metrics['true_negatives']}\n")
            f.write(f"False Negatives: {self.metrics['false_negatives']}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write("==========================================\n")


def start_real_time_detection(model_name='random_forest', speed=1.0, alert_threshold=0.8, duration=60):
    """
    Start real-time intrusion detection with visualization.
    
    Args:
        model_name: Name of the model to use
        speed: Speed multiplier for packet generation
        alert_threshold: Probability threshold for raising alerts
        duration: Duration in seconds to run the simulation
    """
    try:
        # Load test data
        _, X_test, _, y_test = load_processed_data()
        
        # Create packet generator
        packet_gen = PacketGenerator(X_test, y_test, speed=speed)
        
        # Create detector
        detector = NetworkIntrusionDetector(model_name=model_name, alert_threshold=alert_threshold)
        
        # Start the simulation
        packet_gen.start()
        detector.start(packet_gen)
        
        # Run for specified duration
        print(f"Running real-time detection for {duration} seconds...")
        time.sleep(duration)
        
        # Stop the simulation
        detector.stop()
        packet_gen.stop()
        
        print("Real-time detection simulation completed.")
        return detector  # Return for additional analysis if needed
        
    except Exception as e:
        print(f"Error in real-time detection: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    try:
        # Example usage
        start_real_time_detection(
            model_name='random_forest',  # Can be 'random_forest', 'svm', or 'neural_network'
            speed=2.0,                   # Speed multiplier (higher is faster)
            alert_threshold=0.7,         # Probability threshold for alerts
            duration=120                 # Run for 120 seconds
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()