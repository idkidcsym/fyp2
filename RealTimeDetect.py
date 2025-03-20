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
    """Real-