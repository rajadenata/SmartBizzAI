import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.data_loader.m5_loader import M5DataLoader
from backend.features.engineering import generate_features
import tracemalloc

def test():
    tracemalloc.start()
    
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'SmartBiz-Dataset-v1.0'))
    print(f"Using dataset path: {data_path}")
    
    try:
        loader = M5DataLoader(data_path)
        data = loader.load_and_aggregate()
        
        print("\n--- Data Head ---")
        print(data.head())
        
        data_feats = generate_features(data)
        
        print("\n--- Feats Head ---")
        print(data_feats.head())
        
        current, peak = tracemalloc.get_traced_memory()
        print(f"\nMemory usage: Current = {current / 10**6} MB; Peak = {peak / 10**6} MB")
        tracemalloc.stop()
        
    except FileNotFoundError as e:
        print(f"Skipping test due to missing dataset: {e}")

if __name__ == "__main__":
    test()
