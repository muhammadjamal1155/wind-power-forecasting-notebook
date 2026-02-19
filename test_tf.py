import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # Enable all logs
import sys
import time

print("Attempting to import tensorflow...")
sys.stdout.flush()

try:
    import tensorflow as tf
    print(f"TensorFlow imported successfully. Version: {tf.__version__}")
except Exception as e:
    print(f"Failed to import tensorflow: {e}")
