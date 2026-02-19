import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import sys
import time

print("Disabled GPU. Attempting to import tensorflow...")
sys.stdout.flush()

try:
    import tensorflow as tf
    print(f"TensorFlow imported successfully. Version: {tf.__version__}")
except Exception as e:
    print(f"Failed to import tensorflow: {e}")
