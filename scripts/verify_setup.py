from dotenv import load_dotenv
import os

load_dotenv()
path = os.getenv("DATASET_PATH")

if not path:
    print("FAILED TO DOWNLOAD DATASET — DATASET_PATH not found or empty.")
else:
    print("Dataset path loaded successfully:")
    print(path)