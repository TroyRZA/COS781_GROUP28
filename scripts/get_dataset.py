import os
import kagglehub

path = kagglehub.dataset_download("chiranjivdas09/ta-feng-grocery-dataset")
print("Dataset path:", path)

with open(".env", "w") as f:
    f.write(f"DATASET_PATH={path}\n")