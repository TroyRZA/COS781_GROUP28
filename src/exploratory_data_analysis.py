from dotenv import load_dotenv
import pandas as pd
import os

DATA_FILE = None;


load_dotenv()
path = os.getenv("DATASET_PATH")

if not path:
    print("DATASET_PATH NOT FOUND")

print("files available in datset:")
for f in os.listdir(path):
    print(f, "\n")
    if(f):
        DATA_FILE = os.path.join(path, f)

if not DATA_FILE:
    print("COULDNT FIND CSV IN DATASET FOLDER")

print("DATA INSPECTION:")
df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')
print("DF SHAPE:")
print(df.shape)
print(df.head())
print(df.info())
