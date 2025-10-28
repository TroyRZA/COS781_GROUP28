from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()
path = os.getenv("DATASET_PATH")

if not path:
    raise ValueError("DATASET_PATH NOT FOUND")

print("Files available in dataset:")
DATA_FILE = None
for f in os.listdir(path):
    print(f)
    if f.endswith('.csv'):
        DATA_FILE = os.path.join(path, f)

if not DATA_FILE:
    raise ValueError("No CSV file found in dataset folder")

print("\nDATA INSPECTION:")
pd.set_option('display.float_format', '{:.2f}'.format)
df = pd.read_csv(DATA_FILE, encoding='utf-8-sig')

print("\nDF SHAPE:", df.shape)
print("\nFEATURE TYPES:")
print(df.dtypes)

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nNUMERICAL FEATURES SUMMARY:")
print(df.describe())

print("\nCATEGORICAL FEATURES SUMMARY:")
print(df.describe(include=['object', 'category']))

print("\nMISSING VALUE PERCENTAGES:")
print((df.isnull().mean() * 100).round(2))

#problems here because identifiers are treated as numbers need to record this in report. Will have to do preprocessing etc.
