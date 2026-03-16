import pandas as pd
from build_features import build_features

df = pd.read_csv('data/raw/heart_failure_clinical_records_dataset.csv')
df = build_features(df)
df.to_csv('data/processed/heart_failure_clinical_records_dataset_processed.csv', index=False)