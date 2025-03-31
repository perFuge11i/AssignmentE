import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
import copy

# Load the dataset as a dataframe
data = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "nair26/predictive-maintenance-of-machines",
    "CIA-1 Dataset - Dataset (1).csv",
)

learningData = data.copy()

for col in data.columns:
    if data[col].dtype == 'object':
        print(data[col])

# Display the first 5 rows of the dataset
print(data.head())

# Display summary information including data types and non-null counts
print(data.info())

# Get basic descriptive statistics of numerical features
print(data.describe())
