import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("iris.csv")

# Display first 5 rows
print("FIRST FIVE ROWS")
print(df.head())

# Display last 5 rows
print("\nLAST FIVE ROWS")
print(df.tail())

# Dataset shape
print("\nDATASET SHAPE")
print(df.shape)

# Dataset information
print("\nDATASET INFORMATION")
df.info()

# Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Check duplicate values
print("\nDUPLICATE ROWS")
print(df.duplicated().sum())
# Explore trends and patterns

print("\nFLOWER SPECIES COUNT")
print(df["species"].value_counts())

print("\nAVERAGE MEASUREMENTS BY SPECIES")
print(df.groupby("species").mean(numeric_only=True))

print("\nMAXIMUM VALUES")
print(df.max(numeric_only=True))

print("\nMINIMUM VALUES")
print(df.min(numeric_only=True))
# Statistical Analysis

print("\nSTATISTICAL ANALYSIS")

# Correlation between flower measurements
print("\nCORRELATION MATRIX")
print(df.corr(numeric_only=True))

# Standard deviation
print("\nSTANDARD DEVIATION")
print(df.std(numeric_only=True))

# Median values
print("\nMEDIAN VALUES")
print(df.median(numeric_only=True))
# Detect Outliers using IQR

print("\nOUTLIER ANALYSIS")

for column in df.select_dtypes(include=np.number).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]

    print(column, ":", len(outliers), "outliers")
    # Histogram
plt.figure(figsize=(8, 5))
df["sepal_length"].hist(bins=15)
plt.title("Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.savefig("histogram.png")
plt.show()
# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="sepal_length",
                y="petal_length", hue="species")
plt.title("Sepal Length vs Petal Length")
plt.savefig("scatterplot.png")
plt.show()
# Final EDA Findings

print("\n===== FINAL EDA REPORT =====")

print("1. Total number of records:", len(df))

print("\n2. Flower species distribution:")
print(df["species"].value_counts())

print("\n3. Missing values:", df.isnull().sum().sum())

print("\n4. Duplicate records:", df.duplicated().sum())

print("\n5. Average flower measurements:")
print(df.mean(numeric_only=True))

print("\n6. Correlation between measurements:")
print(df.corr(numeric_only=True))

print("\nEDA ANALYSIS COMPLETED SUCCESSFULLY!")