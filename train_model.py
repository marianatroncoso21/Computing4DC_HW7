import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Columns the model uses to predict
FEATURES = [
    "age",
    "height",
    "weight",
    "aids",
    "cirrhosis",
    "hepatic_failure",
    "immunosuppression",
    "leukemia",
    "lymphoma",
    "solid_tumor_with_metastasis",
]

# Column the model tries to predict (1 = has diabetes, 0 = doesn't)
TARGET = "diabetes_mellitus"

# Load the data
df = pd.read_csv("sample_diabetes_mellitus_data.csv")

# Keep only the columns we need and drop rows with missing values
df = df[FEATURES + [TARGET]].dropna()

# Check how many rows and columns we have left
print(df.shape)

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model: it learns the relation between the features and the target
model.fit(df[FEATURES], df[TARGET])

print("Model trained!")

# Save the trained model to a file, so the API can load it later
joblib.dump(model, "model.joblib")

print("Model saved to model.joblib")
