import pandas as pd
from sklearn.datasets import make_classification
import os

# Define the output directory and file path
output_dir = "data"
output_file = os.path.join(output_dir, "generated_data.csv")

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Generate a synthetic dataset for classification
# n_samples: Total number of samples
# n_features: Total number of features
# n_informative: Number of informative features (features that are actually useful)
# n_redundant: Number of redundant features (linear combinations of informative features)
# n_classes: Number of classes (target labels)
# random_state: Seed for reproducibility
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42,
)

# Create a pandas DataFrame
feature_names = [f"feature_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# Save the DataFrame to a CSV file
df.to_csv(output_file, index=False)

print(f"Successfully generated synthetic data and saved to {output_file}")