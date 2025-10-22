from sklearn.datasets import load_wine
import pandas as pd

# Load the dataset
data = load_wine()

# Create a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target  # add the target column

# Save to CSV
df.to_csv("wine_dataset.csv", index=False)

print("✅ wine_dataset.csv saved successfully!")
