# preprocess_wine_csv.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_wine_csv(csv_path="wine_dataset.csv"):
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(df.head(), "\n")

    # 2. Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # 3. Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # 5. Combine with target column again
    train_df = X_train_scaled.copy()
    train_df["target"] = y_train.values

    test_df = X_test_scaled.copy()
    test_df["target"] = y_test.values

    # 6. Save processed files
    train_df.to_csv("wine_train_preprocessed.csv", index=False)
    test_df.to_csv("wine_test_preprocessed.csv", index=False)

    print("Preprocessing complete!")
    print("Saved as: wine_train_preprocessed.csv & wine_test_preprocessed.csv")

    return train_df, test_df

if __name__ == "__main__":
    preprocess_wine_csv()
