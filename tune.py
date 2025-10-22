import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. Load your preprocessed dataset
df_train = pd.read_csv("wine_train_preprocessed.csv")
df_test = pd.read_csv("wine_test_preprocessed.csv")

X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]
X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]

# 2. Define hyperparameter grid to test
param_grid = {
    "kernel": ["linear", "rbf", "poly"],
    "C": [0.1, 1, 10]
}

# 3. Start a main MLflow parent run
with mlflow.start_run(run_name="SVM_Tuning") as parent_run:
    for kernel in param_grid["kernel"]:
        for c in param_grid["C"]:
            with mlflow.start_run(run_name=f"SVM_kernel={kernel}_C={c}", nested=True):
                # 4. Train SVM with current parameters
                model = SVC(kernel=kernel, C=c, random_state=42)
                model.fit(X_train, y_train)

                # 5. Evaluate on test set
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                # 6. Log to MLflow
                mlflow.log_param("kernel", kernel)
                mlflow.log_param("C", c)
                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(model, "model")

                print(f"✅ SVM(kernel={kernel}, C={c}) -> accuracy={acc:.4f}")

print("🎯 All SVM tuning runs completed and logged to MLflow!")
