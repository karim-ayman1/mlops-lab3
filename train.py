import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()

# 1. Load preprocessed training and testing data
train_df = pd.read_csv('wine_train_preprocessed.csv')
test_df = pd.read_csv('wine_test_preprocessed.csv')

# 2. Split features and target
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

# 3. Train SVM model
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# 4. Predict on test data
preds = model.predict(X_test)

# 5. Calculate accuracy on test data
acc = accuracy_score(y_test, preds)

# 6. Save metrics to JSON
with open('metrics_svm.json', 'w') as f:
    json.dump({'accuracy': acc}, f)

# 7. Generate and save confusion matrix
cm = confusion_matrix(y_test, preds, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Confusion Matrix (Test Data)')
plt.savefig('confusion_matrix_svm.png')

print("SVM model training & testing complete!")
print(f"Test Accuracy: {acc:.4f}")
print("Saved: metrics_svm.json & confusion_matrix_svm.png")
