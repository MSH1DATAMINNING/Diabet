from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# ارزیابی مدل Decision Tree
y_pred_tree = decision_tree.predict(X_test)

# ارزیابی مدل SVM
y_pred_svm = svm_model.predict(X_test)

# محاسبه معیارهای ارزیابی برای Decision Tree
tree_accuracy = accuracy_score(y_test, y_pred_tree)
tree_precision = precision_score(y_test, y_pred_tree)
tree_recall = recall_score(y_test, y_pred_tree)
tree_f1 = f1_score(y_test, y_pred_tree)
tree_conf_matrix = confusion_matrix(y_test, y_pred_tree)

# محاسبه معیارهای ارزیابی برای SVM
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)
svm_conf_matrix = confusion_matrix(y_test, y_pred_svm)

# نمایش ماتریس گیج‌زنی برای هر دو مدل
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(tree_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(svm_conf_matrix, annot=True, fmt="d", cmap="Reds", xticklabels=["No Diabetes", "Diabetes"], yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# نمایش نتایج
results = {
    "Model": ["Decision Tree", "SVM"],
    "Accuracy": [tree_accuracy, svm_accuracy],
    "Precision": [tree_precision, svm_precision],
    "Recall": [tree_recall, svm_recall],
    "F1-Score": [tree_f1, svm_f1],
}

import pandas as pd
results_df = pd.DataFrame(results)
print(results_df)