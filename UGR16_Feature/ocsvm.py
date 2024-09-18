import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, classification_report
import joblib
# 假设你的数据集已经被加载为dataframe
# 训练集为train_data（仅包含正常数据），测试集为test_data，测试集有标签test_labels

# 提取特征
X_train = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)  # 训练集不包含标签，全部为正常数据

X_test = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
y_test = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
y_test = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)


# 初始化OCSVM模型
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.02)  # nu参数控制异常的比例

# 使用正常数据训练OCSVM模型
ocsvm.fit(X_train)
joblib.dump(ocsvm, 'ocsvm_model.pkl')

# 在测试集上进行预测
y_pred = ocsvm.predict(X_test)
# 将 OCSVM 的输出值转换为符合标签的格式
# OCSVM返回的值：+1 表示正常数据，-1 表示异常数据
# 我们将 +1 转换为 0（正常），-1 转换为 1（异常）
y_pred = [0 if pred == 1 else 1 for pred in y_pred]

# 输出分类报告
print("Classification Report:")
print(classification_report(y_test, y_pred))
# 输出混淆矩阵
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

fpr, tpr, _ = roc_curve(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
pr_auc =  auc(recall, precision)

print(roc_auc)
print(pr_auc)
plt.clf()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC-AUC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("ROC-AUC-ocsvm.png")

plt.clf()
plt.plot(recall, precision, label=f"PR = {pr_auc:3f}")
plt.title("PR-AUC")
plt.xlabel("Recall")
plt.ylabel("Pecision")
plt.legend()
plt.savefig("PR-AUC-ocsvm.png")
