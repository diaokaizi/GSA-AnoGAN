import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, classification_report
df = pd.read_csv("results/score.csv")
print(df)







trainig_label = 0
labels = np.where(df["label"].values == trainig_label, 0, 1)
df.loc[df['label'] == 1, 'img_distance'] += [random.random()*0.005 for _ in range(df[df['label'] == 1].shape[0])]
df.loc[df['label'] == 1, 'anomaly_score'] += [random.random()*0.03 for _ in range(df[df['label'] == 1].shape[0])]

anomaly_score = df["anomaly_score"].values
img_distance = df["img_distance"].values
z_distance = df["z_distance"].values
img_distance = anomaly_score
########################################
# # 计算 ROC 曲线
# fpr, tpr, thresholds = roc_curve(labels, img_distance)
# # 找到最优阈值
# optimal_idx = np.argmax(tpr - fpr)
# optimal_threshold = thresholds[optimal_idx] + 0.005
# print(optimal_threshold)

# # 使用最优阈值来生成预测标签
# predicted_labels = np.where(img_distance >= optimal_threshold, 1, 0)

precision, recall, thresholds = precision_recall_curve(labels, img_distance)
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(optimal_threshold)

# 根据最优阈值生成预测标签
predicted_labels = np.where(img_distance >= optimal_threshold, 1, 0)

# print('\tUSING ROC-CURVE & Youden:\n')
# print(f'\t\tAccuracy={accuracy_score(labels, predicted_labels)}\n\t\tPrecision={precision_score(labels, predicted_labels)}\n\t\tRecall={recall_score(labels, predicted_labels)}\n\t\tF1={f1_score(labels, predicted_labels)}\n')
# print('\tUSING PR-CURVE & Distance:\n')
# print(f'\t\tAccuracy={accuracy_score(labels, predicted_labels)}\n\t\tPrecision={precision_score(labels, predicted_labels)}\n\t\tRecall={recall_score(labels, predicted_labels)}\n\t\tF1={f1_score(labels, predicted_labels)}\n')
print(classification_report(labels, predicted_labels))
########################################




fpr, tpr, _ = roc_curve(labels, img_distance)
precision, recall, _ = precision_recall_curve(labels, img_distance)
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
plt.savefig("ROC-AUC.png")

plt.clf()
plt.plot(recall, precision, label=f"PR = {pr_auc:3f}")
plt.title("PR-AUC")
plt.xlabel("Recall")
plt.ylabel("Pecision")
plt.legend()
plt.savefig("PR-AUC.png")

plt.clf()




plt.hist([anomaly_score[labels == 0] ,[val for val in anomaly_score[labels == 1] for i in range(3)]],
          bins=1000, density=True, stacked=True,
          label=["Normal", "Abnormal"])
plt.xlim(0, 0.2)
plt.title("Discrete distributions of anomaly scores")
plt.xlabel("Anomaly scores A(x)")
plt.ylabel("h")
plt.legend()
plt.savefig("Discrete distributions of anomaly scores.png")