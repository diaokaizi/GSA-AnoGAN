from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV


# 假设 X 是你的特征矩阵，y 是你的标签
raw_x_train = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(raw_x_train)
y_test = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
y_test = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)


# # 使用 LassoCV 自动选择最佳 alpha 参数
# alphas=np.logspace(-7, 4, 100)
# lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=1000000).fit(X_scaled, y_test)
# best_alpha = lasso_cv.alpha_
# print("Best alpha:", best_alpha)

# # 获取均方误差路径
# mse_path = lasso_cv.mse_path_

# coefs = lasso_cv.path(X_scaled,y_test,alphas = alphas, max_iter = 1000000)[1].T
# plt.figure()
# plt.semilogx(lasso_cv.alphas_,coefs,'-')
# plt.axvline(lasso_cv.alpha_,color = 'black',ls = '--')
# plt.xlabel('Lamda')
# plt.ylabel('Coefficient')
# plt.savefig("lasso_cv.png")

# 使用最佳 alpha 训练 Lasso 模型
# alpha = 0.002
# for i in range(20):
#     alpha += 0.0001
#     print(alpha)
#     lasso = Lasso(alpha=0.023)
#     lasso.fit(X_scaled, y_test)

#     # 获取特征系数
#     coefs = lasso.coef_

#     # 选择非零系数的特征
#     nonzero_features = np.where(coefs != 0)[0]
#     print("Selected features:", nonzero_features)

#     # 打印选中的特征数量
#     print("Number of selected features:", len(nonzero_features))

lasso = Lasso(alpha=0.0023)
lasso.fit(X_scaled, y_test)

# 获取特征系数
coefs = lasso.coef_

# 选择非零系数的特征
nonzero_features = np.where(coefs != 0)[0]
# 获取特征名称
selected_feature_names = [list(raw_x_train.columns)[i] for i in nonzero_features]

# 从原始 DataFrame 中选择特征
df_selected = raw_x_train[selected_feature_names]

print("Selected feature names:", selected_feature_names)
print("Shape of the original DataFrame:", raw_x_train.shape)
print("Shape of the selected DataFrame:", df_selected.shape)

# # 获取重要特征（系数不为零的特征）
# selected_features = [i for i in range(len(lasso.coef_)) if lasso.coef_[i] != 0]
# X_selected = X_scaled[:, selected_features]  # 仅保留重要特征