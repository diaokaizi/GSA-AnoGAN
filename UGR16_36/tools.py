import torch
from torchvision import datasets
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]

        if self.transform:
            data = self.transform(data)

        return data, labels

    def __len__(self):
        return len(self.data)


def load_NF_UNSW_NB15():

    data = pd.read_parquet("/root/work/f-AnoGAN-master/your_own_dataset/dataset/NF-UNSW-NB15.parquet")
    data = data.drop(columns=["Attack", "L7_PROTO"], axis=1)
    train = data.iloc[:int(len(data)*0.8),:]
    test = data.iloc[int(len(data)*0.8):,:]
    test = pd.concat([data.iloc[int(len(data)*0.8):,:],data[data.Label==1]])

    _x_train = train[train.Label==0]
    x_train = torch.from_numpy(_x_train.drop(columns=["Label"], axis=1).values).float().view(-1, 3, 3)
    y_train = torch.from_numpy(_x_train["Label"].values)
    
    x_test = torch.from_numpy(test.drop(columns=["Label"], axis=1).values).float().view(-1, 3, 3)
    y_test = torch.from_numpy(test["Label"].values)
    return (x_train, y_train), (x_test, y_test)
# load_NF_UNSW_NB15()
# data = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Xtest.csv")
# print(data.head)
# print(data.shape)
# print(data.columns)

# data = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Xtrain.csv")
# print(data.head)
# print(data.shape)
# print(data.columns)



def lasso_selected_feature_names():
    raw_x_train = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(raw_x_train)
    y_test = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    y_test = y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1)
    lasso = Lasso(alpha=0.0023)
    lasso.fit(X_scaled, y_test)
    # 获取特征系数
    coefs = lasso.coef_
    # 选择非零系数的特征
    nonzero_features = np.where(coefs != 0)[0]
    # 获取特征名称
    selected_feature_names = [list(raw_x_train.columns)[i] for i in nonzero_features]
    # 从原始 DataFrame 中选择特征
    return selected_feature_names

def fix_name():
    return ["sportsocks", "sportprivate", "dportirc", "sporttelnet", "sportrapservice", "dporthttp",
            "sportsyslog", "sportreserved", "dportkpasswd", "tcpflagsACK", "npacketsmedium",
            "sportcups", "dporttelnet", "sportldaps", "tcpflagsPSH", "dportoracle"]

def load_UGR16():
    selected_feature_names = fix_name()
    raw_x_train = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)
    x_train = raw_x_train[selected_feature_names]
    x_train = torch.from_numpy(x_train.values).float().view(-1, 4, 4)
    y_train = torch.zeros(len(x_train))


    raw_x_test = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = raw_x_test[selected_feature_names]
    x_test = torch.from_numpy(x_test.values).float().view(-1, 4, 4)
    y_test = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    print(x_train)
    print(x_train.shape)
    print(y_train)
    print(x_test)
    print(x_test.shape)
    print(y_test)
    return (x_train, y_train), (x_test, y_test)
load_UGR16()
# print(data["labeldos"].value_counts())
# print(data["labelscan11"].value_counts())
# print(data["labelscan44"].value_counts())
# print(data["labelnerisbotnet"].value_counts())
# print(data["labelblacklist"].value_counts())
# print(data["labelanomalyidpscan"].value_counts())
# print(data["labelanomalysshscan"].value_counts())
# print(data["labelanomalyspam"].value_counts())
