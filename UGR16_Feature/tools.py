import torch
from torchvision import datasets
import pandas as pd
import numpy as np

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

def load_UGR16():
    raw_x_train = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Xtrain.csv").drop(columns=["Row"], axis=1)

    x_train = pd.DataFrame(np.zeros((raw_x_train.shape[0], 144)))
    x_train.iloc[:, :raw_x_train.shape[1]] = raw_x_train
    x_train = torch.from_numpy(x_train.values).float().view(-1, 12, 12)
    y_train = torch.zeros(len(x_train))


    raw_x_test = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Xtest.csv").drop(columns=["Row"], axis=1)
    x_test = pd.DataFrame(np.zeros((raw_x_test.shape[0], 144)))
    x_test.iloc[:, :raw_x_test.shape[1]] = raw_x_test
    x_test = torch.from_numpy(x_test.values).float().view(-1, 12, 12)
    y_test = pd.read_csv("/root/work/UGR16_FeatureData/csv/UGR16v1.Ytest.csv").drop(columns=["Row", "labelanomalyidpscan", "labelanomalysshscan", "labelanomalyidpscan", "labelblacklist"], axis=1)
    y_test = torch.from_numpy(y_test.apply(lambda row: 1 if row.sum() > 0 else 0, axis=1).values)
    return (x_train, y_train), (x_test, y_test)
# print(data["labeldos"].value_counts())
# print(data["labelscan11"].value_counts())
# print(data["labelscan44"].value_counts())
# print(data["labelnerisbotnet"].value_counts())
# print(data["labelblacklist"].value_counts())
# print(data["labelanomalyidpscan"].value_counts())
# print(data["labelanomalysshscan"].value_counts())
# print(data["labelanomalyspam"].value_counts())
