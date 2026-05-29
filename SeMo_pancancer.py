# @Time 2025/6/11
# !/user/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.io
from torch import nn

# 读取模态数据
cnv = pd.read_csv(r'C:\Users\xiaol\Desktop\Recent\backup\MProject\MLOmics\Main_Dataset\Classification_datasets\Pan-cancer\Original\Pan-cancer_CNV.csv', index_col=0)
methy = pd.read_csv(r'C:\Users\xiaol\Desktop\Recent\backup\MProject\MLOmics\Main_Dataset\Classification_datasets\Pan-cancer\Original\Pan-cancer_Methy.csv', index_col=0)
mirna = pd.read_csv(r'C:\Users\xiaol\Desktop\Recent\backup\MProject\MLOmics\Main_Dataset\Classification_datasets\Pan-cancer\Original\Pan-cancer_miRNA.csv', index_col=0)
mrna = pd.read_csv(r'C:\Users\xiaol\Desktop\Recent\backup\MProject\MLOmics\Main_Dataset\Classification_datasets\Pan-cancer\Original\Pan-cancer_mRNA.csv', index_col=0)

# 读取label
labels = pd.read_csv(r'C:\Users\xiaol\Desktop\Recent\backup\MProject\MLOmics\Main_Dataset\Classification_datasets\Pan-cancer\Original\Pan-cancer_label_num.csv', index_col=0)

# 找出共同的样本
# common_ids = set(cnv.index) & set(methy.index) & set(mirna.index) & set(mrna.index)
# common_ids = sorted(list(common_ids))

# 提取对应的模态和标签
# cnv = cnv.loc[common_ids]
# methy = methy.loc[common_ids]
# mirna = mirna.loc[common_ids]
# mrna = mrna.loc[common_ids]
# y = labels.loc[common_ids].values.squeeze() # 标签数组
y = labels.index

# concate all modalities
# X = np.concatenate([cnv.values, methy.values, mirna.values, mrna.values], axis=1)
X = {
    'cnv': cnv.values,
    'methy': methy.values,
    'mirna': mirna.values,
    'mrna': mrna.values,
}

scipy.io.savemat('Pan_cancer_all.mat', {
    'cnv': cnv.values,
    'methy': methy.values,
    'mirna': mirna.values,
    'mrna': mrna.values,
    'y': y.values
})


if __name__ == '__main__':
    # print(cnv.values.shape)
    # print(methy.shape)
    # print(mirna.shape)
    # print(mrna.shape)
    print(y.shape)
    print(y[0])
    print("save data")