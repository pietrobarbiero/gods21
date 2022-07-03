import os

from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import KNNImputer
import torch


def load_data(data_path='../../data/MiceProtein.arff'):
    data = arff.loadarff(data_path)
    df = pd.DataFrame(data[0])
    df.head()

    is_control = df['Genotype'] == b'Control'
    imputer = KNNImputer()
    x = imputer.fit_transform(df.iloc[:, 1:-4])
    control_signature = x[is_control].mean(axis=0)
    x = x - control_signature
    x = torch.FloatTensor(x)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df.iloc[:, -1])
    y = torch.LongTensor(y)
    return x, y, is_control, df.columns[1:-4], df.iloc[:, -1]


def load_data_strasb(data_path='../../data/strasb/'):
    df = pd.read_csv(os.path.join(data_path, 'all_vsd_counts.csv'), index_col=0)
    labels = pd.read_csv(os.path.join(data_path, 'RNASeq_phenotyped.csv'), sep='\t', index_col=0)

    labels = labels.loc[df.index]

    y = labels['YM.memory']
    y[y == -1] = 0

    is_control = labels['genotype'] == 'wt'
    imputer = KNNImputer()
    x = imputer.fit_transform(df)
    control_signature = x[is_control].mean(axis=0)
    x = x - control_signature
    x = torch.FloatTensor(x)

    y = torch.LongTensor(y)
    return x, y, is_control, df.columns, ['mem=1' if i==1 else 'mem=-1' for i in labels['YM.memory']]
