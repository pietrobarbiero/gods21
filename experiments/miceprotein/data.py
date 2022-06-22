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
    return x, y, is_control
