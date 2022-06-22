import sys
sys.path.append('../..')
from data import load_data
from sklearn.linear_model import RidgeClassifier
from meco import MECO
from models import GCN, train_lens, test_lens
import os
import joblib
from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import torch
import torch_explain as te
from torch.nn.functional import one_hot
from torch_explain.logic.nn import entropy
from torch_explain.logic.metrics import test_explanation, complexity
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import to_undirected, add_remaining_self_loops
import numpy as np
import seaborn as sns
import networkx as nx
from torch.nn.functional import one_hot, leaky_relu


def main():
    x, y, is_control = load_data(data_path='../../data/MiceProtein.arff')

    meco = MECO(RidgeClassifier(random_state=42), compression='features', max_features=10, max_generations=100)
    meco.fit(x[train_index], y[train_index])

    x1 = meco.transform(x.numpy())

    for s in meco.solutions_:
        print(list(df.columns[1:-4][s.candidate[1]]))

    sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
    sss.get_n_splits(x1, y)
    y1h = one_hot(torch.LongTensor(y)).float()

    os.makedirs('./results/miceProtein/', exist_ok=True)
    fold = 0
    results = {}
    dt_node_count = []
    for train_index, test_index in sss.split(x1, y):
        clf1 = RandomForestClassifier(random_state=0)
        clf1.fit(x1[train_index], y[train_index])
        y_pred = clf1.predict(x1[test_index])
        f1_rf = f1_score(y[test_index], y_pred, average='weighted')
        print(f'Test F1 (RF): {f1_rf}')

        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(x1[train_index], y[train_index])
        y_pred = clf.predict(x1[test_index])
        f1_dt = f1_score(y[test_index], y_pred, average='weighted')
        print(f'Test F1 (DT): {f1_dt}')
        dt_node_count.append(clf.tree_.node_count)

        model = train_lens(x1, y, edges, train_index)

        fold += 1


if __name__ == 'main':
    main()
