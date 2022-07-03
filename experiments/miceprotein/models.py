from meco import MECO
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


class GCN(nn.Module):
    def __init__(self, num_in_features, num_hidden_features, num_classes, temperature):
        super(GCN, self).__init__()
        self.num_classes = num_classes

        #         self.conv0 = GCNConv(num_hidden_features, num_hidden_features)
        #         self.conv1 = GCNConv(num_hidden_features, 1)

        # linear layers
        self.lens = te.nn.EntropyLinear(num_in_features, num_hidden_features, n_classes=num_classes, temperature=temperature)
        self.linear = nn.Linear(num_hidden_features, 1)

    def forward(self, x, edge_index):
        x = self.lens(x)
        x = leaky_relu(x)
        x = self.linear(x)

        #         preds = []
        #         for nc in range(self.num_classes):
        #             xc = self.conv0(x[:, nc], edge_index)
        #             xc = leaky_relu(xc)

        #             xc = self.conv1(xc, edge_index)
        #             xc = leaky_relu(xc)
        #             preds.append(xc)

        #         preds = torch.hstack(preds)
        return x


def train_lens(x, y, edges, train_index, temperature):
    model = GCN(x.shape[1], 50, len(torch.unique(y)), temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_form = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(10001):
        optimizer.zero_grad()
        y_pred = model(x, edges).squeeze(-1)
        loss = loss_form(y_pred[train_index], y[train_index])  # + 0.0001 * te.nn.functional.entropy_logic_loss(model)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f'Epoch {epoch}: {loss}')

    model.eval()
    return model


def test_lens(model, x, y, edges, train_index, test_index, fnames, cnames):
    y1h = one_hot(y, len(torch.unique(y)))
    y_pred = model(x, edges).squeeze(-1).detach()
    f1 = f1_score(y1h[test_index].argmax(dim=-1), y_pred[test_index].argmax(dim=-1), average='weighted')
    acc = accuracy_score(y1h[test_index].argmax(dim=-1), y_pred[test_index].argmax(dim=-1))
    print(f'Test F1: {f1}')
    explanations, _ = entropy.explain_classes(model, x, y1h, train_index, test_index, edge_index=edges,
                                           #                                            c_threshold=0., y_threshold=0, topk_explanations=500,
                                           #                                            max_minterm_complexity=3)
                                           c_threshold=0., y_threshold=0, topk_explanations=10000,
                                           max_minterm_complexity=5, concept_names=fnames, class_names=cnames)
    # for nc in range(y1h.shape[1]):
    #     explanations[f'{nc}']['explanation'] = te.logic.utils.replace_names(explanations[f'{nc}']['explanation'],
    #                                                                         fnames)
    #     explanations[f'{nc}']['name'] = cnames[nc]

    print(explanations)
    explanations['accuracy'] = acc
    explanations['f1'] = f1
    return explanations
