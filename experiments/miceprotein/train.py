import sys
sys.path.append('../..')
import wandb
from data import load_data
from model_configs import build_model
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
from sklearn.model_selection import cross_val_score, StratifiedKFold
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

feature_selectors = ['MECO']
# models = ['DT']
# models = ['RF']
models = ['RF', 'DT']
# models = ['GCN', 'RF', 'DT']


def save_artifact(artifact, artifact_name, split, run):
    artifact_obj = wandb.Artifact(
        artifact_name + f'-{split}', type="model",
        description=artifact_name,
        metadata=dict(run.config))

    model_path = os.path.join('./artifacts', str(split), artifact_name)
    os.makedirs(model_path, exist_ok=True)
    if artifact_name == 'GCN':
        artifact_file = os.path.join(model_path, 'model.pth')
        torch.save(artifact.state_dict(), artifact_file)
    else:
        artifact_file = os.path.join(model_path, 'model.pkl')
        joblib.dump(artifact, artifact_file)

    artifact_obj.add_file(artifact_file)
    run.log_artifact(artifact_obj)
    return


def load_artifact(artifact, artifact_name, split, run):
    # model_at = run.use_artifact(artifact_name + f'-{split}:latest')
    # model_path = model_at.download()
    model_path = os.path.join('./artifacts', str(split), artifact_name)
    if artifact_name == 'GCN':
        artifact.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
    else:
        artifact = joblib.load(os.path.join(model_path, "model.pkl"))

    return artifact


def cv_loop(feature_selector_name, model_name, x, y, train_index, test_index, split):
    os.makedirs(f'./artifacts/{split}', exist_ok=True)
    trained_models = os.listdir(f'./artifacts/{split}')

    run = wandb.init(project="mice-protein", job_type='train', reinit=True,
                     dir='./results', group=f"{feature_selector_name}-{model_name}")
    # run.config.update({"max_features": 10}, allow_val_change=True)
    # run.config.update({"max_generations": 2}, allow_val_change=True)

    feature_selector, model = build_model(feature_selector_name, model_name, run.config)

    if f'{feature_selector_name}' not in trained_models:
        feature_selector.fit(x[train_index].numpy(), y[train_index].argmax(dim=-1).numpy())
        save_artifact(feature_selector, feature_selector_name, split, run)

    feature_selector = load_artifact(feature_selector, feature_selector_name, split, run)
    x_reduced = feature_selector.transform(x.numpy())

    if f'{model_name}' not in trained_models:
        model.fit(x_reduced[train_index], y[train_index].argmax(dim=-1).numpy())
        save_artifact(model, model_name, split, run)

    model = load_artifact(model, model_name, split, run)
    accuracy = model.score(x_reduced[test_index], y[test_index].argmax(dim=-1).numpy())
    f1 = f1_score(y[test_index].argmax(dim=-1).numpy(), model.predict(x_reduced[test_index]), average='macro')
    run.summary["accuracy"] = accuracy
    run.summary["f1"] = f1
    run.summary["n_features"] = len(feature_selector.best_set_['features'])
    run.summary["val_accuracy"] = feature_selector.best_set_['accuracy']

    run.finish()

    return


# def cv_test_loop(feature_selector_name, model_name, x, y, test_index, split):
#     run = wandb.init(project='mice-protein', job_type="inference",
#                      reinit=True, dir='./results', group=f"{feature_selector_name}-{model_name}")
#     feature_selector, model = build_model(feature_selector_name, model_name, run.config)
#     feature_selector = load_artifact(feature_selector, feature_selector_name, split, run)
#     model = load_artifact(model, model_name, split, run)
#
#     x_reduced = feature_selector.transform(x.numpy())
#     accuracy = model.score(x_reduced[test_index], y[test_index])
#     print(accuracy)
#     run.summary["accuracy"] = accuracy
#
#     run.finish()
#     return


def main():
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./artifacts', exist_ok=True)

    # load data
    x, y, is_control = load_data(data_path='../../data/MiceProtein.arff')

    sss = StratifiedShuffleSplit(n_splits=5, random_state=42)
    sss.get_n_splits(x, y)
    y1h = one_hot(torch.LongTensor(y)).float()
    for split, (train_index, test_index) in enumerate(sss.split(x, y)):
        for feature_selector_name in feature_selectors:
            for model_name in models:
                cv_loop(feature_selector_name, model_name, x, y1h, train_index, test_index, split)

    return


if __name__ == '__main__':
    main()
