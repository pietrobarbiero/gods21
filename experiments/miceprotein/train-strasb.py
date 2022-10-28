import sys
sys.path.append('../..')
import wandb
from data import load_data, load_data_strasb
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
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import to_undirected, add_remaining_self_loops
import numpy as np
import seaborn as sns
import networkx as nx
from torch.nn.functional import one_hot, leaky_relu
from sklearn.feature_selection import RFECV

# feature_selectors = ['MECO', '']
# feature_selectors = []
# models = ['DT']
# models = ['RF']
# models = ['EVOLENS', 'RF', 'DT']
models = ['GCN', 'RF', 'DT']
# models = ['RF', 'DT']
# models = ['DT']
# models = ['GCN']


def save_artifact(artifact, artifact_name, split, run):
    artifact_obj = wandb.Artifact(
        artifact_name + f'-{split}', type="model",
        description=artifact_name,
        metadata=dict(run.config))

    model_path = os.path.join('./stras-artifacts', str(split), artifact_name)
    os.makedirs(model_path, exist_ok=True)
    if artifact_name == 'GCN':
        artifact_file = os.path.join(model_path, 'model.pth')
        # torch.save(artifact.state_dict(), artifact_file)
        torch.save(artifact, artifact_file)
    else:
        artifact_file = os.path.join(model_path, 'model.pkl')
        joblib.dump(artifact, artifact_file)

    artifact_obj.add_file(artifact_file)
    run.log_artifact(artifact_obj)
    return


def load_artifact(artifact, artifact_name, split, run):
    # model_at = run.use_artifact(artifact_name + f'-{split}:latest')
    # model_path = model_at.download()
    model_path = os.path.join('./stras-artifacts', str(split), artifact_name)
    if artifact_name == 'GCN':
        # artifact.load_state_dict(torch.load(os.path.join(model_path, 'model.pth')))
        artifact = torch.load(os.path.join(model_path, 'model.pth'))
    else:
        artifact = joblib.load(os.path.join(model_path, "model.pkl"))

    return artifact


def cv_loop(model_name, x, y, train_index, test_index, split, fnames, cnames):
    os.makedirs(f'./stras-artifacts/{split}', exist_ok=True)
    trained_models = os.listdir(f'./stras-artifacts/{split}')

    run = wandb.init(project="mice-protein", job_type='train', reinit=True,
                     dir='./stras-results', group=f"{model_name}")
    # run.config.update({"max_features": 10}, allow_val_change=True)
    # run.config.update({"max_generations": 2}, allow_val_change=True)
    run.config.update({"n_features": x.shape[1]}, allow_val_change=True)
    run.config.update({"n_classes": len(np.unique(y))}, allow_val_change=True)

    model = build_model(model_name, run.config, train_index=train_index, test_index=test_index)

    if f'{model_name}' not in trained_models:
        # skb = SelectKBest(f_classif, k=100)
        rfe = RFECV(RandomForestClassifier(random_state=42), cv=3, verbose=1, n_jobs=3, min_features_to_select=1)#, min_features_to_select=x.shape[1]-2)
        # rfe.fit(x[train_index], y[train_index])
        rfe.fit(x, y)
        joblib.dump(rfe, os.path.join('./stras-artifacts', f'rfe.joblib'))
        # x_new = rfe.transform(x)
        # x_new = torch.FloatTensor(x_new)
        # https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html#sphx-glr-auto-examples-feature-selection-plot-rfe-with-cross-validation-py
        # if model_name == 'EVOLENS':
        #     model.fit(x_new, y)
        # elif model_name == 'GCN':
            # model = train_lens(x_new, y, [], train_index, run.config['temperature'])
            # print(model.lens.alpha_norm.median(dim=0)[0].detach().numpy())
            # print(model.lens.alpha_norm.max(dim=0)[0].detach().numpy())
            # print(model.lens.alpha_norm.min(dim=0)[0].detach().numpy())
            # explanations = test_lens(model, x_new, y, [], train_index, test_index, fnames[rfe.support_], cnames)
            # joblib.dump(explanations, os.path.join('./stras-artifacts', str(split), f'explanations.joblib'))
            # joblib.dump(rfe, os.path.join('./stras-artifacts', str(split), f'rfe.joblib'))

        # else:
        #     model.fit(x_new[train_index], y[train_index].numpy())
        #     print(f'{model_name} trained')
        #     print(model.score(x_new[test_index], y[test_index].numpy()))
        #     # print(model.tree_.node_count)
        #     # return
        #
        # save_artifact(model, model_name, split, run)

    model = load_artifact(model, model_name, split, run)
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
    os.makedirs('./stras-results', exist_ok=True)
    os.makedirs('./stras-artifacts', exist_ok=True)

    # load data
    x, y, _, fnames, cnames = load_data_strasb(data_path='../../data/strasb/')

    sss = StratifiedShuffleSplit(n_splits=10, random_state=42)
    sss.get_n_splits(x, y)
    # y1h = one_hot(torch.LongTensor(y)).float()
    joblib.dump(fnames, os.path.join('./stras-artifacts', f'fnames.joblib'))
    for split, (train_index, test_index) in enumerate(sss.split(x, y)):
        for model_name in models:
            train_index = test_index = np.arange(0, x.shape[1], 1)
            cv_loop(model_name, x, y, train_index, test_index, 0, fnames, cnames)

            return


if __name__ == '__main__':
    main()
