import torch
import numpy as np
import torch_explain as te
from sklearn.linear_model import RidgeClassifier
from meco import MECO
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from evolens import EvoLENs


def build_model(model_name, config, train_index=None, test_index=None):
    if model_name == 'DT':
        model = DecisionTreeClassifier(random_state=42, max_depth=config['max_depth'])
    elif model_name == 'RF':
        model = RandomForestClassifier(random_state=42, max_depth=config['max_depth'])
    elif model_name == 'EVOLENS':
        layers = [
            te.nn.EntropyLinear(config['n_features'], 20, n_classes=config['n_classes']),
            # torch.nn.Linear(x.shape[1], 100),
            torch.nn.LeakyReLU(),
            # torch.nn.Linear(30, 10),
            # torch.nn.LeakyReLU(),
            torch.nn.Linear(20, 1),
            # torch.nn.Linear(100, len(np.unique(y))),
        ]
        model = torch.nn.Sequential(*layers)
        optimizer = 'adamw'
        loss_form = torch.nn.CrossEntropyLoss()
        model = EvoLENs(model, optimizer, loss_form, compression='features', train_epochs=config.train_epochs,
                        max_generations=config.max_generations, lr=config.lr, pop_size=config.pop_size,

                        trainval_index=train_index, test_index=test_index)
    else:
        return None

    return model
