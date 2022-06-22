from sklearn.linear_model import RidgeClassifier
from meco import MECO
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def build_model(feature_selector_name, model_name, config):
    if feature_selector_name == 'MECO':
        feature_selector = MECO(RidgeClassifier(random_state=42), compression='features',
                                max_features=config.max_features, max_generations=config.max_generations)
    else:
        return None, None

    if model_name == 'DT':
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == 'RF':
        model = RandomForestClassifier(random_state=42)
    else:
        return None, None

    return feature_selector, model
