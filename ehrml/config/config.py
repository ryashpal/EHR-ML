from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


xgb = XGBClassifier(use_label_encoder=False)
lr = LogisticRegression(max_iter=300)
lgbm = LGBMClassifier(verbose=-1)
mlp = MLPClassifier(random_state=1, max_iter=300)


def getXGBClassifier(params={}):
    return XGBClassifier(use_label_encoder=False, **params)


def getLogisticRegression(params={}):
    return LogisticRegression(max_iter=300, **params)


def getLGBMClassifier(params={}):
    return LGBMClassifier(verbose=-1, **params)


def getMLPClassifier(params={}):
    return MLPClassifier(max_iter=300, **params)


def getSVC(params={}):
    return svm.SVC(probability=True, **params)


def getRandomForestClassifier(params={}):
    return RandomForestClassifier(**params)


def getGaussianNB(params={}):
    return GaussianNB(**params)


levelZeroModels = {
    'xgb': {
        'model': getXGBClassifier,
        'search_params' : [
        {'max_depth' : range(1,10)},
        # 'scale_pos_weight': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        # 'n_estimators':range(50,250,10),
        # 'min_child_weight':range(1,10),
        # 'gamma':[i/10. for i in range(0,5)],
        # 'subsample':[i/10.0 for i in range(1,10)],
        # 'colsample_bytree':[i/10.0 for i in range(1,10)],
        # 'reg_alpha':[0, 1e-5, 1e-3, 0.1, 10]
        ]
    },
    'lr': {
        'model': getLogisticRegression,
        'search_params' : [
            {'solver': ['newton-cg', 'liblinear']},
            {'C': [1000, 100, 10, 1.0, 0.1, 0.01]},
        ]
    },
    # 'lgbm': {
    #     'model': getLGBMClassifier,
    #     'search_params' : {
    #     'max_depth' : range(1,10),
    #     'scale_pos_weight': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
    #     'n_estimators':range(50,250,10),
    #     'min_child_weight':range(1,10),
    #     'gamma':[i/10. for i in range(0,5)],
    #     'subsample':[i/10.0 for i in range(1,10)],
    #     'colsample_bytree':[i/10.0 for i in range(1,10)],
    #     'reg_alpha':[0, 1e-5, 1e-3, 0.1, 10]
    #     }
    # },
    'mlp': {
        'model': getMLPClassifier,
        'search_params' : [
            {'hidden_layer_sizes': [(50, 50), (100, 100), (150, 150)]},
            {'max_iter': [1, 2, 5, 10, 20, 50, 100, 200, 300]}
        ]
    },
    # 'svc': {
    #     'model': getSVC,
    #     'search_params' : {
    #     }
    # },
    # 'rfc': {
    #     'model': getRandomForestClassifier,
    #     'search_params' : {
    #     }
    # },
    # 'nbc': {
    #     'model': getGaussianNB,
    #     'search_params' : {
    #     }
    # },
}
