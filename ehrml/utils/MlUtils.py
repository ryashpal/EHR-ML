import logging

log = logging.getLogger("EHR-ML")


def calculateMccF1(x, y):
    import sys
    import os

    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    # import R's "base" package
    mccf1 = importr('mccf1')

    old_stdout = sys.stdout # backup current stdout
    sys.stdout = open(os.devnull, "w")
    print('y', y)
    print('x', x)
    p = robjects.FloatVector(x)
    t = robjects.FloatVector(y)
    calculateMccf1 = robjects.r['mccf1']
    summary = robjects.r['summary']
    out = summary(calculateMccf1(t, p), 50)[0][0]
    sys.stdout = old_stdout # reset old stdout
    return out


def performSfs(X, y):
    log.info('Performing SFS')

    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.tree import DecisionTreeClassifier

    sfs = SequentialFeatureSelector(estimator = DecisionTreeClassifier(), n_features_to_select=25)

    sfs.fit(X, y)

    XMin = X[sfs.get_feature_names_out()]
    return XMin


def buildMLPModel(X, y, layerSize):

    log.info('Building the model')

    from sklearn.metrics import make_scorer

    from sklearn.neural_network import MLPClassifier

    mlp = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes = (layerSize, layerSize))
    mlp.fit(X, y)

    log.info('Performing cross-validation')

    from sklearn.model_selection import cross_validate

    mlpScores = cross_validate(mlp, X, y, cv=5, scoring=['accuracy', 'balanced_accuracy', 'average_precision', 'f1', 'roc_auc'])
    mlpScores['test_mccf1_score'] = cross_validate(mlp, X, y, cv=5, scoring = make_scorer(calculateMccF1, greater_is_better=True))['test_score']
    return mlpScores


def buildLGBMModel(X, y):
    log.info('Performing Hyperparameter optimisation')

    from sklearn.metrics import make_scorer

    from lightgbm import LGBMClassifier

    from sklearn.model_selection import GridSearchCV

    parameters={
        'max_depth': [6, 9, 12],
        'scale_pos_weight': [0.1, 0.15, 0.2, 0.25, 0.3],
    }

    clf = GridSearchCV(LGBMClassifier(verbose=-1), parameters)

    import re
    data = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

    clf.fit(data, y)

    params = clf.cv_results_['params'][list(clf.cv_results_['rank_test_score']).index(1)]

    log.info('Building the model')

    lgbm = LGBMClassifier(verbose=-1)
    lgbm.set_params(**params)

    log.info('Performing cross-validation')

    from sklearn.model_selection import cross_validate

    lgbmScores = cross_validate(lgbm, data, y, cv=5, scoring=['accuracy', 'balanced_accuracy',  'average_precision', 'f1', 'roc_auc'])
    lgbmScores['test_mccf1_score'] = cross_validate(lgbm, data, y, cv=5, scoring = make_scorer(calculateMccF1, greater_is_better=True))['test_score']
    return lgbmScores


def buildLRModel(X, y):
    log.info('Performing Hyperparameter optimisation')

    log.info('y:')
    log.info(y)

    from sklearn.metrics import make_scorer

    from sklearn.linear_model import LogisticRegression

    from sklearn.model_selection import GridSearchCV

    parameters={
        'solver': ['newton-cg', 'liblinear'],
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    clf = GridSearchCV(LogisticRegression(), parameters)
    clf.fit(X, y)

    params = clf.cv_results_['params'][list(clf.cv_results_['rank_test_score']).index(1)]

    log.info('Building the model')

    lr = LogisticRegression()
    lr.set_params(**params)

    log.info('Performing cross-validation')

    from sklearn.model_selection import cross_validate

    lrScores = cross_validate(lr, X, y, cv=5, scoring=['accuracy', 'balanced_accuracy',  'average_precision', 'f1', 'roc_auc'])
    lrScores['test_mccf1_score'] = cross_validate(lr, X, y, cv=5, scoring = make_scorer(calculateMccF1, greater_is_better=True))['test_score']

    return lrScores


def getBestXgbHyperparameter(X, y, tuned_params, parameters):

    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV

    params = {}

    log.info('Hyperparameter optimisation for: ' + str(parameters))

    clf = GridSearchCV(XGBClassifier(use_label_encoder=False, **tuned_params), parameters)
    clf.fit(X, y)

    params = clf.cv_results_['params'][list(clf.cv_results_['rank_test_score']).index(1)]

    return(params)


def performXgbHyperparameterTuning(X, y):

    params = {}

    params.update(getBestXgbHyperparameter(X, y, params, {'max_depth' : range(1,10),'scale_pos_weight': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],}))

    params.update(getBestXgbHyperparameter(X, y, params, {'n_estimators':range(50,250,10)}))

    params.update(getBestXgbHyperparameter(X, y, params, {'min_child_weight':range(1,10)}))

    params.update(getBestXgbHyperparameter(X, y, params, {'gamma':[i/10. for i in range(0,5)]}))

    params.update(getBestXgbHyperparameter(X, y, params, {'subsample':[i/10.0 for i in range(1,10)],'colsample_bytree':[i/10.0 for i in range(1,10)]}))

    params.update(getBestXgbHyperparameter(X, y, params, {'reg_alpha':[0, 1e-5, 1e-3, 0.1, 10]}))

    log.info('params: ' + str(params))

    return params


def buildXGBoostModel(X, y):
    log.info('Performing Hyperparameter optimisation')

    from sklearn.metrics import make_scorer

    from xgboost import XGBClassifier

    log.info('Building the model')
    
    params = performXgbHyperparameterTuning(X, y)

    xgb = XGBClassifier(use_label_encoder=False)
    xgb.set_params(**params)

    log.info('Performing cross-validation')

    from sklearn.model_selection import cross_validate

    xgbScores = cross_validate(xgb, X, y, cv=5, scoring=['accuracy', 'balanced_accuracy',  'average_precision', 'f1', 'roc_auc'])
    xgbScores['test_mccf1_score'] = cross_validate(xgb, X, y, cv=5, scoring = make_scorer(calculateMccF1, greater_is_better=True))['test_score']

    return xgbScores


def buildEnsembleXGBoostModel(XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y):

    log.info('Split data to test and train sets')

    from sklearn.model_selection import train_test_split

    XVitalsAvgTrain, XVitalsAvgTest, XVitalsMinTrain, XVitalsMinTest, XVitalsMaxTrain, XVitalsMaxTest, XVitalsFirstTrain, XVitalsFirstTest, XVitalsLastTrain, XVitalsLastTest, XLabsAvgTrain, XLabsAvgTest, XLabsMinTrain, XLabsMinTest, XLabsMaxTrain, XLabsMaxTest, XLabsFirstTrain, XLabsFirstTest, XLabsLastTrain, XLabsLastTest, yTrain, yTest = train_test_split(
        XVitalsAvg,
        XVitalsMin,
        XVitalsMax,
        XVitalsFirst,
        XVitalsLast,
        XLabsAvg,
        XLabsMin,
        XLabsMax,
        XLabsFirst,
        XLabsLast,
        y,
        test_size=0.5,
        random_state=42
        )

    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    from lightgbm import LGBMClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer

    log.info('Performing Hyperparameter optimisation for XGBoost')

    xgbParams = performXgbHyperparameterTuning(XVitalsAvgTrain, yTrain)

    log.info('Performing Hyperparameter optimisation for Logistic Regression')

    lrParameters={
        'solver': ['newton-cg', 'liblinear'],
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    lrGrid = GridSearchCV(LogisticRegression(), lrParameters)
    lrGrid.fit(XVitalsAvgTrain, yTrain)

    lrParams = lrGrid.cv_results_['params'][list(lrGrid.cv_results_['rank_test_score']).index(1)]

    XDict = {
        'VitalsMax': (XVitalsMaxTrain, XVitalsMaxTest),
        'VitalsMin': (XVitalsMinTrain, XVitalsMinTest),
        'VitalsAvg': (XVitalsAvgTrain, XVitalsAvgTest),
        'VitalsFirst': (XVitalsFirstTrain, XVitalsFirstTest),
        'VitalsLast': (XVitalsLastTrain, XVitalsLastTest),
        'LabsMax': (XLabsMaxTrain, XLabsMaxTest),
        'LabsMin': (XLabsMinTrain, XLabsMinTest),
        'LabsAvg': (XLabsAvgTrain, XLabsAvgTest),
        'LabsFirst': (XLabsFirstTrain, XLabsFirstTest),
        'LabsLast': (XLabsLastTrain, XLabsLastTest),
    }

    probsDict = {}

    log.info('Building individual models')

    for label, (XTrain, XTest) in XDict.items():

        xgb = XGBClassifier(use_label_encoder=False)
        xgb.set_params(**xgbParams)
        xgb.fit(XTrain, yTrain)

        xgbProbs = [p for _, p in xgb.predict_proba(XTest)]

        probsDict[('XGB', label)] = xgbProbs

        lr = LogisticRegression()
        lr.set_params(**lrParams)
        lr.fit(XTrain, yTrain)

        lrProbs = [p2 for p1, p2 in lr.predict_proba(XTest)]

        probsDict[('LR', label)] = lrProbs

        lgbm = LGBMClassifier(verbose=-1)
        lgbm.set_params(**xgbParams)
        lgbm.fit(XTrain, yTrain)

        lgbmProbs = [p2 for p1, p2 in lgbm.predict_proba(XTest)]

        probsDict[('LGBM', label)] = lgbmProbs

        mlp = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes = (200, 200))
        mlp.fit(XTrain, yTrain)

        mlpProbs = [p2 for p1, p2 in mlp.predict_proba(XTest)]

        probsDict[('MLP', label)] = mlpProbs

    log.info('Building ensemble model')

    import pandas as pd

    Xnew = pd.DataFrame()

    for key, value in probsDict.items():
        Xnew[key[0] + '_' + key[1]] = value

    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_validate

    xgb = XGBClassifier(use_label_encoder=False)
    xgbScores = cross_validate(xgb, Xnew, yTest, cv=5, scoring=['accuracy', 'balanced_accuracy',  'average_precision', 'f1', 'roc_auc'])
    xgbScores['test_mccf1_score'] = cross_validate(xgb, Xnew, yTest, cv=5, scoring = make_scorer(calculateMccF1, greater_is_better=True))['test_score']
    # xgbEnsembleNewScores = buildXGBoostModel(Xnew, yTest)

    return xgbScores
