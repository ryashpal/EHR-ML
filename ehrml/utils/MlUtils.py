import pickle

import pandas as pd

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


def performLrHyperparameterTuning(X, y):

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    lrParameters={
        'solver': ['newton-cg', 'liblinear'],
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    log.info('Hyperparameter optimisation for: ' + str(lrParameters))

    lrGrid = GridSearchCV(LogisticRegression(), lrParameters)
    lrGrid.fit(X, y)

    params = lrGrid.cv_results_['params'][list(lrGrid.cv_results_['rank_test_score']).index(1)]

    log.info('params: ' + str(params))

    return params


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


def evaluateEnsembleXGBoostModel(X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y):

    log.info('Split data to test and train sets')

    from sklearn.model_selection import train_test_split

    XTrain, XTest, XVitalsAvgTrain, XVitalsAvgTest, XVitalsMinTrain, XVitalsMinTest, XVitalsMaxTrain, XVitalsMaxTest, XVitalsFirstTrain, XVitalsFirstTest, XVitalsLastTrain, XVitalsLastTest, XLabsAvgTrain, XLabsAvgTest, XLabsMinTrain, XLabsMinTest, XLabsMaxTrain, XLabsMaxTest, XLabsFirstTrain, XLabsFirstTest, XLabsLastTrain, XLabsLastTest, yTrain, yTest = train_test_split(
        X,
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

    log.info('Performing Hyperparameter optimisation for XGBoost smaller models')

    # xgbParams = {}
    xgbParams = performXgbHyperparameterTuning(XVitalsAvgTrain, yTrain)

    log.info('Performing Hyperparameter optimisation for Logistic Regression smaller models')

    lrParameters={
        'solver': ['newton-cg', 'liblinear'],
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    lrGrid = GridSearchCV(LogisticRegression(), lrParameters)
    lrGrid.fit(XVitalsAvgTrain, yTrain)

    # lrParams = {}
    lrParams = lrGrid.cv_results_['params'][list(lrGrid.cv_results_['rank_test_score']).index(1)]

    log.info('Performing Hyperparameter optimisation for XGBoost full model')

    # xgbFullParams = {}
    xgbFullParams = performXgbHyperparameterTuning(XTrain, yTrain)

    log.info('Performing Hyperparameter optimisation for Logistic Regression full model')

    lrParameters={
        'solver': ['newton-cg', 'liblinear'],
        'C': [100, 10, 1.0, 0.1, 0.01],
    }

    lrGrid = GridSearchCV(LogisticRegression(), lrParameters)
    lrGrid.fit(XTrain, yTrain)

    # lrFullParams = {}
    lrFullParams = lrGrid.cv_results_['params'][list(lrGrid.cv_results_['rank_test_score']).index(1)]

    XDict = {
        'Full': (XTrain, XTest, xgbFullParams, lrFullParams),
        'VitalsMax': (XVitalsMaxTrain, XVitalsMaxTest, xgbParams, lrParams),
        'VitalsMin': (XVitalsMinTrain, XVitalsMinTest, xgbParams, lrParams),
        'VitalsAvg': (XVitalsAvgTrain, XVitalsAvgTest, xgbParams, lrParams),
        'VitalsFirst': (XVitalsFirstTrain, XVitalsFirstTest, xgbParams, lrParams),
        'VitalsLast': (XVitalsLastTrain, XVitalsLastTest, xgbParams, lrParams),
        'LabsMax': (XLabsMaxTrain, XLabsMaxTest, xgbParams, lrParams),
        'LabsMin': (XLabsMinTrain, XLabsMinTest, xgbParams, lrParams),
        'LabsAvg': (XLabsAvgTrain, XLabsAvgTest, xgbParams, lrParams),
        'LabsFirst': (XLabsFirstTrain, XLabsFirstTest, xgbParams, lrParams),
        'LabsLast': (XLabsLastTrain, XLabsLastTest, xgbParams, lrParams),
    }

    probsDict = {}

    log.info('Building individual models')

    for label, (XAggTrain, XAggTest, xgbModelParams, lrModelParams) in XDict.items():

        xgb = XGBClassifier(use_label_encoder=False)
        xgb.set_params(**xgbModelParams)
        xgb.fit(XAggTrain, yTrain)

        xgbProbs = [p for _, p in xgb.predict_proba(XAggTest)]

        probsDict[('XGB', label)] = xgbProbs

        lr = LogisticRegression()
        lr.set_params(**lrModelParams)
        lr.fit(XAggTrain, yTrain)

        lrProbs = [p2 for p1, p2 in lr.predict_proba(XAggTest)]

        probsDict[('LR', label)] = lrProbs

        lgbm = LGBMClassifier(verbose=-1)
        lgbm.set_params(**xgbModelParams)
        lgbm.fit(XAggTrain, yTrain)

        lgbmProbs = [p2 for p1, p2 in lgbm.predict_proba(XAggTest)]

        probsDict[('LGBM', label)] = lgbmProbs

        mlp = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes = ((XAggTrain.shape[1] * 2), (XAggTrain.shape[1] * 2)))
        mlp.fit(XAggTrain, yTrain)

        mlpProbs = [p2 for p1, p2 in mlp.predict_proba(XAggTest)]

        probsDict[('MLP', label)] = mlpProbs

    log.info('Performing cross validation for ensemble model')

    import pandas as pd

    Xnew = pd.DataFrame()

    for key, value in probsDict.items():
        Xnew[key[0] + '_' + key[1]] = value

    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_validate

    xgb = XGBClassifier(use_label_encoder=False)
    xgbScores = cross_validate(xgb, Xnew, yTest, cv=5, scoring=['accuracy', 'balanced_accuracy',  'average_precision', 'f1', 'roc_auc'])
    xgbScores['test_mccf1_score'] = cross_validate(xgb, Xnew, yTest, cv=5, scoring = make_scorer(calculateMccF1, greater_is_better=True))['test_score']

    return xgbScores


def buildStandaloneModels(XTrain, yTrain):

        from xgboost import XGBClassifier
        from sklearn.linear_model import LogisticRegression
        from lightgbm import LGBMClassifier
        from sklearn.neural_network import MLPClassifier

        log.info('Performing Hyperparameter optimisation for XGBoost')

        xgbParams = performXgbHyperparameterTuning(XTrain, yTrain)

        log.info('Building XGB model')
        xgb = XGBClassifier(use_label_encoder=False)
        xgb.set_params(**xgbParams)
        xgb.fit(XTrain, yTrain)

        log.info('Performing Hyperparameter optimisation for Logistic Regression')

        lrParams = performLrHyperparameterTuning(XTrain, yTrain)

        log.info('Building LR Model')
        lr = LogisticRegression()
        lr.set_params(**lrParams)
        lr.fit(XTrain, yTrain)

        log.info('Building LGBM Model')
        lgbm = LGBMClassifier(verbose=-1)
        lgbm.set_params(**xgbParams)
        lgbm.fit(XTrain, yTrain)

        log.info('Building MLP Model')
        mlp = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes = (150, 150))
        mlp.fit(XTrain, yTrain)

        return {'xgb': xgb, 'lr': lr, 'lgbm': lgbm, 'mlp': mlp}


def buildEnsembleXGBoostModel(X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y):

    log.info('Split data to test and train sets')

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from xgboost import XGBClassifier

    XTrain, XTest, XVitalsAvgTrain, XVitalsAvgTest, XVitalsMinTrain, XVitalsMinTest, XVitalsMaxTrain, XVitalsMaxTest, XVitalsFirstTrain, XVitalsFirstTest, XVitalsLastTrain, XVitalsLastTest, XLabsAvgTrain, XLabsAvgTest, XLabsMinTrain, XLabsMinTest, XLabsMaxTrain, XLabsMaxTest, XLabsFirstTrain, XLabsFirstTest, XLabsLastTrain, XLabsLastTest, yTrain, yTest = train_test_split(
        X,
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

    XDict = {
        'Full': (XTrain, yTrain, XTest, yTest),
        'VitalsMax': (XVitalsMaxTrain, yTrain, XVitalsMaxTest, yTest),
        'VitalsMin': (XVitalsMinTrain, yTrain, XVitalsMinTest, yTest),
        'VitalsAvg': (XVitalsAvgTrain, yTrain, XVitalsAvgTest, yTest),
        'VitalsFirst': (XVitalsFirstTrain, yTrain, XVitalsFirstTest, yTest),
        'VitalsLast': (XVitalsLastTrain, yTrain, XVitalsLastTest, yTest),
        'LabsMax': (XLabsMaxTrain, yTrain, XLabsMaxTest, yTest),
        'LabsMin': (XLabsMinTrain, yTrain, XLabsMinTest, yTest),
        'LabsAvg': (XLabsAvgTrain, yTrain, XLabsAvgTest, yTest),
        'LabsFirst': (XLabsFirstTrain, yTrain, XLabsFirstTest, yTest),
        'LabsLast': (XLabsLastTrain, yTrain, XLabsLastTest, yTest),
    }

    log.info('Building standalone models')

    standaloneModelsDict = {}
    for label, (XTrain, yTrain, XTest, yTest) in XDict.items():
        if not XTrain.empty:
            log.info('Models for the label: ' + label)
            standaloneModelsDict[label] = buildStandaloneModels(XTrain, yTrain)

    Xnew = pd.DataFrame()
    for label in standaloneModelsDict.keys():
        for model_name in standaloneModelsDict[label].keys():
            log.info('Performing prediction for the label: ' + label + ', model_name: ' + model_name)
            model = standaloneModelsDict[label][model_name]
            probs = [p for _, p in model.predict_proba(XDict[label][2])]
            auroc = roc_auc_score(XDict[label][3], probs)
            log.info('label: ' + label + ', model: ' + model_name + ' - Model (Testing) AUROC score: ' + str(auroc))
            Xnew[label + '_' + model_name] = probs

    log.info('Performing Hyperparameter optimisation for XGBoost Ensemble model')

    xgbParams = {}
    # xgbParams = performXgbHyperparameterTuning(Xnew, yTest)

    log.info('Building XGB Ensemble model')

    xgb = XGBClassifier(use_label_encoder=False)
    xgb.set_params(**xgbParams)
    xgb.fit(Xnew, yTest)
    probs = [p for _, p in xgb.predict_proba(Xnew)]
    auroc = roc_auc_score(yTest, probs)

    log.info('XGB Ensemble Model (Training) AUROC score: ' + str(auroc))

    allModelsDict = {'level_1': xgb, 'level_0': standaloneModelsDict}

    return allModelsDict


def predictEnsembleXGBoostModel(X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y, modelFilePath):

    XDict = {
        'Full': X,
        'VitalsMax': XVitalsMax,
        'VitalsMin': XVitalsMin,
        'VitalsAvg': XVitalsAvg,
        'VitalsFirst': XVitalsFirst,
        'VitalsLast': XVitalsLast,
        'LabsMax': XLabsMax,
        'LabsMin': XLabsMin,
        'LabsAvg': XLabsAvg,
        'LabsFirst': XLabsFirst,
        'LabsLast': XLabsLast,
    }
    allModelsDict = {}

    log.info('Model file: ' + str(modelFilePath))

    with open(modelFilePath, 'rb') as f:
        allModelsDict = pickle.load(f)
    standaloneModelsDict = allModelsDict['level_0']
    Xnew = pd.DataFrame()
    for label in standaloneModelsDict.keys():
        for model_name in standaloneModelsDict[label].keys():
            log.info('Performing prediction for the label: ' + label + ', model_name: ' + model_name)
            model = standaloneModelsDict[label][model_name]
            probs = [p for _, p in model.predict_proba(XDict[label])]
            Xnew[label + '_' + model_name] = probs
    probs = [p for _, p in allModelsDict['level_1'].predict_proba(Xnew)]
    return probs
