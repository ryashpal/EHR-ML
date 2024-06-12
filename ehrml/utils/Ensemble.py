import os

import pickle

from importlib import import_module

from pathlib import Path

import pandas as pd

from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

import logging

log = logging.getLogger("EHR-ML")

from ehrml.utils import DataUtils


def predict(dirPath, idColumns, targetColumn, measurementDateColumn, anchorDateColumn, windowStart, windowEnd, modelPath, savePath):

    log.info('Reading data from file: ' + str(dirPath))

    data = DataUtils.readData(
        dirPath=dirPath,
        idColumns=idColumns,
        targetColumn=targetColumn,
        measurementDateColumn=measurementDateColumn,
        anchorDateColumn=anchorDateColumn,
        windowStart=windowStart,
        windowEnd=windowEnd,
        )
    X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y, idsDf = data

    dataDict = {
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

    log.info('Predicting using XGB ensemble model file: ' + str(modelPath))

    allModelsDict = {}

    with open(modelPath, 'rb') as f:
        allModelsDict = pickle.load(f)

    standaloneModelsDict = allModelsDict['level_0']
    Xnew = pd.DataFrame()
    for label in standaloneModelsDict.keys():
        for model_name in standaloneModelsDict[label].keys():
            log.info('Performing prediction for the label: ' + label + ', model_name: ' + model_name)
            model = standaloneModelsDict[label][model_name]
            probs = [p for _, p in model.predict_proba(dataDict[label])]
            Xnew[label + '_' + model_name] = probs

    probs = [p for _, p in allModelsDict['level_1'].predict_proba(Xnew)]

    saveDf = idsDf
    saveDf['probs'] = probs

    dirPath = Path(savePath).parent
    if not os.path.exists(dirPath):
        log.info('Creating directory: ' + str(dirPath))
        os.makedirs(dirPath)

    log.info('Saving to file: ' + str(savePath))

    saveDf.to_csv(savePath, index=False)


def build(dirPath, idColumns, targetColumn, measurementDateColumn, anchorDateColumn, windowStart, windowEnd, configFile, savePath):
    dataDict = DataUtils.getDataDict(
        dirPath=dirPath,
        idColumns=idColumns,
        targetColumn=targetColumn,
        measurementDateColumn=measurementDateColumn,
        anchorDateColumn=anchorDateColumn,
        windowStart=windowStart,
        windowEnd=windowEnd,
    )

    log.info('Building standalone models')

    allModelsDict = {}
    standaloneModelsDict = {}
    for label, (XTrain, yTrain, XTest, yTest) in dataDict.items():
        levelZeroModelsDict = {}
        if not XTrain.empty:
            log.info('Models for the label: ' + label)
            levelZeroModelsConfig = import_module(configFile).levelZeroModels
            for model_name in levelZeroModelsConfig.keys():
                log.info('Building for the label: ' + label + ', model_name: ' + model_name)
                levelZeroModel = levelZeroModelsConfig[model_name]['model']()
                params = {}
                for searchParams in levelZeroModelsConfig[model_name]['search_params']:
                    params.update(getBestHyperparameter(XTrain, yTrain.values.ravel(), params, searchParams))
                log.info('Optimised Params: ' + str(params))
                levelZeroModel.fit(XTrain, yTrain.values.ravel())
                levelZeroModel.feature_names = list(XTrain.columns.values)
                levelZeroModelsDict[model_name] = levelZeroModel
        standaloneModelsDict[label] = levelZeroModelsDict

    log.info('Performing prediction for the level 0 models')

    probabilityFeatures = pd.DataFrame()
    for label in standaloneModelsDict.keys():
        for model_name in standaloneModelsDict[label].keys():
            model = standaloneModelsDict[label][model_name]
            probs = [p for _, p in model.predict_proba(dataDict[label][2])]
            auroc = roc_auc_score(dataDict[label][3], probs)
            log.info('label: ' + label + ', model: ' + model_name + ' - Model (Testing) AUROC score: ' + str(auroc))
            probabilityFeatures[label + '_' + model_name] = probs

    log.info('Building XGB Ensemble model')

    xgb = XGBClassifier(use_label_encoder=False)
    xgb.fit(probabilityFeatures, yTest)
    probs = [p for _, p in xgb.predict_proba(probabilityFeatures)]
    auroc = roc_auc_score(yTest, probs)

    log.info('XGB Ensemble Model (Training) AUROC score: ' + str(auroc))

    allModelsDict = {'level_1': xgb, 'level_0': standaloneModelsDict}

    savePath = Path(savePath)
    DataUtils.saveModels(allModelsDict, savePath.parent, savePath.name)


def getBestHyperparameter(X, y, tuned_params, parameters):

    from xgboost import XGBClassifier
    from sklearn.model_selection import GridSearchCV

    params = {}

    log.info('Hyperparameter optimisation for: ' + str(parameters))

    clf = GridSearchCV(XGBClassifier(use_label_encoder=False, **tuned_params), parameters)
    clf.fit(X, y)

    params = clf.cv_results_['params'][list(clf.cv_results_['rank_test_score']).index(1)]

    return(params)

if __name__ == '__main__':

    import logging
    import sys

    log = logging.getLogger("EHR-ML")
    log.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)

    import warnings

    warnings.filterwarnings("ignore")

    log.info("Parsing command line arguments")

    # build(
    #     dirPath='/home/vmadmin/workspace/ehr_data/blood_pos_cohort_20240531/data/wb_365_wa_1/splits/los/normal_train.csv',
    #     idColumns=['person_id', 'JOURNEY_ID'],
    #     targetColumn='los_gt_30_days',
    #     measurementDateColumn='measurement_date',
    #     anchorDateColumn='admittime_adm',
    #     windowStart=365,
    #     windowEnd=2,
    #     configFile='ehrml.config.config',
    #     savePath='/tmp/model.pkl'
    #     )

    predict(
        dirPath='/home/vmadmin/workspace/ehr_data/blood_pos_cohort_20240531/data/wb_365_wa_1/splits/los/normal_train.csv',
        idColumns=['person_id', 'JOURNEY_ID'],
        targetColumn='los_gt_30_days',
        measurementDateColumn='measurement_date',
        anchorDateColumn='admittime_adm',
        windowStart=365,
        windowEnd=2,
        modelFilePath='/tmp/model.pkl',
        savePath='/tmp/preds.csv'
        )
