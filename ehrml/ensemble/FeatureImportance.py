import os

import logging

log = logging.getLogger("EHR-ML")

import warnings

warnings.filterwarnings("ignore")

import pickle
import pandas as pd

from pathlib import Path


def run(modelPath, intermediate, savePath):

    featureImportanceDfList = []

    with open(modelPath, 'rb') as f:
        allModelsDict = pickle.load(f)

    levelOneModelsDict = allModelsDict['level_1']
    levelOneScoresDf = pd.DataFrame(zip(levelOneModelsDict.feature_names_in_, levelOneModelsDict.feature_importances_), columns=['Feature_Name', 'Feature_Importance'])
    levelOneScoresDf = levelOneScoresDf[levelOneScoresDf.Feature_Name.str.endswith('_xgb')]
    levelOneScoresDf.Feature_Importance = levelOneScoresDf.Feature_Importance/sum(levelOneScoresDf.Feature_Importance)

    if intermediate:
        featureImportanceDfList.append(levelOneScoresDf)

    levelZeroModelsDict = allModelsDict['level_0']
    levelZeroScoresDfList = []
    for label in levelZeroModelsDict.keys():
        model = 'xgb'
        levelOneFeatureName = label + '_' + model
        levelZeroXgbModel = levelZeroModelsDict[label][model]
        levelOneFeatureImportance = levelOneScoresDf[levelOneScoresDf.Feature_Name == levelOneFeatureName].Feature_Importance.values[0]
        levelZeroScoresDf = pd.DataFrame(zip(levelZeroXgbModel.feature_names_in_, levelZeroXgbModel.feature_importances_), columns=['Feature_Name', 'Feature_Importance'])
        levelZeroScoresDf.Feature_Importance = levelZeroScoresDf.Feature_Importance * levelOneFeatureImportance
        levelZeroScoresDfList.append(levelZeroScoresDf)
    levelZeroAllScoresDf = pd.concat(levelZeroScoresDfList, ignore_index=True)
    levelZeroAllScoresDf = levelZeroAllScoresDf.groupby(by='Feature_Name').sum().reset_index()

    featureImportanceDfList.append(levelZeroAllScoresDf)

    dirPath = Path(savePath).parent
    if not os.path.exists(dirPath):
        log.info('Creating directory: ' + str(dirPath))
        os.makedirs(dirPath)

    log.info('Saving to file: ' + str(savePath))

    featureImportanceDf = pd.concat(featureImportanceDfList, ignore_index=True)

    featureImportanceDf.to_csv(savePath, index=False)


if __name__ == '__main__':

    import logging
    import sys
    import argparse

    log = logging.getLogger("EHR-ML")
    log.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(format)
    log.addHandler(ch)

    log.info("Parsing command line arguments")

    parser = argparse.ArgumentParser(description='EHR-ML machine learning utility')

    parser.add_argument('model_file', nargs=1, default='model.pkl',
                        help='Path of the ensemble model file in pkl format')

    parser.add_argument('-i', '--intermediate', action='store_true',
                        help='Include feature importance of intermediate features')

    parser.add_argument('-sp', '--save_path', nargs=1, default='./ensemble_feature_importance.csv',
                        help='File path to save the feature importance')

    args = parser.parse_args()

    log.info('args.model_file: ' + str(args.model_file[0]))
    log.info('args.intermediate: ' + str(args.intermediate))
    log.info('args.save_path: ' + str(args.save_path[0]))

    run(
        modelPath=args.model_file[0],
        intermediate=args.intermediate,
        savePath=args.save_path[0]
    )
