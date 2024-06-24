import os

import logging

log = logging.getLogger("EHR-ML")

import warnings

warnings.filterwarnings("ignore")

import pickle
import pandas as pd

from pathlib import Path


def run(modelPath, intermediate, plot, features, savePath):

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

    log.info('Saving to path: ' + str(savePath))

    Path(savePath).mkdir(parents=True, exist_ok=True)

    log.info('Saving feature importance scores')
    
    featureImportanceCsvFile = savePath + '/ensemble_feature_importance.csv'

    featureImportanceDf = pd.concat(featureImportanceDfList, ignore_index=True)

    featureImportanceDf.to_csv(featureImportanceCsvFile, index=False)

    if(plot):

        log.info('Plotting feature importance scores')

        featureImportancePlotFile = savePath + '/ensemble_feature_importance_' + str(features) + '.png'

        plotDf = featureImportanceDf.sort_values(by=['Feature_Importance'], ascending=False)[:features]

        from matplotlib import pyplot as plt

        plt.barh(y=plotDf.Feature_Name, width=plotDf.Feature_Importance)
        plt.savefig(featureImportancePlotFile, bbox_inches='tight')


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

    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot the important features')

    parser.add_argument('-f', '--features', nargs=1, type=int, default=10,
                        help='Number of features to plot (required only with the -p option). By default: [features=10]')

    parser.add_argument('-sp', '--save_path', nargs=1, default='./',
                        help='File path to save the feature importance')

    args = parser.parse_args()

    log.info('args.model_file: ' + str(args.model_file[0]))
    log.info('args.intermediate: ' + str(args.intermediate))
    log.info('args.plot: ' + str(args.plot))
    log.info('args.features: ' + str(args.features[0]))
    log.info('args.save_path: ' + str(args.save_path[0]))

    run(
        modelPath=args.model_file[0],
        intermediate=args.intermediate,
        plot=args.plot,
        features=int(args.features[0]),
        savePath=args.save_path[0]
    )
