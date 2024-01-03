import os

import logging

log = logging.getLogger("EHR-ML")

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from ehrml.utils import DataUtils
from ehrml.utils import MlUtils

from ehrml.ensemble.Evaluate import run as evaluate_run

from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from pathlib import Path


def run(dataPath, idColumns, targetColumn, measurementDateColumn, anchorDateColumn, windowStart, windowEnd, sampleSizes, ensemble, savePath):

    datamatrixDf = pd.read_csv(dataPath)

    for sameplSize in sampleSizes:

            log.info('Running evaluation for sample size: ' + str(sameplSize))

            sampledDataMatrix = datamatrixDf[datamatrixDf.visit_occurrence_id.isin(np.random.choice(datamatrixDf.visit_occurrence_id.unique(), size=sameplSize, replace=False))]
            sampledDataMatrix.to_csv(savePath + '/data_matrix_sample_' + str(sameplSize) + '.csv', index=False)

            if ensemble:
                evaluate_run(
                    dataPath=savePath + '/data_matrix_sample_' + str(sameplSize) + '.csv',
                    idColumns=idColumns,
                    targetColumn=targetColumn,
                    measurementDateColumn=measurementDateColumn,
                    anchorDateColumn=anchorDateColumn,
                    windowStart=windowStart,
                    windowEnd=windowEnd,
                    savePath=savePath + '/wb_' + str(windowStart) + '_wa_' + str(windowEnd) + '_sample_' + str(sameplSize) + '.json'
                )
            else:
                data = DataUtils.readData(
                    dirPath=savePath + '/data_matrix_sample_' + str(sameplSize) + '.csv',
                    idColumns=idColumns,
                    targetColumn=targetColumn,
                    measurementDateColumn=measurementDateColumn,
                    anchorDateColumn=anchorDateColumn,
                    windowStart=windowStart,
                    windowEnd=windowEnd,
                    )
                X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y, idsDf = data

                xgb = XGBClassifier(use_label_encoder=False)
                xgbScores = cross_validate(xgb, X, y, cv=5, scoring=['accuracy', 'balanced_accuracy',  'average_precision', 'f1', 'roc_auc'])
                xgbScores['test_mccf1_score'] = cross_validate(xgb, X, y, cv=5, scoring = make_scorer(MlUtils.calculateMccF1, greater_is_better=True))['test_score']

                log.info('xgbScores: ' + str(xgbScores))

                if not os.path.exists(savePath):
                    log.info('Creating directory: ' + str(savePath))
                    os.makedirs(savePath)

                log.info('saving the predictions')

                savePath = Path(savePath)
                DataUtils.saveCvScores(xgbScores, savePath, 'wb_' + str(windowStart) + '_wa_' + str(windowEnd) + '_sample_' + str(sameplSize) + '.json')


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

    parser.add_argument('data_file', nargs=1, default='data.csv',
                        help='Path of the data file in csv format')

    parser.add_argument('-tc', '--target_column', nargs=1, default=['target'],
                        help='Name of the column containing the target variable')

    parser.add_argument('-ic', '--id_columns', nargs='*',
                        help='Name/s of the columns containing the the IDs')

    parser.add_argument('-mdc', '--measurement_date_column', nargs=1, default=['measurement_date'],
                        help='Name of the column containing the measurement date')

    parser.add_argument('-adc', '--anchor_date_column', nargs=1, default=['visit_start_datetime'],
                        help='Name of the anchor date column')

    parser.add_argument('-wb', '--window_before', nargs=1, type=int, default=0,
                        help='Number of days or data to include before time-zero. By default: [window_before=0]')

    parser.add_argument('-wa', '--window_after', nargs=1, type=int, default=3,
                        help='Number of days or data to include after time-zero. By default: [window_after=3]')

    parser.add_argument('-ss', '--sample_size', type=int, nargs='*',
                        help='Sample Sizes')

    parser.add_argument('-e', '--ensemble', action='store_true', default=False,
                        help='Use ensemble model.')

    parser.add_argument('-sp', '--save_path', nargs=1, default='./',
                        help='Directory path to save the results and intermediate files')

    args = parser.parse_args()

    log.info('args.data_file: ' + str(args.data_file[0]))
    log.info('args.target_column: ' + str(args.target_column[0]))
    log.info('args.id_columns: ' + str(args.id_columns))
    log.info('args.measurement_date_column: ' + str(args.measurement_date_column[0]))
    log.info('args.anchor_date_column: ' + str(args.anchor_date_column[0]))
    log.info('args.window_before: ' + str(args.window_before[0]))
    log.info('args.window_after: ' + str(args.window_after[0]))
    log.info('args.sample_size: ' + str(args.sample_size))
    log.info('args.ensemble: ' + str(args.ensemble))
    log.info('args.save_path: ' + str(args.save_path[0]))

    run(
        dataPath=args.data_file[0],
        idColumns=args.id_columns,
        targetColumn=args.target_column[0],
        measurementDateColumn=args.measurement_date_column[0],
        anchorDateColumn=args.anchor_date_column[0],
        windowStart=args.window_before[0],
        windowEnd=args.window_after[0],
        sampleSizes=args.sample_size,
        ensemble=args.ensemble,
        savePath=args.save_path[0]
    )
