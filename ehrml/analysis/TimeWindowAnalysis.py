import os

import logging

log = logging.getLogger("EHR-ML")

import warnings

warnings.filterwarnings("ignore")

from ehrml.utils import DataUtils
from ehrml.utils import MlUtils

from ehrml.ensemble.Evaluate import run as evaluate_run

from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from pathlib import Path


def run(dataPath, idColumns, targetColumn, measurementDateColumn, anchorDateColumn, windowStartList, windowEndList, ensemble, savePath):

    for windowStart in windowStartList:
        for windowEnd in windowEndList:
            log.info('Processing - Window Start: ' + str(windowStart) + ' Window End: ' + str(windowEnd))
            if ensemble:
                evaluate_run(
                    dataPath=dataPath,
                    idColumns=idColumns,
                    targetColumn=targetColumn,
                    measurementDateColumn=measurementDateColumn,
                    anchorDateColumn=anchorDateColumn,
                    windowStart=windowStart,
                    windowEnd=windowEnd,
                    savePath=Path(savePath, 'wb_' + str(windowStart) + '_wa_' + str(windowEnd) + '.json')
                )
            else:
                data = DataUtils.readData(
                    dirPath=dataPath,
                    idColumns=idColumns,
                    targetColumn=targetColumn,
                    measurementDateColumn=measurementDateColumn,
                    anchorDateColumn=anchorDateColumn,
                    windowStart=windowStart,
                    windowEnd=windowEnd,
                    )
                X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y, idsDf = data
                log.info('Unique Persons: ' + str(len(idsDf.person_id.unique())))
                log.info('Unique Visits: ' + str(idsDf.drop_duplicates().shape[0]))

                xgb = XGBClassifier(use_label_encoder=False)
                xgbScores = cross_validate(xgb, X, y, cv=5, scoring=['accuracy', 'balanced_accuracy',  'average_precision', 'f1', 'roc_auc'])
                xgbScores['test_mccf1_score'] = cross_validate(xgb, X, y, cv=5, scoring = make_scorer(MlUtils.calculateMccF1, greater_is_better=True))['test_score']

                log.info('xgbScores: ' + str(xgbScores))

                if not os.path.exists(savePath):
                    log.info('Creating directory: ' + str(savePath))
                    os.makedirs(savePath)

                log.info('saving the predictions')

                DataUtils.saveCvScores(xgbScores, Path(savePath), 'wb_' + str(windowStart) + '_wa_' + str(windowEnd) + '.json')


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

    parser.add_argument('-wb', '--window_before_list', nargs='*', type=int,
                        help='Number of days or data to include before time-zero.')

    parser.add_argument('-wa', '--window_after_list', nargs='*', type=int,
                        help='Number of days or data to include after time-zero.')

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
    log.info('args.window_before_list: ' + str(args.window_before_list))
    log.info('args.window_after_list: ' + str(args.window_after_list))
    log.info('args.ensemble: ' + str(args.ensemble))
    log.info('args.save_path: ' + str(args.save_path[0]))

    run(
        dataPath=args.data_file[0],
        idColumns=args.id_columns,
        targetColumn=args.target_column[0],
        measurementDateColumn=args.measurement_date_column[0],
        anchorDateColumn=args.anchor_date_column[0],
        windowStartList=args.window_before_list,
        windowEndList=args.window_after_list,
        ensemble=args.ensemble,
        savePath=args.save_path[0]
    )
