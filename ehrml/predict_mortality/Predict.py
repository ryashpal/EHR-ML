import logging

log = logging.getLogger("EHR-ML")

import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

from ehrml.utils import DataUtils
from ehrml.utils import MlUtils


def run(dirPath, idColumns, targetColumn, measurementDateColumn, windowStart, windowEnd, modelPath):
    data = DataUtils.readData(
        dirPath=dirPath,
        idColumns=idColumns,
        targetColumn=targetColumn,
        measurementDateColumn=measurementDateColumn,
        windowStart=windowStart,
        windowEnd=windowEnd,
        )
    X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y, idsDf = data

    preds = MlUtils.predictEnsembleXGBoostModel(X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y[args.target_column[0]], modelFilePath=modelPath)

    saveDf = idsDf
    saveDf['preds'] = preds
    log.info('saveDf: ' + str(saveDf))


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

    parser.add_argument('-ic', '--id_columns', nargs='*',
                        help='Name/s of the columns containing the the IDs')

    parser.add_argument('-tc', '--target_column', nargs=1, default=['target'],
                        help='Name of the column containing the target variable')

    parser.add_argument('-mdc', '--measurement_date_column', nargs=1, default=['measurement_date'],
                        help='Name of the column containing the measurement date')

    parser.add_argument('-wb', '--window_before', nargs=1, type=int, default=0,
                        help='Number of days or data to include before time-zero. By default: [window_before=0]')

    parser.add_argument('-wa', '--window_after', nargs=1, type=int, default=3,
                        help='Number of days or data to include after time-zero. By default: [window_after=3]')

    parser.add_argument('-mp', '--model_path', nargs=1, default='./model.pkl',
                        help='File containing the model')

    args = parser.parse_args()

    log.info('args.data_file: ' + str(args.data_file[0]))
    log.info('args.id_columns: ' + str(args.id_columns))
    log.info('args.target_column: ' + str(args.target_column[0]))
    log.info('args.measurement_date_column: ' + str(args.measurement_date_column[0]))
    log.info('args.window_before: ' + str(args.window_before[0]))
    log.info('args.window_after: ' + str(args.window_after[0]))
    log.info('args.model_path: ' + str(args.model_path[0]))

    run(
        dirPath=args.data_file[0],
        idColumns=args.id_columns,
        targetColumn=args.target_column[0],
        measurementDateColumn=args.measurement_date_column[0],
        windowStart=args.window_before[0],
        windowEnd=args.window_after[0],
        modelPath=args.model_path[0]
    )
