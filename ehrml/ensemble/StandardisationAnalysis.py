import os
import logging

log = logging.getLogger("EHR-ML")

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from ehrml.utils import DataUtils
from ehrml.utils import MlUtils


def run(dataPath, idColumns, targetColumn, measurementDateColumn, anchorDateColumn, windowStart, windowEnd, savePath):

    dataDf = pd.read_csv(dataPath)
    dataDf.to_csv(savePath + '/data_matrix_original.csv', index=False)

    data_cols = dataDf.columns[(dataDf.columns.str.startswith('vitals') | dataDf.columns.str.startswith('labs'))]
    non_data_cols = dataDf.columns[~dataDf.columns.isin(data_cols)]

    standardScaler = StandardScaler()
    standardScaler.fit(dataDf[data_cols])
    standardisedData = standardScaler.transform(dataDf[data_cols])

    standardisedDf = pd.concat([dataDf[non_data_cols], pd.DataFrame(standardisedData, columns=data_cols)], axis=1)

    standardisedDf.to_csv(savePath + '/data_matrix_standardised.csv', index=False)

    minmaxScaler = MinMaxScaler()
    minmaxScaler.fit(dataDf[data_cols])
    scaledData = minmaxScaler.transform(dataDf[data_cols])

    scaledDf = pd.concat([dataDf[non_data_cols], pd.DataFrame(scaledData, columns=data_cols)], axis=1)

    scaledDf.to_csv(savePath + '/data_matrix_scaled.csv', index=False)

    for file in ['data_matrix_original.csv', 'data_matrix_standardised.csv', 'data_matrix_scaled.csv']:

        data = DataUtils.readData(
            dirPath=savePath + file,
            idColumns=idColumns,
            targetColumn=targetColumn,
            measurementDateColumn=measurementDateColumn,
            anchorDateColumn=anchorDateColumn,
            windowStart=windowStart,
            windowEnd=windowEnd,
            )
        X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y, idsDf = data

        xgbEnsembleScores = MlUtils.evaluateEnsembleXGBoostModel(X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y[args.target_column[0]])

        dataPath = Path(savePath).parent
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        savePath = Path(savePath)
        DataUtils.saveCvScores(xgbEnsembleScores, savePath, 'wb_' + windowStart + '_wa_' + windowEnd + file.split('.')[0].split('_')[2] + '.json')


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

    parser.add_argument('-wb', '--window_before', nargs=1, type=int, default=[0],
                        help='Number of days or data to include before time-zero. By default: [window_before=0]')

    parser.add_argument('-wa', '--window_after', nargs=1, type=int, default=[3],
                        help='Number of days or data to include after time-zero. By default: [window_after=3]')

    parser.add_argument('-sp', '--save_dir', nargs=1, default='./results.json',
                        help='Directory to save the results')

    args = parser.parse_args()

    log.info('args.data_file: ' + str(args.data_file[0]))
    log.info('args.target_column: ' + str(args.target_column[0]))
    log.info('args.id_columns: ' + str(args.id_columns))
    log.info('args.measurement_date_column: ' + str(args.measurement_date_column[0]))
    log.info('args.anchor_date_column: ' + str(args.anchor_date_column[0]))
    log.info('args.window_before: ' + str(args.window_before[0]))
    log.info('args.window_after: ' + str(args.window_after[0]))
    log.info('args.save_dir: ' + str(args.save_dir[0]))

# THIS UTILITY IS NOT TESTED - DEVELOPMENT STILL IN PROGRESS
    # run(
    #     dataPath=args.data_file[0],
    #     idColumns=args.id_columns,
    #     targetColumn=args.target_column[0],
    #     anchorDateColumn=args.anchor_date_column[0],
    #     measurementDateColumn=args.measurement_date_column[0],
    #     windowStart=args.window_before[0],
    #     windowEnd=args.window_after[0],
    #     savePath=args.save_dir[0]
    # )
