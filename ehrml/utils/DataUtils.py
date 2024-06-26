import logging

log = logging.getLogger("EHR-ML")


def filterWindow(dataDf, anchorDateColumn, measurementDateColumn, windowStart, windowEnd):

    import pandas as pd

    dataDf[anchorDateColumn] = pd.to_datetime(dataDf[anchorDateColumn], format='%Y-%m-%d')
    dataDf[measurementDateColumn] = pd.to_datetime(dataDf[measurementDateColumn], format='%Y-%m-%d')
    dataDf = dataDf[(dataDf[measurementDateColumn] >= (dataDf[anchorDateColumn] - pd.Timedelta(days=windowStart))) & (dataDf[measurementDateColumn] <= (dataDf[anchorDateColumn] + pd.Timedelta(days=windowEnd)))]

    return dataDf


def readData(dirPath, idColumns=['id'], targetColumn='target', anchorDateColumn='visit_start_date_adm', measurementDateColumn='measurement_date', windowStart=0, windowEnd=3):

    import pandas as pd

    dataDf = pd.read_csv(dirPath)

    dataDf = filterWindow(
        dataDf=dataDf,
        anchorDateColumn=anchorDateColumn,
        measurementDateColumn=measurementDateColumn,
        windowStart=windowStart,
        windowEnd=windowEnd
        )

    XVitalsAvg = dataDf[idColumns + [measurementDateColumn] + [col for col in dataDf.columns if (col.startswith('vitals_') and col.endswith('_avg'))]]
    XVitalsMin = dataDf[idColumns + [measurementDateColumn] + [col for col in dataDf.columns if (col.startswith('vitals_') and col.endswith('_min'))]]
    XVitalsMax = dataDf[idColumns + [measurementDateColumn] + [col for col in dataDf.columns if (col.startswith('vitals_') and col.endswith('_max'))]]
    XVitalsFirst = dataDf[idColumns + [measurementDateColumn] + [col for col in dataDf.columns if (col.startswith('vitals_') and col.endswith('_first'))]]
    XVitalsLast = dataDf[idColumns + [measurementDateColumn] + [col for col in dataDf.columns if (col.startswith('vitals_') and col.endswith('_last'))]]
    XLabsAvg = dataDf[idColumns + [measurementDateColumn] + [col for col in dataDf.columns if (col.startswith('labs_') and col.endswith('_avg'))]]
    XLabsMin = dataDf[idColumns + [measurementDateColumn] + [col for col in dataDf.columns if (col.startswith('labs_') and col.endswith('_min'))]]
    XLabsMax = dataDf[idColumns + [measurementDateColumn] + [col for col in dataDf.columns if (col.startswith('labs_') and col.endswith('_max'))]]
    XLabsFirst = dataDf[idColumns + [measurementDateColumn] + [col for col in dataDf.columns if (col.startswith('labs_') and col.endswith('_first'))]]
    XLabsLast = dataDf[idColumns + [measurementDateColumn] + [col for col in dataDf.columns if (col.startswith('labs_') and col.endswith('_last'))]]

    y = dataDf[idColumns + [targetColumn]]

    XVitalsAvgAgg = XVitalsAvg.groupby(idColumns).mean().reset_index().drop(columns=measurementDateColumn)
    XVitalsMinAgg = XVitalsMin.groupby(idColumns).min().reset_index().drop(columns=measurementDateColumn)
    XVitalsMaxAgg = XVitalsMax.groupby(idColumns).max().reset_index().drop(columns=measurementDateColumn)
    XVitalsFirstAgg = XVitalsFirst.sort_values(idColumns + [measurementDateColumn], ascending=True).groupby(idColumns).first().reset_index().drop(columns=measurementDateColumn)
    XVitalsLastAgg = XVitalsLast.sort_values(idColumns + [measurementDateColumn], ascending=False).groupby(idColumns).first().reset_index().drop(columns=measurementDateColumn)
    XLabsAvgAgg = XLabsAvg.groupby(idColumns).mean().reset_index().drop(columns=measurementDateColumn)
    XLabsMinAgg = XLabsMin.groupby(idColumns).min().reset_index().drop(columns=measurementDateColumn)
    XLabsMaxAgg = XLabsMax.groupby(idColumns).max().reset_index().drop(columns=measurementDateColumn)
    XLabsFirstAgg = XLabsFirst.sort_values(idColumns + [measurementDateColumn], ascending=True).groupby(idColumns).first().reset_index().drop(columns=measurementDateColumn)
    XLabsLastAgg = XLabsLast.sort_values(idColumns + [measurementDateColumn], ascending=False).groupby(idColumns).first().reset_index().drop(columns=measurementDateColumn)
    yAgg = y.groupby(idColumns).first().reset_index()

    X = XVitalsAvgAgg\
        .merge(XVitalsMinAgg, on=idColumns, how='outer')\
        .merge(XVitalsMaxAgg, on=idColumns, how='outer')\
        .merge(XVitalsFirstAgg, on=idColumns, how='outer')\
        .merge(XVitalsLastAgg, on=idColumns, how='outer')\
        .merge(XLabsAvgAgg, on=idColumns, how='outer')\
        .merge(XLabsMinAgg, on=idColumns, how='outer')\
        .merge(XLabsMaxAgg, on=idColumns, how='outer')\
        .merge(XLabsFirstAgg, on=idColumns, how='outer')\
        .merge(XLabsLastAgg, on=idColumns, how='outer')

    XVitalsAvgAgg = XVitalsAvgAgg.drop(columns=idColumns)
    XVitalsMinAgg = XVitalsMinAgg.drop(columns=idColumns)
    XVitalsMaxAgg = XVitalsMaxAgg.drop(columns=idColumns)
    XVitalsFirstAgg = XVitalsFirstAgg.drop(columns=idColumns)
    XVitalsLastAgg = XVitalsLastAgg.drop(columns=idColumns)
    XLabsAvgAgg = XLabsAvgAgg.drop(columns=idColumns)
    XLabsMinAgg = XLabsMinAgg.drop(columns=idColumns)
    XLabsMaxAgg = XLabsMaxAgg.drop(columns=idColumns)
    XLabsFirstAgg = XLabsFirstAgg.drop(columns=idColumns)
    XLabsLastAgg = XLabsLastAgg.drop(columns=idColumns)
    XAgg = X.drop(columns=idColumns)
    idsDf =yAgg[idColumns]
    yAgg = yAgg.drop(columns=idColumns)

    return XAgg, XVitalsAvgAgg, XVitalsMinAgg, XVitalsMaxAgg, XVitalsFirstAgg, XVitalsLastAgg, XLabsAvgAgg, XLabsMinAgg, XLabsMaxAgg, XLabsFirstAgg, XLabsLastAgg, yAgg, idsDf

def getDataDict(dirPath, idColumns, targetColumn, measurementDateColumn, anchorDateColumn, windowStart, windowEnd):

    from sklearn.model_selection import train_test_split

    data = readData(
        dirPath=dirPath,
        idColumns=idColumns,
        targetColumn=targetColumn,
        measurementDateColumn=measurementDateColumn,
        anchorDateColumn=anchorDateColumn,
        windowStart=windowStart,
        windowEnd=windowEnd,
        )
    X, XVitalsAvg, XVitalsMin, XVitalsMax, XVitalsFirst, XVitalsLast, XLabsAvg, XLabsMin, XLabsMax, XLabsFirst, XLabsLast, y, idsDf = data
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
    dataDict = {
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
    return dataDict


def saveCvScores(scores_dict, dirPath, fileName):

    import os
    import json
    from pathlib import Path

    log.info('Saving the results!!')

    if not os.path.exists(dirPath):
        log.info('Creating directory: ' + str(dirPath))
        os.makedirs(dirPath)

    cvScoresPath = Path(dirPath, fileName)

    for key, value in scores_dict.items():
        scores_dict[key] = value.tolist()

    log.info('Save path:' + str(cvScoresPath))

    with open(cvScoresPath, 'w') as fp:
        json.dump(scores_dict, fp, indent=4)


def saveModels(model, dirPath, fileName):

    import os
    import pickle
    from pathlib import Path

    log.info('Saving the model!!')

    if not os.path.exists(dirPath):
        log.info('Creating directory: ' + str(dirPath))
        os.makedirs(dirPath)

    savePath = Path(dirPath, fileName)

    log.info('Save path:' + str(savePath))

    with open(savePath, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generateLabels(dataDf, anchorColumn, targetColumn, timeDelta):
    import pandas as pd
    return ((dataDf[targetColumn] > dataDf[anchorColumn]) & (dataDf[targetColumn] < (dataDf[anchorColumn] + pd.to_timedelta(timeDelta,'d')))).apply(lambda x: 1 if x else 0)

