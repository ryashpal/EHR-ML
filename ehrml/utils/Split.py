from sklearn.model_selection import train_test_split
import pandas as pd


def split_random(dataDf, idColumns, anchorDateColumn, measurementDateColumn, anchorDateColumnFormat='%d/%m/%Y', measurementDateColumnFormat='%d/%m/%Y', windowStart=0, windowEnd=0, testSize=0.8, shuffle=True, seed=42):

    dataDf[anchorDateColumn] = pd.to_datetime(dataDf[anchorDateColumn], format=anchorDateColumnFormat)
    dataDf[measurementDateColumn] = pd.to_datetime(dataDf[measurementDateColumn], format=measurementDateColumnFormat)
    dataDf = dataDf[(dataDf[measurementDateColumn] >= (dataDf[anchorDateColumn] - pd.Timedelta(days=windowStart))) & (dataDf[measurementDateColumn] <= (dataDf[anchorDateColumn] + pd.Timedelta(days=windowEnd)))]
    idsDf = dataDf[idColumns].drop_duplicates()
    testIdsDf, trainIdsDf = train_test_split(idsDf, test_size=testSize, shuffle=shuffle, random_state=seed)
    testDataDf = dataDf.merge(testIdsDf, how='inner', on=idColumns)
    trainDataDf = dataDf.merge(trainIdsDf, how='inner', on=idColumns)
    return testDataDf, trainDataDf
