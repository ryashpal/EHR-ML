from sklearn.model_selection import train_test_split
import pandas as pd


def split_random(dataDf, anchorDateColumn, measurementDateColumn, windowStart, windowEnd, testSize, shuffle, seed):

    dataDf[anchorDateColumn] = pd.to_datetime(dataDf[anchorDateColumn], format='%d/%m/%Y')
    dataDf[measurementDateColumn] = pd.to_datetime(dataDf[measurementDateColumn], format='%d/%m/%Y')
    dataDf = dataDf[(dataDf[measurementDateColumn] >= (dataDf[anchorDateColumn] - pd.Timedelta(days=windowStart))) & (dataDf[measurementDateColumn] <= (dataDf[anchorDateColumn] + pd.Timedelta(days=windowEnd)))]
    xTestDataDf, xTrainDataDf = train_test_split(dataDf, test_size=testSize, shuffle=shuffle, seed=seed)
    return xTestDataDf, xTrainDataDf
