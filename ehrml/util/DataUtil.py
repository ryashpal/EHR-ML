import logging

log = logging.getLogger("EHR-ML")


def filterWindow(dataDf, anchorTimeColumn, measurementTimeColumn, windowStart, windowEnd):

    import pandas as pd

    dataDf[anchorTimeColumn] = pd.to_datetime(dataDf[anchorTimeColumn])
    dataDf[measurementTimeColumn] = pd.to_datetime(dataDf[measurementTimeColumn])
    dataDf = dataDf[(dataDf[measurementTimeColumn] >= (dataDf[anchorTimeColumn] - pd.Timedelta(days=windowStart))) & (dataDf[measurementTimeColumn] <= (dataDf[anchorTimeColumn] + pd.Timedelta(days=windowEnd)))]

    return dataDf
