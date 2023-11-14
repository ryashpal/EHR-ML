import logging

log = logging.getLogger("EHR-ML")


def filterWindow(dataDf, anchorTimeColumn, measurementTimeColumn, windowStart, windowEnd):

    import pandas as pd

    dataDf[anchorTimeColumn] = pd.to_datetime(dataDf[anchorTimeColumn])
    dataDf[measurementTimeColumn] = pd.to_datetime(dataDf[measurementTimeColumn])
    dataDf = dataDf[(dataDf[measurementTimeColumn] >= (dataDf[anchorTimeColumn] - pd.Timedelta(days=windowStart))) & (dataDf[measurementTimeColumn] <= (dataDf[anchorTimeColumn] + pd.Timedelta(days=windowEnd)))]

    return dataDf


def readEicuData(dirPath, windowStart=0, windowEnd=3, targetColumn='death_adm', anchorTimeColumn='visit_start_date_adm', measurementTimeColumn='measurement_date'):

    import pandas as pd

    dataDf = pd.read_csv(dirPath)

    dataDf = filterWindow(
        dataDf=dataDf,
        anchorTimeColumn=anchorTimeColumn,
        measurementTimeColumn=measurementTimeColumn,
        windowStart=windowStart,
        windowEnd=windowEnd
        )

    vitalsCols = ['systemic_mean', 'systemic_diastolic', 'systemic_systolic', 'respiration', 'heartrate', 'sao2']
    labsCols = ['Sodium level', 'Blood urea nitrogen', 'Creatinine level', 'Potassium level', 'Chloride', 'Hematocrit', 'Haemoglobin estimation', 'Platelet count', 'Red blood cell count', 'Calcium level', 'MCV - Mean corpuscular volume', 'MCHC - Mean corpuscular haemoglobin concentration', 'MCH - Mean corpuscular haemoglobin', 'White blood cell count', 'Red blood cell distribution width', 'Glucose level', 'Bicarbonate level', 'Anion gap']

    XVitalsAvg = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [vitalCol + '_avg' for vitalCol in vitalsCols if vitalCol + '_avg' in dataDf.columns]]
    XVitalsMin = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [vitalCol + '_min' for vitalCol in vitalsCols if vitalCol + '_min' in dataDf.columns]]
    XVitalsMax = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [vitalCol + '_max' for vitalCol in vitalsCols if vitalCol + '_max' in dataDf.columns]]
    XVitalsFirst = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [vitalCol + '_first' for vitalCol in vitalsCols if vitalCol + '_first' in dataDf.columns]]
    XVitalsLast = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [vitalCol + '_last' for vitalCol in vitalsCols if vitalCol + '_last' in dataDf.columns]]
    XLabsAvg = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [labsCol + '_avg' for labsCol in labsCols if labsCol + '_avg' in dataDf.columns]]
    XLabsMin = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [labsCol + '_min' for labsCol in labsCols if labsCol + '_min' in dataDf.columns]]
    XLabsMax = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [labsCol + '_max' for labsCol in labsCols if labsCol + '_max' in dataDf.columns]]
    XLabsFirst = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [labsCol + '_first' for labsCol in labsCols if labsCol + '_first' in dataDf.columns]]
    XLabsLast = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [labsCol + '_last' for labsCol in labsCols if labsCol + '_last' in dataDf.columns]]

    y = dataDf[['person_id', 'visit_occurrence_id'] + [targetColumn]]

    XVitalsAvgAgg = XVitalsAvg.groupby(['person_id', 'visit_occurrence_id']).mean().reset_index().drop(columns='measurement_date')
    XVitalsMinAgg = XVitalsMin.groupby(['person_id', 'visit_occurrence_id']).min().reset_index().drop(columns='measurement_date')
    XVitalsMaxAgg = XVitalsMax.groupby(['person_id', 'visit_occurrence_id']).max().reset_index().drop(columns='measurement_date')
    XVitalsFirstAgg = XVitalsFirst.sort_values(['person_id', 'visit_occurrence_id', 'measurement_date'], ascending=True).groupby(['person_id', 'visit_occurrence_id']).first().reset_index().drop(columns='measurement_date')
    XVitalsLastAgg = XVitalsLast.sort_values(['person_id', 'visit_occurrence_id', 'measurement_date'], ascending=False).groupby(['person_id', 'visit_occurrence_id']).first().reset_index().drop(columns='measurement_date')
    XLabsAvgAgg = XLabsAvg.groupby(['person_id', 'visit_occurrence_id']).mean().reset_index().drop(columns='measurement_date')
    XLabsMinAgg = XLabsMin.groupby(['person_id', 'visit_occurrence_id']).min().reset_index().drop(columns='measurement_date')
    XLabsMaxAgg = XLabsMax.groupby(['person_id', 'visit_occurrence_id']).max().reset_index().drop(columns='measurement_date')
    XLabsFirstAgg = XLabsFirst.sort_values(['person_id', 'visit_occurrence_id', 'measurement_date'], ascending=True).groupby(['person_id', 'visit_occurrence_id']).first().reset_index().drop(columns='measurement_date')
    XLabsLastAgg = XLabsLast.sort_values(['person_id', 'visit_occurrence_id', 'measurement_date'], ascending=False).groupby(['person_id', 'visit_occurrence_id']).first().reset_index().drop(columns='measurement_date')
    yAgg = y.groupby(['person_id', 'visit_occurrence_id']).first().reset_index()

    X = XVitalsAvgAgg\
        .merge(XVitalsMinAgg, on=['person_id', 'visit_occurrence_id'], how='outer')\
        .merge(XVitalsMaxAgg, on=['person_id', 'visit_occurrence_id'], how='outer')\
        .merge(XVitalsFirstAgg, on=['person_id', 'visit_occurrence_id'], how='outer')\
        .merge(XVitalsLastAgg, on=['person_id', 'visit_occurrence_id'], how='outer')\
        .merge(XLabsAvgAgg, on=['person_id', 'visit_occurrence_id'], how='outer')\
        .merge(XLabsMinAgg, on=['person_id', 'visit_occurrence_id'], how='outer')\
        .merge(XLabsMaxAgg, on=['person_id', 'visit_occurrence_id'], how='outer')\
        .merge(XLabsFirstAgg, on=['person_id', 'visit_occurrence_id'], how='outer')\
        .merge(XLabsLastAgg, on=['person_id', 'visit_occurrence_id'], how='outer')

    XVitalsAvgAgg = XVitalsAvgAgg.drop(columns=['person_id', 'visit_occurrence_id'])
    XVitalsMinAgg = XVitalsMinAgg.drop(columns=['person_id', 'visit_occurrence_id'])
    XVitalsMaxAgg = XVitalsMaxAgg.drop(columns=['person_id', 'visit_occurrence_id'])
    XVitalsFirstAgg = XVitalsFirstAgg.drop(columns=['person_id', 'visit_occurrence_id'])
    XVitalsLastAgg = XVitalsLastAgg.drop(columns=['person_id', 'visit_occurrence_id'])
    XLabsAvgAgg = XLabsAvgAgg.drop(columns=['person_id', 'visit_occurrence_id'])
    XLabsMinAgg = XLabsMinAgg.drop(columns=['person_id', 'visit_occurrence_id'])
    XLabsMaxAgg = XLabsMaxAgg.drop(columns=['person_id', 'visit_occurrence_id'])
    XLabsFirstAgg = XLabsFirstAgg.drop(columns=['person_id', 'visit_occurrence_id'])
    XLabsLastAgg = XLabsLastAgg.drop(columns=['person_id', 'visit_occurrence_id'])
    X = X.drop(columns=['person_id', 'visit_occurrence_id'])
    yAgg = yAgg.drop(columns=['person_id', 'visit_occurrence_id'])

    return X, XVitalsAvgAgg, XVitalsMinAgg, XVitalsMaxAgg, XVitalsFirstAgg, XVitalsLastAgg, XLabsAvgAgg, XLabsMinAgg, XLabsMaxAgg, XLabsFirstAgg, XLabsLastAgg, yAgg
