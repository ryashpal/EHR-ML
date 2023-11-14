import logging

log = logging.getLogger("EHR-ML")

from ehrml.util import DataUtil


if __name__ == '__main__':

    import os

    import pandas as pd

    dataDf = pd.read_csv(os.environ['EICU_EHR_PIPELINE_BASE'] + '/data/final/data_matrix.csv')

    dataDf = DataUtil.filterWindow(
        dataDf=dataDf,
        anchorTimeColumn='visit_start_date_adm',
        measurementTimeColumn='measurement_date',
        windowStart=1,
        windowEnd=3
        )

    dropCols = ['person_id', 'visit_occurrence_id', 'measurement_date', 'visit_start_date_adm', 'death_adm']

    vitalsCols = ['systemic_mean', 'systemic_diastolic', 'systemic_systolic', 'respiration', 'heartrate', 'sao2']
    labsCols = ['Sodium level', 'Blood urea nitrogen', 'Creatinine level', 'Potassium level', 'Chloride', 'Hematocrit', 'Haemoglobin estimation', 'Platelet count', 'Red blood cell count', 'Calcium level', 'MCV - Mean corpuscular volume', 'MCHC - Mean corpuscular haemoglobin concentration', 'MCH - Mean corpuscular haemoglobin', 'White blood cell count', 'Red blood cell distribution width', 'Glucose level', 'Bicarbonate level', 'Anion gap']

    X = dataDf.drop(dropCols, axis = 1)
    XVitalsAvg = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [vitalCol + '_avg' for vitalCol in vitalsCols if vitalCol + '_avg' in dataDf.columns]]
    XVitalsMin = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [vitalCol + '_min' for vitalCol in vitalsCols if vitalCol + '_min' in dataDf.columns]]
    XVitalsMax = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [vitalCol + '_max' for vitalCol in vitalsCols if vitalCol + '_max' in dataDf.columns]]
    XVitalsFirst = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [vitalCol + '_first' for vitalCol in vitalsCols if vitalCol + '_first' in dataDf.columns]]
    XVitalsLast = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [vitalCol + '_last' for vitalCol in vitalsCols if vitalCol + '_last' in dataDf.columns]]
    XLabsAvg = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [labsCol + '_avg' for labsCol in labsCols if labsCol + '_avg' in dataDf.columns]]
    XLabsMax = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [labsCol + '_min' for labsCol in labsCols if labsCol + '_min' in dataDf.columns]]
    XLabsMin = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [labsCol + '_max' for labsCol in labsCols if labsCol + '_max' in dataDf.columns]]
    XLabsFirst = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [labsCol + '_first' for labsCol in labsCols if labsCol + '_first' in dataDf.columns]]
    XLabsLast = dataDf[['person_id', 'visit_occurrence_id', 'measurement_date'] + [labsCol + '_last' for labsCol in labsCols if labsCol + '_last' in dataDf.columns]]
    y = dataDf["death_adm"]

    print('X: ', X)
