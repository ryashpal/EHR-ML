import numpy as np
import pandas as pd
import json

from pathlib import Path


def generateSummaryTables(resulFileList):
    data = []
    for resulFile in resulFileList:
        resulFilePath = Path(resulFile)
        with open(resulFilePath) as f:
            results = json.load(f)
            data.append([
                resulFilePath.name.split('.')[0].split('_')[1],
                resulFilePath.name.split('.')[0].split('_')[3],
                resulFile.split('/')[7],
                resulFile.split('/')[8],
                np.average(results['test_accuracy']),
                np.average(results['test_balanced_accuracy']),
                np.average(results['test_average_precision']),
                np.average(results['test_f1']),
                np.average(results['test_roc_auc']),
                np.average(results['test_mccf1_score'])
            ])
    df = pd.DataFrame(data, columns=['WB', 'WA', 'target', 'model', 'test_accuracy', 'test_balanced_accuracy', 'test_average_precision', 'test_f1', 'test_roc_auc', 'test_mccf1_score'])
    print(df)


if __name__ == '__main__':
    generateSummaryTables([
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_7_day/standalone/wb_365_wa_0.json',
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_7_day/standalone/wb_365_wa_1.json',
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_7_day/standalone/wb_365_wa_2.json',
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_7_day/standalone/wb_365_wa_3.json',
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_14_day/standalone/wb_365_wa_0.json',
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_14_day/standalone/wb_365_wa_1.json',
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_14_day/standalone/wb_365_wa_2.json',
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_14_day/standalone/wb_365_wa_3.json',
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_30_day/standalone/wb_365_wa_0.json',
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_30_day/standalone/wb_365_wa_1.json',
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_30_day/standalone/wb_365_wa_2.json',
        # '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_30_day/standalone/wb_365_wa_3.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_7_day/ensemble/wb_365_wa_0.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_7_day/ensemble/wb_365_wa_1.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_7_day/ensemble/wb_365_wa_2.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_7_day/ensemble/wb_365_wa_3.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_14_day/ensemble/wb_365_wa_0.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_14_day/ensemble/wb_365_wa_1.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_14_day/ensemble/wb_365_wa_2.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_14_day/ensemble/wb_365_wa_3.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_30_day/ensemble/wb_365_wa_0.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_30_day/ensemble/wb_365_wa_1.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_30_day/ensemble/wb_365_wa_2.json',
        '/home/ehrqcadmin/workspace/ehr_data/blood_pos_cohort_20240119/analysis/death_30_day/ensemble/wb_365_wa_3.json',
    ])
