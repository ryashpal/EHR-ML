import logging

log = logging.getLogger("EHR-ML")

import warnings

warnings.filterwarnings("ignore")

import pandas as pd


def generate(admissionsFilePath, patientIdColumn, admissionIdColumn, admittimeColumn, dischargetimeColumn, withinDays, savePath):
        log.info('Reading data from file: ' + str(admissionsFilePath))
        admissionsDf = pd.read_csv(admissionsFilePath)
        nextAdmissionIdColumn = 'NEXT_' + admissionIdColumn
        log.info('Obtaining next episode IDs')
        admissionsDf[nextAdmissionIdColumn] = admissionsDf.sort_values(by=[admittimeColumn], ascending=False).groupby([patientIdColumn])[admissionIdColumn].shift(1)
        log.info('Merging current and next rows')
        mergedDf = admissionsDf[[patientIdColumn, admissionIdColumn, nextAdmissionIdColumn, dischargetimeColumn]].add_suffix('_CUR').merge(
                admissionsDf[[patientIdColumn, admissionIdColumn, admittimeColumn]].add_suffix('_NEXT'),
                how='inner',
                left_on=[patientIdColumn + '_CUR', nextAdmissionIdColumn + '_CUR'],
                right_on=[patientIdColumn + '_NEXT', admissionIdColumn + '_NEXT']
                )
        log.info('Calculating readmission')
        mergedDf['READMISSION'] = (mergedDf[admittimeColumn + '_NEXT'] < (pd.to_datetime(mergedDf[dischargetimeColumn + '_CUR'], format='%Y-%m-%d %H:%M:%S.%f')) + pd.DateOffset(days=withinDays))
        log.info('Dropping Duplicate IDs')
        mergedDf = mergedDf.drop_duplicates(subset=[patientIdColumn + '_CUR', admissionIdColumn + '_CUR'])
        log.info('saving the output')
        mergedDf.to_csv(savePath, index=False)


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

    parser.add_argument('admission_file', nargs=1, default='admission.csv',
                        help='Path of the admission file in csv format')

    parser.add_argument('-pc', '--patient_id_column', nargs=1, default=['patient_id'],
                        help='Name of the column containing the patient ID')

    parser.add_argument('-ac', '--admission_id_column', nargs=1, default=['admission_id'],
                        help='Name of the column containing the admission ID')

    parser.add_argument('-atc', '--admission_time_column', nargs=1, default=['admittime'],
                        help='Name of the column containing the admission time')

    parser.add_argument('-dtc', '--discharge_time_column', nargs=1, default=['dischargetime'],
                        help='Name of the column containing the discharge time')

    parser.add_argument('-wd', '--within_days', nargs=1, type=int, default=30,
                        help='Number of days within which to be considered as readmission. By default: [window_before=30 days]')

    parser.add_argument('-sp', '--save_path', nargs=1, default='./',
                        help='Path to save file with calculated readmission column')

    args = parser.parse_args()

    log.info('args.admission_file: ' + str(args.admission_file[0]))
    log.info('args.patient_id_column: ' + str(args.patient_id_column[0]))
    log.info('args.admission_id_column: ' + str(args.admission_id_column[0]))
    log.info('args.admission_time_column: ' + str(args.admission_time_column[0]))
    log.info('args.discharge_time_column: ' + str(args.discharge_time_column[0]))
    log.info('args.within_days: ' + str(args.within_days[0]))
    log.info('args.save_path: ' + str(args.save_path[0]))

    generate(
            admissionsFilePath=args.admission_file[0],
            patientIdColumn=args.patient_id_column[0],
            admissionIdColumn=args.admission_id_column[0],
            admittimeColumn=args.admission_time_column[0],
            dischargetimeColumn=args.discharge_time_column[0],
            withinDays=args.within_days[0],
            savePath=args.save_path[0]
            )

