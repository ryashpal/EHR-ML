import logging

log = logging.getLogger("EHR-ML")

import warnings

warnings.filterwarnings("ignore")

import pandas as pd


def generate(admissionsFilePath, admittimeColumn, dischargetimeColumn, withinDays, savePath):
        log.info('Reading data from file: ' + str(admissionsFilePath))
        admissionsDf = pd.read_csv(admissionsFilePath)
        log.info('Calculating mortality')
        admissionsDf['LOS_LESS_THAN_' + str(withinDays) + '_DAYS'] = pd.to_datetime(admissionsDf[dischargetimeColumn], format='%Y-%m-%d %H:%M:%S.%f') < (pd.to_datetime(admissionsDf[admittimeColumn], format='%Y-%m-%d %H:%M:%S.%f') + pd.DateOffset(days=withinDays))
        log.info('saving the output')
        admissionsDf.to_csv(savePath, index=False)


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
    log.info('args.admission_time_column: ' + str(args.admission_time_column[0]))
    log.info('args.discharge_time_column: ' + str(args.discharge_time_column[0]))
    log.info('args.within_days: ' + str(args.within_days[0]))
    log.info('args.save_path: ' + str(args.save_path[0]))

    generate(
            admissionsFilePath=args.admission_file[0],
            admittimeColumn=args.admission_time_column[0],
            dischargetimeColumn=args.discharge_time_column[0],
            withinDays=args.within_days[0],
            savePath=args.save_path[0]
            )

