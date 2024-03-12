import logging

log = logging.getLogger("EHR-ML")

import warnings

warnings.filterwarnings("ignore")

import pandas as pd


def generate(icuFilePath, patientIdColumn, admissionIdColumn, icuIdColumn, icuAdmittimeColumn, anchortimeColumn, savePath):
        log.info('Reading data from file: ' + str(icuFilePath))
        icuDf = pd.read_csv(icuFilePath)
        log.info('Calculating readmission')
        icuDf['ICU_ADMISSION'] = ((icuDf[icuIdColumn].notna()) & (icuDf[icuAdmittimeColumn].notna()) & (icuDf[anchortimeColumn] < icuDf[icuAdmittimeColumn]))
        log.info('Dropping Duplicate IDs')
        icuDf = icuDf[[patientIdColumn, admissionIdColumn, 'ICU_ADMISSION']].drop_duplicates()
        log.info('saving the output')
        icuDf.to_csv(savePath, index=False)


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

    parser.add_argument('-ic', '--icu_id_column', nargs=1, default=['icu_id'],
                        help='Name of the column containing the ICU ID')

    parser.add_argument('-iac', '--icu_admittime_column', nargs=1, default=['icu_admittime'],
                        help='Name of the column containing the ICU admission time')

    parser.add_argument('-atc', '--anchortime_column', nargs=1, default=['anchortime'],
                        help='Name of the column containing the anchor time')

    parser.add_argument('-sp', '--save_path', nargs=1, default='./',
                        help='Path to save file with calculated readmission column')

    args = parser.parse_args()

    log.info('args.admission_file: ' + str(args.admission_file[0]))
    log.info('args.patient_id_column: ' + str(args.patient_id_column[0]))
    log.info('args.admission_id_column: ' + str(args.admission_id_column[0]))
    log.info('args.icu_id_column: ' + str(args.icu_id_column[0]))
    log.info('args.icu_admittime_column: ' + str(args.icu_admittime_column[0]))
    log.info('args.anchortime_column: ' + str(args.anchortime_column[0]))
    log.info('args.save_path: ' + str(args.save_path[0]))

    generate(
            icuFilePath=args.admission_file[0],
            patientIdColumn=args.patient_id_column[0],
            admissionIdColumn=args.admission_id_column[0],
            icuIdColumn=args.icu_id_column[0],
            icuAdmittimeColumn=args.icu_admittime_column[0],
            anchortimeColumn=args.anchortime_column[0],
            savePath=args.save_path[0]
            )

