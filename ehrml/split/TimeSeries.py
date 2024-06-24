def run(data_file, n_splits, gap, save_path):

    import pandas as pd

    from ehrml.utils import Split

    log.info("Reading the data from the file: " + str(data_file))

    datamatrixDf = pd.read_csv(data_file)

    log.info("Splitting the data")

    data_splits = Split.split_timeseries(
        data=datamatrixDf,
        n_splits=n_splits,
        gap=gap,
    )

    log.info("Saving the splits to the directory: " + str(save_path))

    i = 0
    for fold_data in data_splits:
        commonDf = fold_data['test'][fold_data['test'].person_id.isin(fold_data['train'].person_id)]
        testDf = fold_data['test'][~fold_data['test'].person_id.isin(fold_data['train'].person_id)]
        trainDf = pd.concat([fold_data['train'], commonDf], ignore_index=True)
        trainDf.to_csv(save_path + '/timeseries_split_' + str(i + 1) + '_train.csv', index=False)
        testDf.to_csv(save_path + '/timeseries_split_' + str(i + 1) + '_test.csv', index=False)
        i += 1


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

    parser.add_argument('data_file', nargs=1, default='data.csv',
                        help='Path of the data file in csv format')

    parser.add_argument('-s', '--n_splits', nargs=1, type=int, default=5,
                        help='Number of splits to perform. By default: [n_splits=5]')

    parser.add_argument('-g', '--gap', nargs=1, type=int, default=0,
                        help='Gap between the training and test set. By default: [gap=0]')

    parser.add_argument('-sp', '--save_path', nargs=1, default='./preds.csv',
                        help='File path to save the model predictions')

    args = parser.parse_args()

    log.info('args.data_file: ' + str(args.data_file[0]))
    log.info('args.n_splits: ' + str(args.n_splits[0]))
    log.info('args.gap: ' + str(args.gap[0]))
    log.info('args.save_path: ' + str(args.save_path[0]))

    run(data_file=args.data_file[0], n_splits=args.n_splits[0], gap=args.gap[0], save_path=args.save_path[0])
