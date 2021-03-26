import argparse
import os
import timeit
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import log_loss
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--workload', type=str, default='all',
                    help='Choose worload for SVM. Default all worloads')
parser.add_argument('--library', type=str, default='idp_sklearn',
                    choices=['sklearn', 'thunder', 'cuml', 'idp_sklearn'],
                    help='Choose library for SVM. Default idp_sklearn')

args = parser.parse_args()
arg_name_workload = args.workload
arg_name_library = args.library
times_worloads = []

if arg_name_library == 'idp_sklearn':
    from daal4py.sklearn import patch_sklearn
    patch_sklearn()
    from sklearn.svm import SVC
elif arg_name_library == 'sklearn':
    from sklearn.svm import SVC
elif arg_name_library == 'thunder':
    from thundersvm import SVC
elif arg_name_library == 'cuml':
    from cuml import SVC
    from cupy import unique
    import cudf


cache_size = 8*1024  # 8 GB
tol = 1e-3

workloads = {
    'a9a':               {'C': 500.0,  'kernel': 'rbf'},        # n_classes = 2
    'ijcnn':             {'C': 1000.0, 'kernel': 'linear'},     # n_classes = 2
    'sensit':            {'C': 500.0,  'kernel': 'linear'},     # n_classes = 3
    'connect':           {'C': 100.0,  'kernel': 'linear'},     # n_classes = 3
    'gisette':           {'C': 0.0015, 'kernel': 'linear'},     # n_classes = 2
    'mnist':             {'C': 100.0,  'kernel': 'linear'},     # n_classes = 10
    'klaverjas':         {'C': 1.0,    'kernel': 'rbf'},        # n_classes = 2
    'skin_segmentation': {'C': 1.0,    'kernel': 'rbf'},        # n_classes = 2
    'covertype':         {'C': 100.0,  'kernel': 'rbf'},        # n_classes = 7
    # 'creditcard':        {'C': 100.0,  'kernel': 'linear'},
    'codrnanorm':        {'C': 1000.0, 'kernel': 'linear'},     # n_classes = 2
}


def load_data(name_workload):
    root_dir = os.environ['DATASETSROOT']
    dataset_dir = os.path.join(root_dir, 'workloads', name_workload, 'dataset')
    x_train_path = os.path.join(
        dataset_dir, '{}_x_train.csv'.format(name_workload))
    x_test_path = os.path.join(
        dataset_dir, '{}_x_test.csv'.format(name_workload))
    y_train_path = os.path.join(
        dataset_dir, '{}_y_train.csv'.format(name_workload))
    y_test_path = os.path.join(
        dataset_dir, '{}_y_test.csv'.format(name_workload))
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    if arg_name_library == 'cuml':
        x_train = cudf.read_csv(x_train_path, header=None, dtype='double')
        x_test = cudf.read_csv(x_test_path, header=None, dtype='double')
        y_train = cudf.read_csv(y_train_path, header=None, dtype='double')
        y_test = cudf.read_csv(y_test_path, header=None, dtype='double')
    else:
        x_train = pd.read_csv(x_train_path, header=None, dtype='double')
        x_test = pd.read_csv(x_test_path, header=None, dtype='double')
        y_train = pd.read_csv(y_train_path, header=None, dtype='double')
        y_test = pd.read_csv(y_test_path, header=None, dtype='double')
    return x_train, x_test, y_train, y_test


def run_svm_proba_workload(workload_name, x_train, x_test, y_train, y_test, C=1.0, kernel='linear'):
    gamma = 1.0 / x_train.shape[1]
    # Create C-SVM classifier
    clf = SVC(C=C, kernel=kernel, max_iter=-1, cache_size=cache_size,
              tol=tol, gamma=gamma, probability=True)

    t0 = timeit.default_timer()
    clf.fit(x_train, y_train)
    t1 = timeit.default_timer()
    time_fit_train_run = t1 - t0

    t0 = timeit.default_timer()
    y_pred_train = clf.predict_proba(x_train)
    t1 = timeit.default_timer()
    time_predict_train_run = t1 - t0

    t0 = timeit.default_timer()
    y_pred = clf.predict_proba(x_test)
    t1 = timeit.default_timer()
    time_predict_test_run = t1 - t0

    n_classes = 0
    if arg_name_library == 'cuml':
        n_classes = len(unique(y_train.values))
        acc_train = log_loss(y_train.to_pandas().values, y_pred_train.to_pandas().values)
        acc_test = log_loss(y_test.to_pandas().values, y_pred.to_pandas().values)
    else:
        n_classes = len(np.unique(y_train))
        acc_train = log_loss(y_train, y_pred_train)
        acc_test = log_loss(y_test, y_pred)

    print('{}: n_samples:{}; n_features:{}; n_classes:{}; C:{}; kernel:{}'.format(
        workload_name, x_train.shape[0], x_train.shape[1], n_classes, C, kernel))
    print('Fit   [Train n_samples:{:6d}]: {:6.2f} sec'.format(
        x_train.shape[0], time_fit_train_run))
    print('Infer [Train n_samples:{:6d}]: {:6.2f} sec. log_loss_score: {:.3f}'.format(
        x_train.shape[0], time_predict_train_run, acc_train))
    print('Infer [Test  n_samples:{:6d}]: {:6.2f} sec. log_loss_score: {:.3f}'.format(
        x_test.shape[0], time_predict_test_run, acc_test))


for name_workload, params in workloads.items():
    if arg_name_workload in [name_workload, 'all']:
        x_train, x_test, y_train, y_test = load_data(name_workload)
        run_svm_proba_workload(name_workload, x_train, x_test, y_train, y_test,
                         C=params['C'], kernel=params['kernel'])
