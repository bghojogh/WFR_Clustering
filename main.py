from utils.wfr import WFR
from utils.toy_datasets import make_two_spirals
import yaml, os
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score


def main(config: dict):
    # make toy dataset (optional):
    if config['make_toy_data']:
        X_train, y_train_true = make_two_spirals(n_points=500*2, noise=0.5, factor=0.75)
        X_test, y_test_true = make_two_spirals(n_points=50*2, noise=0.55, factor=0.75)
        os.makedirs(Path(config['traininig_data_path']).parent, exist_ok=True)
        os.makedirs(Path(config['test_data_path']).parent, exist_ok=True)
        pd.DataFrame(X_train).to_csv(config['traininig_data_path'], index=False)
        pd.DataFrame(X_test).to_csv(config['test_data_path'], index=False)
    else:
        y_train_true, y_test_true = None, None

    # load data:
    X_train = pd.read_csv(config['traininig_data_path']).to_numpy()
    if config['test_data_path'] is not None:
        X_test = pd.read_csv(config['test_data_path']).to_numpy()
    else:
        X_test = None

    # make an instance object of the class:
    wfr = WFR(resemblance_measure=config['resemblance_measure'],
            resemblance_threshold=config['resemblance_threshold'],
            resemblance_threshold_grid_search_step=config['resemblance_threshold_grid_search_step'],
            mark_outliers_as_minus_one=config['mark_outliers_as_minus_one'],
            knn_k=config['knn_k'],
            knn_sklearn_algorithm=config['knn_sklearn_algorithm'],
            knn_approx_method=None,
            kernel=config['kernel'])
    
    # training: 
    y_pred_training = wfr.fit_predict(X=X_train)
    print("Predicted labels of training: ", y_pred_training)
    if y_train_true is not None:
        print("True labels of training: ", y_train_true)
        accuracy_train = accuracy_score(y_train_true, y_pred_training)
        f1_train = f1_score(y_train_true, y_pred_training)
        print("Training accuracy:", accuracy_train)
        print("Training F1-score:", f1_train)
    
    # test (out of sample):
    if X_test is not None:
        y_pred_test = wfr.predict(X=X_test)
        print("Predicted labels of test: ", y_pred_test)
        if y_test_true is not None:
            print("True labels of test: ", y_test_true)
            accuracy_test = accuracy_score(y_test_true, y_pred_test)
            f1_test = f1_score(y_test_true, y_pred_test)
            print("Test accuracy:", accuracy_test)
            print("Test F1-score:", f1_test)


if __name__ == "__main__":
    with open('./config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)