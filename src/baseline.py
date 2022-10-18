import copy
import pickle
import time

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA

import loader
from config.config import TRAFFIC_COL_INDEX
from config.hparams import (
    ARIMA_ORDER,
    PREDICTION_HORIZON,
    RANDOM_SEED,
    SVR_KERNEL,
)
from config.storage import X_TEST, X_TRAIN, Y_TEST, Y_TRAIN
from utils.helper import fix_seed, validate


def get_allcars_xy(X, y):
    dataset = loader.create_dataset(X, y, space_window=None)

    X_, y_ = dataset[:]
    # specify `allCars` column
    X_ = X_[TRAFFIC_COL_INDEX].numpy()
    # next step only
    y_ = y_[:, 0].numpy()

    return X_, y_


def test_ha(X, prediction_horizon=PREDICTION_HORIZON):
    y_pred = np.zeros((X.shape[0], prediction_horizon))

    for i in range(prediction_horizon):
        y_pred[:, i] = X.mean(axis=1)

        X[:, :-1] = X[:, 1:]
        X[:, -1] = y_pred[:, i]

    return y_pred


def test_periodic_ha(X, prediction_horizon=PREDICTION_HORIZON):
    N, T = X.shape
    y_pred = np.zeros((N, prediction_horizon))

    for i in range(prediction_horizon):
        periodic_ind = np.arange(i, T, prediction_horizon)
        y_pred[:, i] = X[:, periodic_ind].mean(axis=1)

    return y_pred


def test_arima(X, prediction_horizon=PREDICTION_HORIZON):
    start = time.time()
    predicted = []

    for i in range(X.shape[0]):
        model = ARIMA(X[i], order=ARIMA_ORDER)
        model_fitted = model.fit()

        y_pred = model_fitted.forecast(steps=prediction_horizon)
        predicted.append(y_pred)

        if (i + 1) % 500 == 0:
            print(f"learned {i+1} samples: {time.time() - start :.3f} [sec]")

    print(f"Prediction Time: {time.time() - start :.3f} [sec]")

    predicted = np.stack(predicted, axis=0)
    return predicted


def test_classical_ml(model):
    def predict(X, prediction_horizon=PREDICTION_HORIZON):
        start = time.time()
        predicted = []

        for i in range(prediction_horizon):
            start_in_step = time.time()
            y_pred = model.predict(X)
            predicted.append(y_pred)

            X[:, :-1] = X[:, 1:]
            X[:, -1] = y_pred
            print(
                f"{i+1} step ahead: {time.time() - start_in_step :.3f} [sec]"
            )

        print(f"Prediction Time: {time.time() - start :.3f} [sec]")

        predicted = np.stack(predicted, axis=-1)
        return predicted

    return predict


def test_baseline(
    model_name,
    X_train,
    X_test,
    y_train,
    y_test,
    kernel=SVR_KERNEL,
    random_state=RANDOM_SEED,
):
    fix_seed(random_state)

    X_train = copy.deepcopy(X_train)
    X_test = copy.deepcopy(X_test)

    if model_name == "HA":
        f_predict = test_ha
    elif model_name == "PeriodicHA":
        f_predict = test_periodic_ha
    elif model_name == "ARIMA":
        f_predict = test_arima
    elif model_name in {"SVR", "RF"}:
        if model_name == "SVR":
            model = SVR(kernel=kernel)
        else:
            model = RandomForestRegressor()

        start = time.time()
        print(f"Training {model_name}...")
        model.fit(X_train, y_train[:, 0])
        print(f"Training Time: {time.time() - start} [sec]")

        f_predict = test_classical_ml(model)

    predicted_train = f_predict(X_train)
    train_mae, train_rmse = validate(predicted_train, y_train)
    print(
        "-" * 20,
        f"Train Loss: MAE = {train_mae :.3f}, RMSE = {train_rmse :.3f}",
        "-" * 20,
    )

    predicted_test = f_predict(X_test)
    test_mae, test_rmse = validate(predicted_test, y_test)
    print(
        "-" * 20,
        f"Test Loss: MAE = {test_mae :.3f}, RMSE = {test_rmse :.3f}",
        "-" * 20,
    )

    result = {
        "pred_train": predicted_train,
        "pred_test": predicted_test,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
    }
    return result


def main(args):
    X_train = torch.load(args.X_train)
    X_test = torch.load(args.X_test)
    y_train = torch.load(args.y_train)
    y_test = torch.load(args.y_test)

    X_train, y_train = get_allcars_xy(X_train, y_train)
    X_test, y_test = get_allcars_xy(X_test, y_test)

    result = test_baseline(
        args.model,
        X_train,
        X_test,
        y_train,
        y_test,
        kernel=args.kernel,
        random_state=args.random_state,
    )

    if args.result:
        with open(args.result, "wb") as f:
            pickle.dump(result, f)
        print(f"saved result to {args.result}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        help="baseline model",
        choices=["HA", "PeriodicHA", "ARIMA", "SVR", "RF"],
    )
    parser.add_argument("--result", help="file to store result")
    parser.add_argument(
        "--X_train", help="pickle file of features for train", default=X_TRAIN
    )
    parser.add_argument(
        "--X_test", help="pickle file of features for test", default=X_TEST
    )
    parser.add_argument(
        "--y_train", help="pickle file of labels for train", default=Y_TRAIN
    )
    parser.add_argument(
        "--y_test", help="pickle file of labels for test", default=Y_TEST
    )
    parser.add_argument(
        "--kernel",
        help="SVM kernel",
        choices=["rbf", "linear", "poly"],
        default=SVR_KERNEL,
    )
    parser.add_argument(
        "-r",
        "--random_state",
        help="seed of random number generator",
        type=int,
        default=RANDOM_SEED,
    )

    args = parser.parse_args()

    main(args)
