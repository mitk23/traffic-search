import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from config.hparams import PREDICTION_HORIZON
from utils.helper import fix_seed


def test_baseline(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    prediction_horizon=PREDICTION_HORIZON,
):
    fix_seed()

    # train
    start = time.time()
    model.fit(X_train, y_train[:, 0])
    print(f"Train Time: {time.time() - start:.3f} [sec]")

    # test
    def test(X_test, y_test):
        start = time.time()
        predicted = []

        for i in range(prediction_horizon):
            start_in_step = time.time()
            y_pred = model.predict(X_test)
            predicted.append(y_pred)

            X_test[:, :-1] = X_test[:, 1:]
            X_test[:, -1] = y_pred
            print(f"{i+1} step ahead: {time.time() - start_in_step} [sec]")

        print(f"Test Time: {time.time() - start} [sec]")

        predicted = np.stack(predicted, axis=-1)

        mae_mat = np.abs(predicted - y_test)
        mae = mae_mat.mean()
        mape_mat = mae / (y_test + 1e-7)
        mape = mape_mat.mean()

        return predicted, mae, mape

    predicted_train, mae, mape = test(X_train, y_train)
    print(f"Train Loss: MAE = {mae :.3f}, MAPE = {mape :.3f}")
    predicted_test, mae, mape = test(X_test, y_test)
    print(f"Test Loss: MAE = {mae :.3f}, MAPE = {mape :.3f}")

    return predicted_train, predicted_test


def get_xy(dataset):
    X, y = dataset[:]
    # specify `cars` column
    X = X[-1].numpy()
    # next step only
    y = y[:, 0].numpy()
    return X, y


def main():
    train_X, train_y = get_xy()
    test_X, test_y = get_xy()

    predicted_train, predicted_test = test_baseline(
        SVR(kernel="rbf"), train_X, test_X, train_y, test_y
    )
    predicted_train, predicted_test = test_baseline(
        RandomForestRegressor(), train_X, test_X, train_y, test_y
    )


if __name__ == "__main__":
    main()
