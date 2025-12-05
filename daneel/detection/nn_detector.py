import numpy as np
import pandas as pd
from pathlib import Path

from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.utils import shuffle

import tensorflow as tf


class LightFluxProcessor:

    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize

    def fourier_transform(self, X):
        return np.abs(fft.fft(X, n=X.size))

    def process(self, df_train_x, df_test_x):

        if self.fourier:
            print("Applying Fourier...")
            shape_train = df_train_x.shape
            shape_test = df_test_x.shape

            df_train_x = df_train_x.apply(self.fourier_transform, axis=1)
            df_test_x  = df_test_x.apply(self.fourier_transform,  axis=1)

            X_train = np.zeros(shape_train)
            X_test  = np.zeros(shape_test)

            for i, row in enumerate(df_train_x):
                X_train[i] = row
            for i, row in enumerate(df_test_x):
                X_test[i] = row

            X_train = X_train[:, : X_train.shape[1] // 2]
            X_test  = X_test[:,  : X_test.shape[1] // 2]

        else:
            X_train = df_train_x.values
            X_test  = df_test_x.values

        if self.normalize:
            print("Normalizing...")
            X_train = normalize(X_train)
            X_test = normalize(X_test)

        if self.gaussian:
            print("Applying Gaussian Filter...")
            X_train = ndimage.gaussian_filter(X_train, sigma=0.02)
            X_test  = ndimage.gaussian_filter(X_test,  sigma=0.02)

        if self.standardize:
            print("Standardizing...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        print("Finished Processing!\n")
        return X_train, X_test

class NNDetector:

    def __init__(self, dataset_path, kernel_name=None, degree=None):
        self.dataset_path = dataset_path

    def load_data(self):
        train = f"{self.dataset_path}/exoTrain.csv"
        test  = f"{self.dataset_path}/exoTest.csv"
        df_train = pd.read_csv(train, encoding="ISO-8859-1")
        df_test  = pd.read_csv(test,  encoding="ISO-8859-1")
        return df_train, df_test

    def np_X_Y_from_df(self, df):
        df = shuffle(df)
        X = np.array(df.drop(["LABEL"], axis=1))
        Y_raw = np.array(df["LABEL"]).reshape((-1, 1))
        Y = (Y_raw == 2).astype(int)
        return X, Y

    def build_network(self, input_shape):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(input_shape),

            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),

            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def run_detection(self):

        print("Loading dataset...")
        df_train, df_test = self.load_data()

        df_train_x = df_train.drop("LABEL", axis=1)
        df_test_x  = df_test.drop("LABEL", axis=1)
        df_train_y = df_train["LABEL"]
        df_test_y  = df_test["LABEL"]

        processor = LightFluxProcessor(
            fourier=True, normalize=True,
            gaussian=True, standardize=True
        )

        X_train, X_test = processor.process(df_train_x, df_test_x)

        df_train_processed = pd.DataFrame(X_train).join(df_train_y)
        df_test_processed  = pd.DataFrame(X_test).join(df_test_y)

        X_train, Y_train = self.np_X_Y_from_df(df_train_processed)
        X_test,  Y_test  = self.np_X_Y_from_df(df_test_processed)

        print("Building neural network...")
        model = self.build_network(X_train.shape[1:])

        print("Training network...")
        model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0)

        print("Evaluating...")

        train_preds = (model.predict(X_train) > 0.25).astype(int)
        test_preds  = (model.predict(X_test)  > 0.25).astype(int)

        cm = confusion_matrix(Y_test, test_preds)
        acc = accuracy_score(Y_test, test_preds)
        prec = precision_score(Y_test, test_preds, zero_division=0)
        rec = recall_score(Y_test, test_preds, zero_division=0)

        print("====================================")
        print("Neural Network Results")
        print("Confusion Matrix:\n", cm)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print("====================================")