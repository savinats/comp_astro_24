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
            X_train = ndimage.gaussian_filter(X_train, sigma=0.2)
            X_test  = ndimage.gaussian_filter(X_test,  sigma=0.2)

        if self.standardize:
            print("Standardizing...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        print("Finished Processing!\n")
        return X_train, X_test

class CNNDetector:

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
            tf.keras.layers.Input(shape=input_shape),

            tf.keras.layers.Conv2D(16, (3,3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2,2)),

            tf.keras.layers.Conv2D(32, (3,3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2,2)),

            tf.keras.layers.Conv2D(64, (3,3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2,2)),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
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

        print("Reshaping into 40x40 images...")

        target_size = 1600
        side = 40

        def pad_to_square(arr, size):
            if arr.shape[1] >= size:
                return arr[:, :size]
            pad_width = size - arr.shape[1]
            return np.hstack([arr, np.zeros((arr.shape[0], pad_width))])

        X_train = pad_to_square(X_train, target_size)
        X_test  = pad_to_square(X_test,  target_size)

        X_train = X_train.reshape((-1, side, side, 1))
        X_test  = X_test.reshape((-1, side, side, 1))
        
        df_train_processed = pd.DataFrame(X_train.reshape(len(X_train), -1)).join(df_train_y)
        df_test_processed  = pd.DataFrame(X_test.reshape(len(X_test), -1)).join(df_test_y)
        
        X_train_arr, Y_train = self.np_X_Y_from_df(df_train_processed)
        X_test_arr,  Y_test  = self.np_X_Y_from_df(df_test_processed)

        print("Building CNN...")
        model = self.build_network((40, 40, 1))

        print("Training CNN...")
        class_weights = {1: 1, 2: 150}
        model.fit(X_train, Y_train, epochs=5, batch_size=32, class_weight=class_weights, verbose=0)

        print("Evaluating...")
        test_preds = (model.predict(X_test) > 0.25).astype(int)

        cm = confusion_matrix(Y_test, test_preds)
        acc = accuracy_score(Y_test, test_preds)
        prec = precision_score(Y_test, test_preds, zero_division=0)
        rec = recall_score(Y_test, test_preds, zero_division=0)

        print("====================================")
        print("CNN Results")
        print("Confusion Matrix:\n", cm)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print("====================================")

