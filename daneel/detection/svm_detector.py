import numpy as np
import pandas as pd

from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score


class LightFluxProcessor:

    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize

    def fourier_transform(self, X):
        X = np.asarray(X, dtype=float)
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
            X_train = ndimage.gaussian_filter(X_train, sigma=2)
            X_test  = ndimage.gaussian_filter(X_test,  sigma=2)

        if self.standardize:
            print("Standardizing...")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        print("Finished Processing!\n")
        return X_train, X_test

class SVMDetector:
    def __init__(self, dataset_path, kernel_name, degree=None):
        self.dataset_path = dataset_path
        self.kernel_name = kernel_name
        self.degree = degree

    def load_data(self):
        train_path = f"{self.dataset_path}/exoTrain.csv"
        test_path  = f"{self.dataset_path}/exoTest.csv"

        df_train = pd.read_csv(train_path, encoding="ISO-8859-1")
        df_test  = pd.read_csv(test_path,  encoding="ISO-8859-1")

        return df_train, df_test

    def run_detection(self):

        df_train, df_test = self.load_data()

        df_train_x = df_train.drop("LABEL", axis=1)
        df_test_x  = df_test.drop("LABEL", axis=1)
        Y_train = df_train["LABEL"]
        Y_test  = df_test["LABEL"]

        processor = LightFluxProcessor(
            fourier=True,
            normalize=True,
            gaussian=True,
            standardize=True
        )
        X_train, X_test = processor.process(df_train_x, df_test_x)

        if self.kernel_name == 'polynomial':
            model = SVC(kernel='poly', degree=self.degree, class_weight={1:1, 2:20})
        elif self.kernel_name == 'gaussian':
            model = SVC(kernel='rbf', class_weight={1:1, 2:20})
        else:
            model = SVC(kernel='linear', class_weight={1:1, 2:20})

        print("Training SVM...")
        model.fit(X_train, Y_train)
        preds = model.predict(X_test)

        cm = confusion_matrix(Y_test, preds)
        precision = precision_score(Y_test, preds, zero_division=0)

        print(f"\nKernel: {self.kernel_name}")
        print("Confusion Matrix:\n", cm)
        print(f"Precision: {precision:.4f}")
