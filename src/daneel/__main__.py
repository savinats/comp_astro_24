import os
import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import *


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        required=False,
        help="Initialise detection algorithms for Exoplanets",
        action="store_true",
    )

    parser.add_argument(
        "-a",
        "--atmosphere",
        dest="atmosphere",
        required=False,
        help="Atmospheric Characterisazion from input transmission spectrum",
        action="store_true",
    )

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    input_pars = Parameters(args.input_file).params

    if args.detect:
        """Example usage

        The following commands have been taken from the SVM detector notebook.
        Everything can be arranged in a separated file within the detection folder
        as it has been done for the LightFluxProcessor object.

        Take it as example to build your own package
        """

        train_dataset_path = input_pars["detection"]["dataset"]["train"]
        dev_dataset_path = input_pars["detection"]["dataset"]["test"]

        print("Loading datasets...")
        df_train = pd.read_csv(train_dataset_path, encoding="ISO-8859-1")
        df_dev = pd.read_csv(dev_dataset_path, encoding="ISO-8859-1")
        print("Loaded datasets!")

        # Generate X and Y dataframe sets
        df_train_x = df_train.drop("LABEL", axis=1)
        df_dev_x = df_dev.drop("LABEL", axis=1)
        df_train_y = df_train.LABEL
        df_dev_y = df_dev.LABEL

        LFP = LightFluxProcessor(
            fourier=True, normalize=True, gaussian=True, standardize=True
        )
        df_train_x, df_dev_x = LFP.process(df_train_x, df_dev_x)

        # display(df_train_x)

        # Rejoin X and Y
        df_train_processed = pd.DataFrame(df_train_x).join(pd.DataFrame(df_train_y))
        df_dev_processed = pd.DataFrame(df_dev_x).join(pd.DataFrame(df_dev_y))

        # Load X and Y numpy arrays
        X_train, Y_train = np_X_Y_from_df(df_train_processed)
        X_dev, Y_dev = np_X_Y_from_df(df_dev_processed)

        print("X train shape: ", X_train.shape)
        print("X test shape: ", X_dev.shape)
    if args.atmosphere:
        pass

    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()
