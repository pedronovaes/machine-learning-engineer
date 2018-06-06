# Importing libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display  # Allows the use of display() for dataframes
# import visuals as vs


# This function returns accuracy score for input truth and predictions
def accuracy_score(truth, pred):
    if (len(truth) == len(pred)):
        return "Predictions have an accuracy of {:.2f}%.".format((truth == pred).mean()*100)
    else:
        return "Number of predictions does not match number of outcomes!"


# Model with no features. Always predicts a passenger did not survive
def predictions_0(data):
    predictions = []
    for _, passenger in data.iterrows():
        # predict the survival of "passenger"
        predictions.append(0)
    return pd.Series(predictions)


# Model with one feature:
# - predict a passenger survived if they are female
def predictions_1(data):
    predictions = []
    for _, passenger in data.iterrows():
        if (passenger["Sex"] == "female"):
            predictions.append(1)
        else:
            predictions.append(0)
    return pd.Series(predictions)


if __name__ == '__main__':
    # Loading dataset
    in_file = "titanic_data.csv"
    full_data = pd.read_csv(in_file)

    # Print the first few entries of the RMS Titanic data
    # display(full_data.head())

    # Removing "Survived" variable from the dataset
    # We will use these outcomes as our prediction targets
    outcomes = full_data["Survived"]
    data = full_data.drop("Survived", axis=1)

    predictions = pd.Series(np.ones(5, dtype=int))
    print(accuracy_score(outcomes[:5], predictions))

    predictions = predictions_0(data)
    print(accuracy_score(outcomes, predictions))

    predictions = predictions_1(data)
    print(accuracy_score(outcomes, predictions))
