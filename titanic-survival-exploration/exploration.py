# Importing libraries
import numpy as np
import pandas as pd
from IPython.display import display
# import visuals as vs


if __name__ == '__main__':
    # Loading dataset
    in_file = "titanic_data.csv"
    full_data = pd.read_csv(in_file)

    # Print the first few entries of the RMS Titanic data
    display(full_data.head())
