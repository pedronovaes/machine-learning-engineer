import numpy as np
import pandas as pd
from IPython.display import display


if __name__ == '__main__':
    # Loading dataset
    try:
        data = pd.read_csv('customers.csv')
        data.drop(['Region', 'Channel'], axis=1, inplace=True)
        print('Wholesale customers dataset has {} samples with {} features each.'.format(*data.shape))
    except:
        print('Dataset could not be loaded. Is the dataset missing?')
