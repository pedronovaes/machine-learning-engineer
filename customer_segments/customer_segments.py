import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


def loading_dataset():
    try:
        data = pd.read_csv('customers.csv')
        data.drop(['Region', 'Channel'], axis=1, inplace=True)
        print('Wholesale customers dataset has {} samples with {} features each.'.format(*data.shape))
        return data
    except:
        print('Dataset could not be loaded. Is the dataset missing?')


def initial_exploratory_analysis(data):
    # Scatter matrix
    pd.plotting.scatter_matrix(data, alpha=0.3, figsize=(20, 20), diagonal='kde')
    plt.show()

    # Heatmap
    plt.figure(figsize=(10, 10))
    sns.heatmap(data.corr(), annot=True)
    plt.show()


def preprocessing(data):
    # Applying logarithmic scale
    log_data = np.log(data)
    pd.plotting.scatter_matrix(log_data, alpha=0.3, figsize=(20, 20), diagonal='kde')
    plt.show()
    return log_data


if __name__ == '__main__':
    # Loading dataset
    data = loading_dataset()
    display(data.head())
    display(data.describe())

    # Scatter matrix and heatmap plots
    initial_exploratory_analysis(data)

    log_data = preprocessing(data)
