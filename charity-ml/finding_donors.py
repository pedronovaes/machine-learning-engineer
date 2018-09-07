# Importing libraries necessary for this project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler


def plot_relationships(x_axis, hue, data, y_axis=None):
    sns.countplot(x=x_axis, y=y_axis,  hue=hue, data=data)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Loading dataset
    in_file = 'census.csv'
    data = pd.read_csv(in_file)

    # display(data.head())

    # Making some data exploration
    n_records = len(data)
    n_greater_50k = len(data[data['income'] == '>50K'])
    n_at_most_50k = len(data[data['income'] == '<=50K'])
    greater_percent = n_greater_50k / n_records * 100

    # Print the results
    print("Total number of records: {}".format(n_records))
    print("Individuals making more than $50,000: {}".format(n_greater_50k))
    print("Individuals making at most $50,000: {}".format(n_at_most_50k))
    print("Percentage of individuals making more than $50,000: {0:.2f}%".format(greater_percent))

    # plot_relationships(x_axis="age", hue="income", data=data)

    # Preparing the data
    # --- Transforming skewed continuous features
    # --- Split the data into features and target lagel
    income_raw = data['income']
    features_raw = data.drop('income', axis=1)

    # --- Log-transform the skewed features
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data=features_raw)
    # display(features_log_transformed.head())
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))
    display(features_log_transformed.head())

    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()  # default=(0, 1)
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
    display(features_log_minmax_transform.head())
