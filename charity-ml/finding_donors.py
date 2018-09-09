# Importing libraries necessary for this project
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def plot_relationships(x_axis, hue, data, y_axis=None):
    sns.countplot(x=x_axis, y=y_axis,  hue=hue, data=data)
    plt.tight_layout()
    plt.show()


def preprocessing(data):
    # Transforming skewed continuous features and splitting the data into features and target label
    income_raw = data['income']
    features_raw = data.drop('income', axis=1)

    # Log-transform the skewed features
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data=features_raw)
    features_log_transformed[skewed] = features_log_transformed[skewed].apply(lambda x: np.log(x + 1))

    # Normalizing numerical features
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler()
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data=features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

    # Applying one-hot encoding to non-numerical features
    features_final = pd.get_dummies(features_log_minmax_transform)

    # Encode the 'income_raw' data to numerical values
    income = income_raw.apply(lambda x: 1 if x == '>50K' else 0)

    encoded = list(features_final.columns)
    print("{} total features after one-hot encoding".format(len(encoded)))

    return features_final, income


# Naive model that always predict an individual made more than $50,000
def naive_predictor_performance(income, n_records):
    TP = np.sum(income)
    FP = income.count() - TP
    TN = 0
    FN = 0

    # Calculate accuracy, precision and recall
    accuracy = TP / n_records
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    # Calculate F-score for beta = 0.5
    fscore = (1 + 0.5 ** 2) * precision * recall / ((0.5 ** 2 * precision) + recall)
    print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


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

    # Data Preprocessing
    features_final, income = preprocessing(data)

    # Shuffle and Split data
    X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size=0.2, random_state=0)

    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

    # Naive Predictor Performance
    naive_predictor_performance(income, n_records)
