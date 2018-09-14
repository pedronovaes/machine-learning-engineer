import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display
from time import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import fbeta_score, accuracy_score, make_scorer
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


def print_results(results):
    # Printing out the values
    for i in results.items():
        print("")
        print(i[0])
        display(pd.DataFrame(i[1]).rename(columns={0: '1%', 1: '10%', 2: '100%'}))
    print("")


def plot_relationships(data):
    sns.pairplot(data, palette='Set1', hue='income')
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


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
        - learner: the learning algorithm to be trained and predicted on
        - sample_size: the size of samples (number) to be drawn from training set
        - X_train: features training set
        - y_train: income training set
        - X_test: features testing set
        - y_test: income testing set
    '''
    results = {}

    # Training data
    start = time()
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()
    results['train_time'] = end - start

    # Predictions
    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()
    results['pred_time'] = end - start

    # Computing accuracy on the first 300 training samples and on the test set
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # Computing F-score on the frist 300 training samples and on the test set
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, 0.5)
    results['f_test'] = fbeta_score(y_test, predictions_test, 0.5)

    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results


def models_evaluation(X_train, X_test, y_train, y_test):
    # Initial model validation
    clf_A = GaussianNB()
    clf_B = LogisticRegression(random_state=42)
    clf_C = GradientBoostingClassifier(random_state=42)

    samples_100 = len(y_train)
    samples_10 = int(len(y_train) / 10)
    samples_1 = int(len(y_train) / 100)

    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

    print_results(results)


def improving_results(X_train, X_test, y_train, y_test):
    start = time()

    # Using grid search to improve best model
    clf = GradientBoostingClassifier(random_state=42)

    parameters = {
        'learning_rate': [0.1, 1, 1.3],
        'n_estimators': [100, 300, 500]
    }

    # Make an fbeta_score scoring object using make_scorer()
    scorer = make_scorer(fbeta_score, beta=0.5)

    # Perform grid search
    grid_obj = GridSearchCV(clf, parameters, scoring=scorer, n_jobs=4, verbose=10)

    # Fit the grid search
    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the estimator
    best_clf = grid_fit.best_estimator_

    # Make predictions using unoptimized and optimized model
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    end = time()

    print("Unoptimized model\n------")
    print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5)))
    print("\nOptimized Model\n------")
    print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5)))
    print("Improving results in {:.2f} seconds".format(end - start))

    return best_clf, best_predictions


def extracting_features(best_clf, best_predictions, X_train, X_test, y_train, y_test):
    # Extracting feature importance
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    importances = model.feature_importances_

    # Reduce the feature space
    X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
    X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

    clf = (clone(best_clf)).fit(X_train_reduced, y_train)

    reduced_predictions = clf.predict(X_test_reduced)

    print("Final Model trained on full data\n------")
    print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=0.5)))
    print("\nFinal Model trained on reduced data\n------")
    print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta=0.5)))


if __name__ == '__main__':
    # Loading dataset
    in_file = 'census.csv'
    data = pd.read_csv(in_file)

    display(data.head())
    print(data.describe())

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

    plot_relationships(data)

    # Data Preprocessing
    features_final, income = preprocessing(data)

    # Shuffle and Split data
    X_train, X_test, y_train, y_test = train_test_split(features_final, income, test_size=0.2, random_state=0)

    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

    # Naive Predictor Performance
    naive_predictor_performance(income, n_records)

    # Creating a training and predicting pipeline:
    # 1 - Training three supervised models
    # 2 - Improving results of the best model for this dataset (Gradient Boosting)
    # 3 - Extracting the most important features and training Gradient Boosting with these features
    models_evaluation(X_train, X_test, y_train, y_test)
    best_clf, best_predictions = improving_results(X_train, X_test, y_train, y_test)
    extracting_features(best_clf, best_predictions, X_train, X_test, y_train, y_test)
