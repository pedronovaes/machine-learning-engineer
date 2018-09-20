import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helper
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


if __name__ == '__main__':
    # Import the Movies dataset
    movies = pd.read_csv('ml-latest-small/movies.csv')
    print(movies.head())

    # Import the Ratings dataset
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    print(ratings.head())

    print('The dataset contains: {} ratings of {} movies.'.format(len(ratings), len(movies)))

    # Taking a subset of users and seeing the respective rating of romance and scifi movies
    genre_ratings = helper.get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'], ['avg_romance_rating', 'avg_scifi_rating'])
    print(genre_ratings.head())

    # Removing people who like both scifi and romance
    biased_dataset = helper.bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)
    print('Number of records: {}'.format(len(biased_dataset)))
    print(biased_dataset.head())

    # Plotting the dataset
    # helper.draw_scatterplot(biased_dataset['avg_scifi_rating'],'Avg scifi rating', biased_dataset['avg_romance_rating'], 'Avg romance rating')

    # Turning the dataset into a list
    X = biased_dataset[['avg_scifi_rating', 'avg_romance_rating']].values

    # Creating an instance of KMeans to find two clusters
    kmeans_1 = KMeans(n_clusters=2)
    predictions_1 = kmeans_1.fit_predict(X)
    # helper.draw_clusters(biased_dataset, predictions_1)

    # Three clusters
    kmeans_2 = KMeans(n_clusters=3)
    predictions_2 = kmeans_2.fit_predict(X)
    # helper.draw_clusters(biased_dataset, predictions_2)

    # Four clusters
    kmeans_3 = KMeans(n_clusters=4)
    predictions_3 = kmeans_3.fit_predict(X)
    helper.draw_clusters(biased_dataset, predictions_3)
