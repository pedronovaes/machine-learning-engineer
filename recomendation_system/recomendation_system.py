import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import helper
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans


def plot_elbow_method(possible_k_values, errors_per_k):
    # Plot the each value of K vs. the silhouette score at that value
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlabel('K - number of clusters')
    ax.set_ylabel('Silhouette Score (higher is better)')
    ax.plot(possible_k_values, errors_per_k)

    # Ticks and grid
    xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
    ax.set_xticks(xticks, minor=False)
    ax.set_xticks(xticks, minor=True)
    ax.xaxis.grid(True, which='both')
    yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
    ax.set_yticks(yticks, minor=False)
    ax.set_yticks(yticks, minor=True)
    ax.yaxis.grid(True, which='both')


def making_cluster(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    predictions = kmeans.fit_predict(X)

    return predictions


if __name__ == '__main__':
    # Import the Movies dataset
    movies = pd.read_csv('ml-latest-small/movies.csv')
    # print(movies.head())

    # Import the Ratings dataset
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    # print(ratings.head())

    # print('The dataset contains: {} ratings of {} movies.'.format(len(ratings), len(movies)))

    # Taking a subset of users and seeing the respective rating of romance and scifi movies
    genre_ratings = helper.get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi'], ['avg_romance_rating', 'avg_scifi_rating'])
    # print(genre_ratings.head())

    # Removing people who like both scifi and romance
    biased_dataset = helper.bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)
    # print('Number of records: {}'.format(len(biased_dataset)))
    # print(biased_dataset.head())

    # Plotting the dataset
    # helper.draw_scatterplot(biased_dataset['avg_scifi_rating'],'Avg scifi rating', biased_dataset['avg_romance_rating'], 'Avg romance rating')

    # Turning the dataset into a list
    X = biased_dataset[['avg_scifi_rating', 'avg_romance_rating']].values

    # Creating two clusters
    predictions_1 = making_cluster(X, 2)
    # helper.draw_clusters(biased_dataset, predictions_1)

    # Three clusters
    predictions_2 = making_cluster(X, 3)
    # helper.draw_clusters(biased_dataset, predictions_2)

    # Four clusters
    predictions_3 = making_cluster(X, 4)
    # helper.draw_clusters(biased_dataset, predictions_3)

    # Using Elbow Method to find best K value
    # possible_k_values = range(2, len(X) + 1, 5)
    # errors_per_k = [helper.clustering_errors(k, X) for k in possible_k_values]
    # plot_elbow_method(possible_k_values, errors_per_k)

    # Seven clusters
    predictions_4 = making_cluster(X, 7)
    # helper.draw_clusters(biased_dataset, predictions_4)

    # Taking a subset of users and seeing the respective rating of romance, scifi and action movies
    biased_dataset_3_genres = helper.get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi', 'Action'], ['avg_romance_rating', 'avg_scifi_rating', 'avg_action_rating'])
    biased_dataset_3_genres = helper.bias_genre_rating_dataset(biased_dataset_3_genres, 3.2, 2.5).dropna()
    # print('Number of records: {}'.format(len(biased_dataset_3_genres)))
    # print(biased_dataset_3_genres.head())

    X_with_action = biased_dataset_3_genres[['avg_scifi_rating', 'avg_romance_rating', 'avg_action_rating']].values

    # Clustering with 3 genres subset
    predictions_5 = making_cluster(X_with_action, 7)
    # helper.draw_clusters(biased_dataset_3_genres, predictions_5)

    # Movie-level Clustering
    # Merge the two tables then pivot so we have Users X Movies dataframe
    ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId' )
    user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
    # print('Dataset dimensions: {}\nSubset example:'.format(user_movie_ratings.shape))
    # print(user_movie_ratings.iloc[:6, :10])

    # 30 most-rated movies and 18 users with the most ratings
    n_movies = 30
    n_users = 18
    most_rated_movies_users_selection = helper.sort_by_rating_density(user_movie_ratings, n_movies, n_users)
    # print('Dataset dimensions: ', most_rated_movies_users_selection.shape)
    # print(most_rated_movies_users_selection.head())
    # helper.draw_movies_heatmap(most_rated_movies_users_selection)

    user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
    most_rated_movies_1k = helper.get_most_rated_movies(user_movie_ratings, 1000)

    # Converting a dataframe to a sparse matrix
    sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies_1k).to_coo())

    # Twenty clusters
    predictions = KMeans(n_clusters=20, algorithm='full', random_state=42).fit_predict(sparse_ratings)

    # Visualizing some clusters
    max_users = 70
    max_movies = 50
    clustered = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group':predictions})], axis=1)
    # helper.draw_movie_clusters(clustered, max_users, max_movies)

    # Prediction
    # Pick a cluster ID from the clusters above
    cluster_number = 8
    n_users = 75
    n_movies = 300
    cluster = clustered[clustered.group == cluster_number].drop(['index', 'group'], axis=1)
    cluster = helper.sort_by_rating_density(cluster, n_movies, n_users)
    # helper.draw_movies_heatmap(cluster, axis_labels=False)
    # print(cluster.fillna('').head())

    movie_name = 'Godfather, The (1972)'
    # print(cluster[movie_name].mean())

    # Recommendation
    # print(cluster.mean().head(20))
    user_id = 2
    user_2_ratings = cluster.loc[user_id, :]
    user_2_unrated_movies = user_2_ratings[user_2_ratings.isnull()]
    avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:, 0]
    print(avg_ratings.sort_values(ascending=False)[:20])
