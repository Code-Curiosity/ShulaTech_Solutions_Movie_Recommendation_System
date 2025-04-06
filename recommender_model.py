import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Merging the columns of two datasets to create a single dataset with all the required features
def load_and_prepare_data():
    # Load datasets
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    #Mergint the two of them
    data = pd.merge(ratings, movies, on='movieId')
    # Drop unnecessary columns
    finaldata = data.drop(columns=['timestamp'])
    return finaldata, movies

# Now this is the merged data where the columns are : userId, movieId, rating, title, genres
# And rows are the ratings given by users to movies, but this is not feasable for the recommender system
# So we need to create a user-item matrix where the rows are users and columns are movies
# We are loading this finaldata to a variable so that the function can be called later
data, movies = load_and_prepare_data()
#Create a user-Movie-Matrix
def create_user_movie_matrix(data):
    matrix = data.pivot_table(index='userId',columns='movieId',values='rating')
    return matrix

# Calling the function and storing it in a variable
user_movie_matrix = create_user_movie_matrix(data)

# Fill NaN with 0 (optional, but needed for cosine similarity)
user_movie_matrix = user_movie_matrix.fillna(0)

# Now we have a user-movie matrix where the rows are users and columns are movies, and the values are the ratings given by users to movies
# This is the core dataset which will be used for making recommender and for the same we are using user-based collaborative filtering
# Cosine similarity calculates similarity between each user with other users based on their ratings.

def compute_user_similarity(user_movie_matrix):
    # 👇 Fill NaNs with 0 — assumes unrated movies are just 0
    user_movie_matrix_filled = user_movie_matrix.fillna(0)

    # Compute cosine similarity
    user_similarity = cosine_similarity(user_movie_matrix_filled)

    # Convert to DataFrame for easier handling
    similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    return similarity_df
# Creating the recommender logic
# We are using the top n similar users to the target user to make recommendations better
# We are using the weighted average of the ratings given by similar users to make recommendations
# The recommender logic will take user_id, user_movie_matrix, similarity_matrix and n_recommendations as inputs
# The recommender logic will return the top n recommendations for the user based on the ratings given by similar users
# The recommender logic will use the user_movie_matrix and similarity_matrix to make recommendations
def recommender_logic(user_id, user_movie_matrix, similarity_matrix, movies, n_recommendations=5):
    # Get the user's ratings
    user_ratings = user_movie_matrix.loc[user_id]

    # Get the indices of the top n similar users
    similar_users_indices = similarity_matrix[user_id].argsort()[-n_recommendations-1:-1][::-1]

    # Get the ratings of the similar users
    similar_users_ratings = user_movie_matrix.iloc[similar_users_indices]

    # Calculate the weighted average of the ratings from similar users
    weighted_ratings = np.dot(similarity_matrix[user_id][similar_users_indices], similar_users_ratings)

    # Normalize by the sum of similarities
    recommendations = weighted_ratings / np.sum(similarity_matrix[user_id][similar_users_indices])

    # Create a DataFrame for recommendations
    recommendations_df = pd.DataFrame(recommendations, index=user_movie_matrix.columns, columns=['Predicted Rating'])

    # Remove movies the user has already rated
    watched_movies = user_ratings[user_ratings > 0].index
    recommendations_df = recommendations_df.drop(watched_movies, errors='ignore')

    # Sort by predicted rating and get top n recommendations
    top_recommendations = recommendations_df.sort_values(by='Predicted Rating', ascending=False).head(n_recommendations)

    # Merge with movies DataFrame to get titles
    top_recommendations = top_recommendations.merge(movies, left_index=True, right_on='movieId')

    return top_recommendations[['title', 'Predicted Rating']]

