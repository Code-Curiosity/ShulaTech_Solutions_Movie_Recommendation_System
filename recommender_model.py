import pandas as pd
# Merging the columns of two datasets to create a single dataset with all the required features
def load_and_prepare_data():
    # Load datasets
    ratings = pd.read.csv('ratings.csv')
    movies = pd.read.csv('movies.csv')
    #Mergint the two of them
    data = pd.merge(ratings, movies)
    # Drop unnecessary columns
    data = data.drop(columns=['timestamp'])
    return data

# Now this is the merged data where the columns are : userId, movieId, rating, title, genres
# And rows are the ratings given by users to movies, but this is not feasable for the recommender system
# So we need to create a user-item matrix where the rows are users and columns are movies

#Create a user-Movie-Matrix
def create_user_movie_matrix(data):
    user_movie_matrix = data.pivot_table(index='userId',columns='movieId',values='rating')
    return user_movie_matrix
print(create_user_movie_matrix)
# Now we have a user-movie matrix where the rows are users and columns are movies, and the values are the ratings given by users to movies
# This will be used for the training of the model

