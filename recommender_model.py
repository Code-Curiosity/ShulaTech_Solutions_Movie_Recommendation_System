import pandas as pd
# Merging the columns of two datasets to create a single dataset with all the required features
def load_and_prepare_data():
    # Load datasets
    ratings = pd.read_csv('ratings.csv')
    movies = pd.read_csv('movies.csv')
    #Mergint the two of them
    data = pd.merge(ratings, movies, on='movieId')
    # Drop unnecessary columns
    finaldata = data.drop(columns=['timestamp'])
    return finaldata

# Now this is the merged data where the columns are : userId, movieId, rating, title, genres
# And rows are the ratings given by users to movies, but this is not feasable for the recommender system
# So we need to create a user-item matrix where the rows are users and columns are movies

#Create a user-Movie-Matrix
def create_user_movie_matrix(finaldata):
    matrix = finaldata.pivot_table(index='userId',columns='movieId',values='rating')
    return matrix
# Load data from the function and storing it in a variable name data 
data = load_and_prepare_data()
# Calling the function and storing it in a variable
user_movie_matrix = create_user_movie_matrix(data)

# Fill NaN with 0 (optional, but needed for cosine similarity)
user_movie_matrix = user_movie_matrix.fillna(0)

# Now we have a user-movie matrix where the rows are users and columns are movies, and the values are the ratings given by users to movies
# This is the core dataset which will be used to derive recommendations using collaborative filtering


