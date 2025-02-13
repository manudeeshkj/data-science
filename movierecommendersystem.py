import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate, train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load MovieLens dataset
movies = pd.read_csv("https://files.grouplens.org/datasets/movielens/ml-latest-small/movies.csv")
ratings = pd.read_csv("https://files.grouplens.org/datasets/movielens/ml-latest-small/ratings.csv")

# Collaborative Filtering (Matrix Factorization with SVD)
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)
predictions = model.test(testset)

# Content-Based Filtering using TF-IDF on movie genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on content similarity
def recommend_movies(movie_title, top_n=10):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]

# Hybrid Recommendation System
def hybrid_recommend(user_id, movie_title, top_n=10):
    content_recs = recommend_movies(movie_title, top_n)
    user_ratings = ratings[ratings['userId'] == user_id]
    user_movies = user_ratings.merge(movies, on='movieId')[['title', 'rating']]
    return content_recs.merge(user_movies, on='title', how='left').fillna('Not Rated')

# Example usage
print(recommend_movies("Toy Story (1995)", 5))