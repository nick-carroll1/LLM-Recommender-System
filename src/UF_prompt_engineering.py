import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import openai
from key import OPENAI_API_KEY
import pandas as pd
import numpy as np


# Function to find most similar users
def get_similar_users(user_id, matrix, m):
    if user_id not in matrix.index:
        return []
    sim_users = matrix.loc[user_id].sort_values(ascending=False).iloc[1:m+1].index.tolist()
    return sim_users

def user_filtering_recommendations(dataframe, target_user_id, m, ns):
    """
    Generate movie recommendations for a target user based on user-filtering.

    :param dataframe: A pandas DataFrame containing columns 'user_id', 'movie_id', 'rating', 'movie title'.
    :param target_user_id: The user ID for whom recommendations are to be generated.
    :param m: The number of similar users to consider.
    :param ns: The number of candidate items to recommend.
    :return: A list of recommended movie titles.
    """
    # Create a pivot table
    user_movie_matrix = dataframe.pivot_table(index='user_id', columns='movie_id', values='avg_rating', fill_value=0)

    # Convert to sparse matrix
    sparse_matrix = csr_matrix(user_movie_matrix)

    # Compute Cosine Similarity
    cosine_sim = cosine_similarity(sparse_matrix)

    # Convert to DataFrame
    cosine_sim_df = pd.DataFrame(cosine_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    # Find similar users
    similar_users = get_similar_users(target_user_id, cosine_sim_df, m)

    # Get candidate movie IDs
    candidate_ids = dataframe[dataframe['user_id'].isin(similar_users)]['movie_id'].value_counts().head(ns).index

    # Map IDs to Titles
    candidate_titles = dataframe[dataframe['movie_id'].isin(candidate_ids)]['movie_title'].unique().tolist()

    return candidate_titles

# wrap into a function
def rec_from_openai(df, m, n, ns, user_id, temp_1, temp_2, random_seed=42):
    """
    Generate movie recommendations for a target user based on user-filtering.
    Input:
        df: dataframe
        m: number of similar users to consider
        n: number of movies to select from total watched movies of this user
        ns: number of candidate items to recommend
        user_id: target user id
        temp_1: OpenAI prompt template for step 1
        temp_2: OpenAI prompt template for step 2
    Output:
        recommendations: a list of recommended movie titles
        hit_rate: hit rate of recommendations in watched movies
    """

    np.random.seed(random_seed)   

    watched_movies = df[df['user_id'] == user_id]['movie_title'].unique().tolist()
    selected_watched_movies = np.random.choice(watched_movies, min(len(watched_movies), n), replace=False).tolist()
    # combine movie name and genre
    selected_watched_movies_genres = set(df[df['movie_title'].isin(selected_watched_movies)]['genres'].unique().tolist())
    candidate_movies = user_filtering_recommendations(df, target_user_id=user_id, m=m, ns=ns)
    
    Input_1 = temp_1.format(candidate_movies, selected_watched_movies, selected_watched_movies_genres)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [{ 'role':'user','content' : Input_1}],
        # temperature=0,
        # max_tokens=512,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0,
        )
    prediction_1 = response.choices[0].message.content

    Input_2 = temp_2.format(candidate_movies, selected_watched_movies, selected_watched_movies_genres, prediction_1)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages= [{ 'role':'user','content' : Input_2}],
        # temperature=0,
        # max_tokens=512,
        # top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0,
        )
    prediction_2 = response.choices[0].message.content

    # print(Input_1)
    # print(prediction_1)
    # print(Input_2)
    # print(prediction_2)

    recommendations = []
    for movie in prediction_2.split('\n'):
        split_movie = movie.split('.')
        if len(split_movie) > 1:
            recommendations.append(split_movie[1].strip())
    # hit rate of recommendations in watched movies
    if len(recommendations) == 0:
        hit_rate = 0
    else:
        hit_rate = len(set(recommendations).intersection(set(watched_movies))) / len(recommendations)

    return recommendations, hit_rate
    

# test
if __name__ == '__main__':
    openai.api_key = OPENAI_API_KEY

    df = pd.read_csv('/workspaces/LLM-Recommender-System/data/processed_movie100k.csv')

    temp_1 = """
    Candidate Set (candidate movies): {}.
    The movies I have watched (watched movies): {}.
    Their genres are: {}.
    Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? 
    Answer: 
    """

    temp_2 = """
    Candidate Set (candidate movies): {}.
    The movies I have watched (watched movies): {}.
    Their genres are: {}.
    Step 1: What features are most important to me when selecting movies (Summarize my preferences briefly)? 
    Answer: {}.
    Step 2: Can you recommend 10 movies from the Candidate Set similar to but not in the selected movies I've watched?.
    (Format: Here are the 10 movies recommended for you: [no. a candidate movie])
    Answer: 
    """
    hit_rates = []
    for i in range(1,101):
        user_id = i
        m = 10
        n = 5
        ns = 20
        recommendations, hit_rate = rec_from_openai(df, m, n, ns, user_id, temp_1, temp_2)
        hit_rates.append(hit_rate)
        print(recommendations)
        print(hit_rate)
        print('='*50)
    print(f'Average hit rate: {np.mean(hit_rates)}')