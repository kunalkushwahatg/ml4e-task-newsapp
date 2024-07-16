import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_pickle('data.pkl')
similarity_matrix = cosine_similarity(list(df['vector']))

def recommend_based_on_history(article_indices, top_n=5):
    mean_similarity_scores = np.mean([similarity_matrix[i] for i in article_indices], axis=0)
    sorted_indices = np.argsort(mean_similarity_scores)[::-1]
    recommended_indices = [i for i in sorted_indices if i not in article_indices][:top_n]
    return recommended_indices

last_watched_articles = [0, 1, 1000, 3, 4] 
recommended_articles = recommend_based_on_history(last_watched_articles)
for i in recommended_articles:
    print("Title : ", df['title'][i], "URL : ", df['url'][i])
    print('\n\n')
    print('content : \n', df['article'][i])
    print('\n\n')
    print('\n\n')