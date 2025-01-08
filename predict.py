# Define the function that predicts the top 5 movies for a given user
def predict(user_data, itm_vectors, itm_biases , movie_file, top=5, k=20, lamda=0.01, tau=0.8):

    score_item = np.zeros(len(map_index_to_movie))
    first_term = 0
    second_term = np.zeros(k)

    # Create user vector
    for (n,r) in user_data:
        r = float(r)
        first_term += lamda * np.outer(itm_vectors[n], itm_vectors[n])
        second_term += lamda * (r - itm_biases[n]) * itm_vectors[n]

    first_term += (tau * np.eye(k))
    first_term = np.linalg.inv(first_term)

    user_vector = first_term @ second_term

    # Cite the data of the user
    index_in_user_data = []
    for (n,r) in user_data:
        index_in_user_data.append(n)
        movie_index = map_index_to_movie[n]
        r = float(r)

        movie = movie_file['title'][movie_file['movieId'] == int(movie_index)].values[0]
        print(f'The user rated {r} stars to **{movie}**')
    print('-------------------------------------------------------------')
    print(f'The top {top} recommendations for this user are :')

    # Get the top movies for the user
    for n in range(len(map_index_to_movie)):
        if n in index_in_user_data:
            continue
        score_item[n] = np.inner(user_vector, itm_vectors[n])+ 0.05*itm_biases[n]
    top_movies = np.argsort(score_item, axis=0)[-top:][::-1]


    for i in range(top):
        movie_id = map_index_to_movie[top_movies[i]]
        movie_name = movie_file['title'][movie_file['movieId'] == int(movie_id)].values[0]
        print(f'Top {i+1}: {movie_name}')

# Take a movie index
movie_index = 693 #7078 #1017 #202 #1054 #664 #70 #

# Create dummy user with similar data as in data_by_user_index 
DBUI_dummy = [(movie_index, '5')]

# Make prediction of the top 5 movies that this user might also like
predict(DBUI_dummy, item_vectors, item_biases, df, top=5, k=20, lamda=0.002, tau=0.01)  
