# Import the file 
from google.colab import drive
drive.mount('/content/drive')

import csv
import matplotlib.pyplot as plt
import numpy as np


ratings = []
with open(ratings_path, 'r') as ratings_file:
    data_reader = csv.reader(ratings_file, delimiter=',')
    next(data_reader, None)
    for row in data_reader:
        user_id, movie_id, rating, timestamp = row
        ratings.append([user_id, movie_id, rating, timestamp])

# A function to sparse the matrix 
def sparse(ratings):

    map_user_to_index={}
    map_index_to_user=[]
    map_movie_to_index={}
    map_index_to_movie=[]

    data_by_user_index = []
    data_by_movie_index = []

    for row in ratings:
      # User mapping
      user_name = row[0]
      movie_name = row[1]
      rating = row[2]
      if user_name not in map_user_to_index:
        map_user_to_index[user_name]=len(map_user_to_index)
        map_index_to_user.append(user_name)
        data_by_user_index.append([])

      # Movie mapping
      if movie_name not in map_movie_to_index:
        map_movie_to_index[movie_name]=len(map_movie_to_index)
        map_index_to_movie.append(movie_name)
        data_by_movie_index.append([])
      # Fill data_by_user_index and data_by_movie_index
      data_by_user_index[map_user_to_index[user_name]].append((map_movie_to_index[movie_name],rating))
      data_by_movie_index[map_movie_to_index[movie_name]].append((map_user_to_index[user_name],rating))
    sparse = map_user_to_index, map_index_to_user, map_movie_to_index, map_index_to_movie, data_by_user_index, data_by_movie_index
    return sparse

# Plot the power law distribution
movies_by_user = [len(user_list) for user_list in data_by_user_index]
user_per_movie = [len(movie_list) for movie_list in data_by_movie_index]

movies_by_user = [len(user_list) for user_list in data_by_user_index]
user_per_movie = [len(movie_list) for movie_list in data_by_movie_index]

# Split the data randomly
data_by_movie_index_train = []
data_by_movie_index_test = []
for i in range (len(data_by_movie_index)):
    data_by_movie_index_train.append([])
    data_by_movie_index_test.append([])

    for j in range (len(data_by_movie_index[i])):
        coin = random.random()
        if coin > 0.1 :
            data_by_movie_index_train[i].append(data_by_movie_index[i][j])
        else :
            data_by_movie_index_test[i].append(data_by_movie_index[i][j])

# Define the update for the biases
def update_user_bias(m, data_user_index, lamda, gamma):

        bias = 0
        item_counter = 0

        for (n,r) in data_user_index[m]:
            r = float(r)
            bias += lamda * (r - item_biases[n] - np.inner(user_vectors[m], item_vectors[n]))
            item_counter += 1
        bias = bias / (lamda * item_counter + gamma)
        return bias

def update_item_bias(n, data_movie_index, lamda, gamma):

        bias = 0
        user_counter = 0

        for (u,r) in data_movie_index[n]:
            r = float(r)
            bias += lamda * (r - user_biases[u] - np.inner(user_vectors[u], item_vectors[n]))
            user_counter += 1
        bias = bias / (lamda * user_counter + gamma)
        return bias

# Update the vectors
def update_user_vector(m, user_biases, item_biases, user_vects, item_vects, data_user_index, lamda, tau, k):
        first_term = 0
        second_term = np.zeros(k)

        for (n,r) in data_user_index[m]:
            r = float(r)
            first_term += lamda * np.outer(item_vects[n], item_vects[n])
            second_term += lamda * (r - user_biases[m] - item_biases[n]) * item_vects[n]

        first_term += (tau * np.eye(k))
        first_term = np.linalg.inv(first_term)

        user_vects[m] = first_term @ second_term

def update_item_vector(n, user_biases, item_biases, user_vects, item_vects, data_movie_index, lamda, tau, k):
        first_term = 0
        second_term = np.zeros(k)

        for (u,r) in data_movie_index[n]:
            r = float(r)
            first_term += lamda * np.outer(user_vects[u], user_vects[u])
            second_term += lamda * (r - user_biases[u] - item_biases[n]) * user_vects[u]

        first_term += (tau * np.eye(k))
        first_term = np.linalg.inv(first_term)

        item_vects[n] = first_term @ second_term

# Define a fuction which calculate the loss and the RMSE at the same time
def loss_RMSE(user_biases, item_biases, user_vects, item_vects, data_user_index, data_item_index, lamda, gamma, tau):
    loss = 0
    rmse_one = 0
    counter = 0

    for m in range(len(data_user_index)):

        for (n,r) in data_user_index[m]:
            r = float(r)
            error = r - (np.inner(user_vects[m] , item_vects[n]) + user_biases[m] + item_biases[n])
            loss += (lamda / 2) * (error**2)
            rmse_one += error**2
            counter += 1

    reg_user = 0
    for m in range(len(data_user_index)):
        reg_user += (tau / 2) * (np.inner(user_vects[m], user_vects[m]))

    reg_item = 0
    for n in range(len(data_item_index)):
        reg_item += (tau / 2) * (np.inner(item_vects[n], item_vects[n]))

    loss = loss + (gamma / 2) * (np.sum(user_biases**2) + np.sum(item_biases**2)) + reg_user + reg_item
    rmse = np.sqrt(rmse_one / counter)

    return -loss, rmse

def predict(user_data, itm_vectors, itm_biases , top=5,  k=20, lamda=0.01, tau=0.8):

    score_item = np.zeros(len(data_by_movie_index))
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

    # Get the top movies for the user
    for n in range(len(data_by_movie_index)):
        score_item[n] = np.inner(user_vector, itm_vectors[n]) + 0.05*itm_biases[n]
    top_movies = np.argsort(score_item, axis=0)[-top:][::-1]

    # Cite the data of the user
    for (n,r) in user_data:
        movie_index = map_index_to_movie[n]
        r = float(r)

        movie = movie_file['title'][movie_file['movieId'] == int(movie_index)].values[0]
        print(f'The user rated {r} stars to **{movie}**')
    print(f'The top {top} recommendations for this user are :')
    for i in range(top):
        movie_id = map_index_to_movie[top_movies[i]]
        movie_name = movie_file['title'][movie_file['movieId'] == int(movie_id)].values[0]
        print(f'Top {i+1}: {movie_name}')

# Initialize the parameters and and train the model 

# Initialisation of the parameters
k = 20
lamda = 0.01
gamma = 0.001
tau = 0.8
epoch = 10

# Initialisation of biases and vectors
M = len(data_by_user_index)
N = len(data_by_movie_index)

user_biases = np.zeros(M)
item_biases = np.zeros(N)

user_vectors = np.random.normal(0, 1 / np.sqrt(k), size = [M,k])
item_vectors = np.random.normal(0, 1 / np.sqrt(k), size = [N,k])

# Initialisation of history
Loss_history = []
RMSE_list = []
Loss_history_test = []
RMSE_list_test = []

# Train the model
for i in range (epoch):

    for m in range (M):
        user_biases[m] = update_user_bias(m, data_by_user_index, lamda, gamma)
        update_user_vector(m, user_biases, item_biases, user_vectors, item_vectors, data_by_user_index, lamda, tau, k)

    for n in range (N):
        item_biases[n] = update_item_bias(n, data_by_movie_index, lamda, gamma)
        update_item_vector(n, user_biases, item_biases, user_vectors, item_vectors, data_by_movie_index, lamda, tau, k)

    loss, RMSE = loss_RMSE(user_biases, item_biases, user_vectors, item_vectors, data_by_user_index, data_by_movie_index, lamda, gamma, tau)
    Loss_history.append(loss)
    RMSE_list.append(RMSE)
    
    loss_test, RMSE_test = loss_RMSE(user_biases, item_biases, user_vectors, item_vectors, data_by_user_index_test, data_by_movie_index_test, lamda, gamma, tau)
    Loss_history_test.append(loss_test)
    RMSE_list_test.append(RMSE_test)

    print(f'Epoch{i+1} ---- Loss: {loss} ---- Loss_test: {loss_test} ---- RMSE: {RMSE} ---- RMSE_test: {RMSE_test}')
