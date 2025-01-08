# Import the libraries
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the paths of the data 
path_ratings = '/home/andor/Desktop/ml-25m/ratings.csv'
path_movies = '/home/andor/Desktop/ml-25m/movies.csv'

# Organize the data following the order user_id, movie_id, rating, timestamp
ratings = []
with open(path_ratings, 'r') as ratings_file:
    data_reader = csv.reader(ratings_file, delimiter=',')
    next(data_reader, None)
    for row in data_reader:
        user_id, movie_id, rating, timestamp = row
        ratings.append([user_id, movie_id, rating, timestamp])

# Looking at the rating distribution
rating_dist = [row[2] for row in ratings]
rating_dist.sort()

plt.hist(rating_dist, bins=len(set(rating_dist)))

plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.title('Rating distribution')

rating_dist = [row[2] for row in ratings]
rating_dist.sort()


# Plotting a basic histogram
plt.hist(rating_dist, bins=len(set(rating_dist)))

# Adding labels and title
plt.xlabel('Ratings')
plt.ylabel('Frequency')
plt.title('Rating distribution')

# A function that sparse the file ratings 
def sparse(ratings):

    map_user_to_index={}
    map_index_to_user=[]
    map_movie_to_index={}
    map_index_to_movie=[]

    data_by_user_index = []
    data_by_movie_index = []

    for row in ratings:
      #User mapping
      user_name = row[0]
      movie_name = row[1]
      rating = row[2]
      if user_name not in map_user_to_index:
        map_user_to_index[user_name]=len(map_user_to_index)
        map_index_to_user.append(user_name)
        data_by_user_index.append([])

      #Movie mapping
      if movie_name not in map_movie_to_index:
        map_movie_to_index[movie_name]=len(map_movie_to_index)
        map_index_to_movie.append(movie_name)
        data_by_movie_index.append([])
      #Fill data_by_user_index and data_by_movie_index
      data_by_user_index[map_user_to_index[user_name]].append((map_movie_to_index[movie_name],rating))
      data_by_movie_index[map_movie_to_index[movie_name]].append((map_user_to_index[user_name],rating))
    sparse = map_user_to_index, map_index_to_user, map_movie_to_index, map_index_to_movie, data_by_user_index, data_by_movie_index
    return sparse

map_user_to_index, map_index_to_user, map_movie_to_index, map_index_to_movie, data_by_user_index, data_by_movie_index=sparse(ratings)

# Visualize the power law in the data 
movies_by_user = [len(user_list) for user_list in data_by_user_index]
user_per_movie = [len(movie_list) for movie_list in data_by_movie_index]

fig, ax = plt.subplots(figsize = (5, 4))

ax.scatter(movies_by_user, [movies_by_user.count(number) for number in movies_by_user], marker = '+', label = 'Movie')
ax.scatter(user_per_movie, [user_per_movie.count(number) for number in user_per_movie], marker = 'v', label = 'User')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Degree')
plt.ylabel('Frequencies')
plt.legend()
plt.title('Power law')
