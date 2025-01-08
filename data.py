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
