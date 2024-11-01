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
