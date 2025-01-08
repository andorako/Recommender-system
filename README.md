# Movie recomendation system using MovieLens dataset. 

This project implements a collaborative filtering-based recommendation system using matrix factorization. 
The system predicts movie ratings and provides recommendations based on past user interactions with movies.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Code Structure](#code-structure)

## 1. Overview 

This project uses data from the MovieLens dataset to create a movie recommender system. To reproduce the code,
first download the data named 'ml-25m.zip' [here](https://grouplens.org/datasets/movielens/).

## 2. Installation 

To run this project, you'll need the following Python packages:

- `numpy`
- `matplotlib`
- `csv`
- `pandas`

## 3. Code structure 

The code structure is divided into three main parts which include data organization and visualization, training of the model with the data, prediction and the visualizatoin of the embedding of the item vectors for a selection of movies. 

### 3.1. Data visualization and organization

The code should begin by this part, it allows to make a visualization of the data and preprocess it to be ready for the training. All the necessary libraries and also the data are imported in this section, none of the other codes will work if this part is skipped. It can be found in the section 'data.py'.

### 3.2. Training of the model with the data

In this part, the training of the model using the data is done. It contains all the functions needed for the training and the training itself. It's the main part of the code. The parameters are set by default with the optimal parameters. At the end, a visualization of the loss for the training data and the RMSE for both the training and validation set can be done. 

### 3.3. Prediction

Before the prediction, a training of the model with the whole data must be done. To make that, change every variable named `data_by_user_index_train` and `data_by_movie_index_train` respectively into `data_by_user_index` and `data_by_movie_index`. Then use the vectors and biases derived from this training with the code in the file 'predict.py'.

## 3.4. Embedding

To visualize the embedding of a selection of movie vectors in 2D. A training like in the prediction section must be done again with `k=2`. The movie vectors derived from that should be named 'item_vectors2D' and used with the code in the file 'embedding.py'.  
