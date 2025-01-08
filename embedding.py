# Read the file movies as a dataframe
df = pd.read_csv(movies_path)

# Movie indexes
drama_indexes = [1, 2, 8, 1014, 7]
romance_indexes = [5, 3338, 3327, 3767, 3328, 10314, 3306]
children_indexes= [92, 31216, 142] 
crime_indexes = [96, 0, 924, 276]
mystery_indexes = [2012, 19722, 9914]

# Movie vectors
drama_vectors = np.array([item_vectors2D[index] for index in drama_indexes])
children_vectors = np.array([item_vectors2D[index] for index in children_indexes])
crime_vectors = np.array([item_vectors2D[index] for index in crime_indexes])
mystery_vectors = np.array([item_vectors2D[index] for index in mystery_indexes])

vectors = np.concatenate((drama_vectors, children_vectors, crime_vectors, mystery_vectors))

# Movie ids
drama_ids = [map_index_to_movie[i] for i in drama_indexes]
children_ids = [map_index_to_movie[i] for i in children_indexes]
crime_ids = [map_index_to_movie[i] for i in crime_indexes]
mystery_ids = [map_index_to_movie[i] for i in mystery_indexes]

ids = drama_ids + children_ids + crime_ids + mystery_ids

# Pot the embedding
plt.figure(figsize=(6,5))

plt.scatter(drama_vectors[:, 0], drama_vectors[:, 1], label = 'Drama', marker='1')
plt.scatter(children_vectors[:, 0], children_vectors[:, 1], label = 'children', marker='o')
plt.scatter(crime_vectors[:, 0], crime_vectors[:, 1], label = 'Crime', marker='^')
plt.scatter(mystery_vectors[:, 0], mystery_vectors[:, 1], label = 'Mystery', marker='+')

annotations = []

titles = [df.loc[df['movieId']==int(id), 'title'].values[0] for id in ids]
for i, title in enumerate(titles):
    annotations.append(plt.text(vectors[i, 0], vectors[i, 1], title, fontsize=8))

# Adjust annotations to avoid overlap
adjust_text(annotations, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

plt.tick_params(left = True, right = False , labelleft = False ,
                labelbottom = False, bottom = True)
plt.legend(fontsize=9, loc='upper left')
