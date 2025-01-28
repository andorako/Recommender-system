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

# Initialisation of the parameters
k = 20
lamda = 0.002
gamma = 0.5
tau = 0.01
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

# Get the history
for i in range (epoch):

    for m in range (M):
        user_biases[m] = update_user_bias(m, data_by_user_index_train, lamda, gamma)
        update_user_vector(m, user_biases, item_biases, user_vectors, item_vectors, data_by_user_index_train, lamda, tau, k)

    for n in range (N):
        item_biases[n] = update_item_bias(n, data_by_movie_index_train, lamda, gamma)
        update_item_vector(n, user_biases, item_biases, user_vectors, item_vectors, data_by_movie_index_train, lamda, tau, k)

    loss, RMSE = loss_RMSE(user_biases, item_biases, user_vectors, item_vectors, data_by_user_index_train, data_by_movie_index_train, lamda, gamma, tau)
    Loss_history.append(loss)
    RMSE_list.append(RMSE)

    loss_test, RMSE_test = loss_RMSE(user_biases, item_biases, user_vectors, item_vectors, data_by_user_index_test, data_by_movie_index_test, lamda, gamma, tau)
    Loss_history_test.append(loss_test)
    RMSE_list_test.append(RMSE_test)

    print(f'Epoch{i+1} ---- Loss: {loss} ---- Loss_test: {loss_test} ---- RMSE: {RMSE} ---- RMSE_test: {RMSE_test}')

# Plot the loss for the training data 
plt.figure(figsize=(5,4))
plt.plot(Loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss for the training Data')
plt.tight_layout()

# Plot the RMSE for the training and the test data 
plt.figure(figsize=(5,4))
plt.plot(RMSE_list, label = 'RMSE train')
plt.plot(RMSE_list_test, label = 'RMSE test')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('RMSE for training and test Data')
plt.legend()
