import numpy as np
import pickle
import math
import sys
import time


def linear_backward(dZ, cache):
    A_prev, W, _ = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True) 
    dZ[Z <= 0] = 0
    return dZ

def softmax_backward(Y, AL):
    dZ = AL - Y
    return dZ

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    AL_clipped = np.clip(AL, 1e-8, 1 - 1e-8)
    dAL = - (np.divide(Y, AL_clipped) - np.divide(1 - Y, 1 - AL_clipped))
    current_cache = caches[L-1]
    grads["dZ" + str(L)] = softmax_backward(Y, AL_clipped)
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(grads["dZ" + str(L)], current_cache[0])
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        linear_cache, activation_cache = current_cache
        dZ = relu_backward(grads["dA" + str(l + 1)], activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db
    return grads

def compute_cost(AL, Y):
    m = Y.shape[1]
    epsilon = 1e-8
    cost = -np.sum(Y * np.log(AL + epsilon)) / m
    cost = np.squeeze(cost)
    return cost

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters

def initialize_layer(n_input, n_output):
    np.random.seed(2)
    W = np.random.randn(n_output, n_input) * 0.01
    b = np.zeros((n_output, 1))
    return {"W": W, "b": b}

def layer_forward(A_prev, parameters, activation):
    W, b = parameters['W'], parameters['b']
    Z = np.dot(W, A_prev) + b
    linear_cache = (A_prev, W, b)
    if activation == "sigmoid":
        A = 1/(1 + np.exp(-Z))
    elif activation == "relu":
        A = np.maximum(0, Z)
    elif activation == "softmax":
        shift_Z = Z - np.max(Z, axis=0, keepdims=True)
        exps = np.exp(shift_Z)
        sum_exps = np.sum(exps, axis=0, keepdims=True)
        A= exps / sum_exps
    else:
        raise ValueError("Unsupported activation function")
    activation_cache = Z
    cache = (linear_cache, activation_cache)
    return A, cache

def initialize_network(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A 
        A, cache = layer_forward(A_prev, {"W": parameters['W'+str(l)], "b": parameters['b'+str(l)]}, activation='relu')
        caches.append(cache)
    AL, cache = layer_forward(A, {"W": parameters['W'+str(L)], "b": parameters['b'+str(L)]}, activation='softmax')
    caches.append(cache)
    return AL, caches

def train(X, Y, parameters, learning_rate=0.0075, num_iterations=3000, print_cost=True):
    costs = []
    ttime=[]
    for i in range(0, num_iterations):
        T1=time.time()
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        print(time.time()-T1)
        ttime.append(time.time()-T1)
        if print_cost and i % 10 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
    return parameters,costs,ttime

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot_encode(Y, C):
    m = Y.shape[0]
    Y_one_hot = np.zeros((C, m))
    Y_one_hot[Y, np.arange(m)] = 1
    return Y_one_hot

def read_data(metadata_path):
    metadata = unpickle(metadata_path + 'meta')
    superclass_dict = dict(list(enumerate(metadata[b'coarse_label_names'])))
    data_pre_path = metadata_path
    data_train_path = data_pre_path + 'train'
    data_test_path = data_pre_path + 'test'
    data_train_dict = unpickle(data_train_path)
    data_test_dict = unpickle(data_test_path)
    return data_train_dict,data_test_dict

def pre_processing(data_train_dict,data_test_dict):
    X_train = data_train_dict[b'data']/255
    Y_train = one_hot_encode(np.array(data_train_dict[b'fine_labels']),100)
    X_test = data_test_dict[b'data']/255
    Y_test = one_hot_encode(np.array(data_test_dict[b'fine_labels']),100)    
    return X_train,Y_train,X_test,Y_test

def write_costs_to_file(costs, file_name):
    with open(file_name, 'w') as f:
        for cost in costs:
            f.write(f"{cost}\n")

def save_parameters(parameters, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(parameters, file)

def load_parameters(file_name):
    with open(file_name, 'rb') as file:
        parameters = pickle.load(file)
    return parameters

def update_parameters_with_sgd(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters

def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

def train_with_sgd(X, Y, parameters, learning_rate=0.0075, num_iterations=3000, mini_batch_size=64, print_cost=True):
    np.random.seed(1)
    costs = []
    ttime=[]
    for i in range(0, num_iterations):
        T1=time.time()
        mini_batches = random_mini_batches(X, Y, mini_batch_size)
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y, caches)
            parameters = update_parameters_with_sgd(parameters, grads, learning_rate)
        print(time.time()-T1)
        ttime.append(time.time()-T1)
        if print_cost and i % 10 == 0 or i == num_iterations - 1:
            print(f"Cost after iteration {i}: {cost}")
            costs.append(cost)
    return parameters, costs,ttime

def initialize_velocity(parameters):
    L = len(parameters) // 2
    v = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    return v

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1 - beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1 - beta) * grads["db" + str(l+1)]
        parameters["W" + str(l+1)] -= learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * v["db" + str(l+1)]
    return parameters, v

def train_with_sgdm(X, Y, parameters, learning_rate=0.01, beta=0.9, num_iterations=1000, mini_batch_size=64, print_cost=True):
    np.random.seed(1)
    costs = []
    ttime=[]
    velocities = initialize_velocity(parameters)
    for i in range(num_iterations):
        T1=time.time()
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed=i)
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y, caches)
            parameters, velocities = update_parameters_with_momentum(parameters, grads, velocities, beta, learning_rate)
        print(time.time()-T1)
        ttime.append(time.time()-T1)
        if print_cost and i % 10 == 0:
            print(f"Cost after epoch {i}: {cost}")
            costs.append(cost)
    return parameters, costs,ttime

def train_with_sgdmGC(X, Y, parameters, learning_rate=0.01, beta=0.9, num_iterations=1000, mini_batch_size=64, print_cost=True):
    np.random.seed(1)
    costs = []
    ttime=[]
    velocities = initialize_velocity(parameters)
    for i in range(num_iterations):
        T1=time.time()
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed=i)
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y, caches)
            grads = gradient_centralization(grads)
            grads = clip_gradients(grads, max_value=1.0)
            parameters, velocities = update_parameters_with_momentum(parameters, grads, velocities, beta, learning_rate)
        print(time.time()-T1)
        ttime.append(time.time()-T1)
        if print_cost and i % 10 == 0:
            print(f"Cost after epoch {i}: {cost}")
            costs.append(cost)
    return parameters, costs,ttime

def initialize_rmsprop(parameters):
    L = len(parameters) // 2
    S = {}
    for l in range(L):
        S["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        S["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    return S

def update_parameters_with_rmsprop(parameters, grads, S, learning_rate, beta=0.9, epsilon=1e-8):
    L = len(parameters) // 2
    for l in range(L):
        S["dW" + str(l+1)] = beta * S["dW" + str(l+1)] + (1 - beta) * np.square(grads["dW" + str(l+1)])
        S["db" + str(l+1)] = beta * S["db" + str(l+1)] + (1 - beta) * np.square(grads["db" + str(l+1)])
        parameters["W" + str(l+1)] -= (learning_rate * grads["dW" + str(l+1)]) / (np.sqrt(S["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] -= (learning_rate * grads["db" + str(l+1)]) / (np.sqrt(S["db" + str(l+1)]) + epsilon)
    return parameters, S

def update_parameters_with_rmspropgc(parameters, grads, S, learning_rate, beta=0.9, epsilon=1e-8):
    gc_grads=gradient_centralization(grads)
    gc_grads = clip_gradients(grads, max_value=1.0)
    return update_parameters_with_rmsprop(parameters, gc_grads, S, learning_rate, beta, epsilon)

def train_with_rmsprop(X, Y, parameters, learning_rate=0.01, beta=0.9, num_iterations=1000, mini_batch_size=64, print_cost=True):
    np.random.seed(1)
    costs = []
    epsilon=1e-8
    S = initialize_rmsprop(parameters)
    ttime=[]
    for i in range(num_iterations):
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed=i)
        T1=time.time()
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y, caches)
            parameters, S = update_parameters_with_rmsprop(parameters, grads, S, beta, learning_rate,epsilon)
        print(time.time()-T1)
        ttime.append(time.time()-T1)
        if print_cost and i % 10 == 0:
            print(f"Cost after epoch {i}: {cost}")
            costs.append(cost)
    return parameters, costs,ttime

def train_with_rmspropgc(X, Y, parameters, learning_rate=0.01, beta=0.9, num_iterations=1000, mini_batch_size=64, print_cost=True):
    np.random.seed(1)
    costs = []
    epsilon=1e-8
    S = initialize_rmsprop(parameters)
    ttime=[]
    for i in range(num_iterations):
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed=i)
        T1=time.time()
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y, caches)
            parameters, S = update_parameters_with_rmspropgc(parameters, grads, S, beta, learning_rate,epsilon)
        print(time.time()-T1)
        ttime.append(time.time()-T1)
        if print_cost and i % 10 == 0:
            print(f"Cost after epoch {i}: {cost}")
            costs.append(cost)
    return parameters, costs,ttime

def gradient_centralization(grads):
    gc_grads = {}
    for key, grad in grads.items():
        if 'dW' in key:
            grad_mean = np.mean(grad, axis=0, keepdims=True)
            gc_grads[key] = grad - grad_mean
        else:
            gc_grads[key] = grad
    return gc_grads

def update_parameters_with_sgdm(parameters, grads, velocities, learning_rate, beta):
    for l in range(len(parameters) // 2):
        velocities['dW' + str(l+1)] = beta * velocities['dW' + str(l+1)] + (1 - beta) * grads['dW' + str(l+1)]
        velocities['db' + str(l+1)] = beta * velocities['db' + str(l+1)] + (1 - beta) * grads['db' + str(l+1)]
        parameters['W' + str(l+1)] -= learning_rate * velocities['dW' + str(l+1)]
        parameters['b' + str(l+1)] -= learning_rate * velocities['db' + str(l+1)]
    return parameters, velocities

def clip_gradients(grads, max_value=1.0):
    for key in grads.keys():
        np.clip(grads[key], -max_value, max_value, out=grads[key])
    return grads

def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        v["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate,beta1, beta2, epsilon):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * np.square(grads["dW" + str(l+1)])
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * np.square(grads["db" + str(l+1)])
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)
        parameters["W" + str(l+1)] -= learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] -= learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
    return parameters, v, s

def update_parameters_with_adamgc(parameters, grads, v, s, t, learning_rate,beta1, beta2, epsilon):
    gc_grads = gradient_centralization(grads)    
    gc_grads = clip_gradients(grads, max_value=1.0)
    return update_parameters_with_adam(parameters, gc_grads, v, s, t, learning_rate,beta1, beta2, epsilon)

def train_with_adam(X, Y, parameters, learning_rate=0.0001, num_iterations=100, mini_batch_size=64, beta1=0.9, beta2=0.999, epsilon=1e-8, print_cost=True):
    np.random.seed(1)
    costs = []
    ttime=[]
    v, s = initialize_adam(parameters)
    t = 0
    for i in range(num_iterations):
        seed = i  
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        T1=time.time()
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y, caches)
            t += 1
            parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
        print(time.time()-T1)
        ttime.append(time.time()-T1)
        if print_cost and i % 10 == 0 or i == num_iterations - 1:
            print(f"Cost after epoch {i}: {cost}")
            costs.append(cost)
    return parameters, costs,ttime

def train_with_adamgc(X, Y, parameters, learning_rate=0.0001, num_iterations=100, mini_batch_size=64, beta1=0.9, beta2=0.999, epsilon=1e-8, print_cost=True):
    np.random.seed(1)
    costs = []
    ttime=[]
    v, s = initialize_adam(parameters)
    t = 0
    for i in range(num_iterations):
        seed = i  
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        T1=time.time()
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y, caches)
            t += 1
            parameters, v, s = update_parameters_with_adamgc(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
        print(time.time()-T1)
        ttime.append(time.time()-T1)
        if print_cost and i % 10 == 0 or i == num_iterations - 1:
            print(f"Cost after epoch {i}: {cost}")
            costs.append(cost)
    return parameters, costs,ttime

def initialize_adagrad(parameters):
    L = len(parameters) // 2
    s = {}
    for l in range(L):
        s["dW" + str(l+1)] = np.zeros_like(parameters["W" + str(l+1)])
        s["db" + str(l+1)] = np.zeros_like(parameters["b" + str(l+1)])
    return s

def update_parameters_with_adagrad(parameters, grads, s, learning_rate=0.01, epsilon=1e-8):
    L = len(parameters) // 2
    for l in range(L):
        s["dW" + str(l+1)] += np.square(grads["dW" + str(l+1)])
        s["db" + str(l+1)] += np.square(grads["db" + str(l+1)])
        parameters["W" + str(l+1)] -= (learning_rate * grads["dW" + str(l+1)]) / (np.sqrt(s["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] -= (learning_rate * grads["db" + str(l+1)]) / (np.sqrt(s["db" + str(l+1)]) + epsilon)
    return parameters, s

def update_parameters_with_adagrad_gc(parameters, grads, s, learning_rate=0.01, epsilon=1e-8):
    gc_grads = gradient_centralization(grads)
    gc_grads = clip_gradients(grads, max_value=1.0)
    return update_parameters_with_adagrad(parameters, gc_grads, s, learning_rate, epsilon)

def train_with_adagrad(X, Y, parameters, learning_rate=0.01, num_iterations=100, mini_batch_size=64, print_cost=True):
    np.random.seed(1)
    costs = []
    s = initialize_adagrad(parameters)
    for i in range(num_iterations):
        seed = i
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y, caches)
            parameters, s = update_parameters_with_adagrad(parameters, grads, s, learning_rate)
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after epoch {i}: {cost}")
            costs.append(cost)
    return parameters, costs

def train_with_adagradgc(X, Y, parameters, learning_rate=0.01, num_iterations=100, mini_batch_size=64, print_cost=True):
    np.random.seed(1)
    costs = []
    s = initialize_adagrad(parameters)
    for i in range(num_iterations):
        seed = i
        mini_batches = random_mini_batches(X, Y, mini_batch_size, seed)
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y, caches)
            parameters, s = update_parameters_with_adagrad_gc(parameters, grads, s, learning_rate)
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after epoch {i}: {cost}")
            costs.append(cost)
    return parameters, costs

def train_with_sgd_gc(X, Y, layers_dims, learning_rate=0.01, num_iterations=100, mini_batch_size=64, print_cost=True):
    np.random.seed(1)
    costs = []
    parameters = initialize_network(layers_dims)
    for i in range(num_iterations):
        mini_batches = random_mini_batches(X, Y, mini_batch_size, i)
        for mini_batch in mini_batches:
            (mini_batch_X, mini_batch_Y) = mini_batch
            AL, caches = L_model_forward(mini_batch_X, parameters)
            cost = compute_cost(AL, mini_batch_Y)
            grads = L_model_backward(AL, mini_batch_Y, caches)
            gc_grads = gradient_centralization(grads)
            gc_grads = clip_gradients(grads, max_value=1.0)
            parameters = update_parameters_with_sgd(parameters, gc_grads, learning_rate)
        if print_cost and (i % 100 == 0 or i == num_iterations - 1):
            print(f"Cost after epoch {i}: {cost}")
            costs.append(cost)
    return parameters, costs


data_train_dict,data_test_dict=read_data("Data/cifar-100-python/")
X_train,Y_train,X_test,Y_test=pre_processing(data_train_dict,data_test_dict)

layer_dims = [3072, 512, 100]  
parameters = initialize_network(layer_dims)
optimizer = input("Enter the optimizer name (gd,sgd, sgdm, rmsprop, adagrad, adam,sgdgc, sgdmgc, rmspropgc, adagradgc, adamgc): ").lower()
costs=[]
ttime=[]
if optimizer == "gd":
    parameters,costs,ttime = train(X_train.T, Y_train, parameters, learning_rate=0.1, num_iterations=250, print_cost=True)
elif optimizer == "sgd":
    parameters,costs,ttime = train_with_sgd(X_train.T, Y_train, parameters, learning_rate=0.1, num_iterations=250, print_cost=True)
elif optimizer == "sgdgc":
    parameters,costs,ttime = train_with_sgd_gc(X_train.T, Y_train, parameters, learning_rate=0.1, num_iterations=250, print_cost=True)
elif optimizer == "sgdm":
    parameters,costs,ttime = train_with_sgdm(X_train.T, Y_train, parameters, learning_rate=0.1, beta=0.9, num_iterations=250, mini_batch_size=10, print_cost=True)
elif optimizer == "sgdmgc":
    parameters,costs,ttime = train_with_sgdmGC(X_train.T, Y_train, parameters, learning_rate=0.1, beta=0.8, num_iterations=100, mini_batch_size=10, print_cost=True)
elif optimizer == "rmsprop":
    parameters,costs,ttime = train_with_rmsprop(X_train.T, Y_train, parameters, learning_rate=0.1, beta=0.9, num_iterations=250, mini_batch_size=10, print_cost=True)
elif optimizer == "rmspropgc":
    parameters,costs,ttime = train_with_rmspropgc(X_train.T, Y_train, parameters, learning_rate=0.1, beta=0.9, num_iterations=250, mini_batch_size=10, print_cost=True)
elif optimizer == "adam":
    parameters,costs,ttime = train_with_adam(X_train.T, Y_train, parameters, learning_rate=0.1, num_iterations=100, mini_batch_size=10, beta1=0.9, beta2=0.999, epsilon=1e-8, print_cost=True)
elif optimizer == "adamgc":
    parameters,costs,ttime = train_with_adamgc(X_train.T, Y_train, parameters, learning_rate=0.1, num_iterations=100, mini_batch_size=10, beta1=0.9, beta2=0.999, epsilon=1e-8, print_cost=True)
elif optimizer == "adagrad":
    parameters,costs,ttime = train_with_adagrad(X_train.T, Y_train, parameters, learning_rate=0.1, num_iterations=250, mini_batch_size=10, print_cost=True)
elif optimizer == "adagradgc":
    parameters,costs,ttime = train_with_adagradgc(X_train.T, Y_train, parameters, learning_rate=0.1, num_iterations=100, mini_batch_size=10, print_cost=True)
else:
    print("choose proper optimizer")
    sys.exit("Invalid Input")
save_parameters(parameters, f'Output/my_model_parameters_{optimizer}.pkl')
write_costs_to_file(costs, f'Output/training_costs_{optimizer}.txt')
write_costs_to_file(ttime, f'Output/training_time_{optimizer}.txt')

print(parameters)
print(costs)
print(ttime[:10])
