import numpy as np
import pickle

def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        if l < L:
            A = relu(Z)
        cache = (A_prev, W, b, Z)
        caches.append(cache)
    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    ZL = np.dot(WL, A) + bL
    AL = softmax(ZL)
    cache = (A, WL, bL, ZL)
    caches.append(cache)
    return AL, caches

def predict(X, parameters):
    AL, cache = forward_propagation(X, parameters)
    predictions = (AL > 0.5).astype(int)  # assuming binary classification
    return predictions

def accuracy(predictions, Y):
    return (predictions == Y).mean()

def load_parameters(file_name):
    with open(file_name, 'rb') as file:
        parameters = pickle.load(file)
    return parameters

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot_encode(Y, C):
    m = Y.shape[0]
    Y_one_hot = np.zeros((C, m))
    Y_one_hot[Y, np.arange(m)] = 1
    return Y_one_hot

def read_data(metadata_path):
    data_pre_path = metadata_path
    data_train_path = data_pre_path + 'train'
    data_test_path = data_pre_path + 'test'
    data_train_dict = unpickle(data_train_path)
    data_test_dict = unpickle(data_test_path)
    return data_train_dict, data_test_dict

def pre_processing(data_train_dict, data_test_dict):
    X_train = data_train_dict[b'data'] / 255
    Y_train = one_hot_encode(np.array(data_train_dict[b'fine_labels']), 100)
    X_test = data_test_dict[b'data'] / 255
    Y_test = one_hot_encode(np.array(data_test_dict[b'fine_labels']), 100)
    return X_train, Y_train, X_test, Y_test

data_train_dict, data_test_dict = read_data("Data/cifar-100-python/")
X_train, Y_train, X_test, Y_test = pre_processing(data_train_dict, data_test_dict)


optimizer = input("Enter the optimizer name (sgd, sgdm, rmsprop, adagrad, adam): ").lower()

# Load parameters for both normal and GC version of the optimizer
parameters_normal = load_parameters(f'Output/my_model_parameters_{optimizer}.pkl')
parameters_gc = load_parameters(f'Output/my_model_parameters_{optimizer}GC.pkl')

# Predictions for both normal and GC version
predictions_normal = predict(X_test.T, parameters_normal)
predictions_gc = predict(X_test.T, parameters_gc)

# Accuracy calculation for both
acc_normal = accuracy(predictions_normal, Y_test)
acc_gc = accuracy(predictions_gc, Y_test)

print(f"Accuracy with {optimizer.upper()}: {acc_normal}")
print(f"Accuracy with {optimizer.upper()}GC: {acc_gc}")
