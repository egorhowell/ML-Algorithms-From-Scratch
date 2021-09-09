
# import packages
import numpy as np

print('This is a neural network from scratch using one hidden layer')


# define the layer size for the network
def layer_size(X, Y):
    
    input_size = X.shape[0]
    hidden_size = 3
    output_size = Y.shape[0]
    
    return input_size, hidden_size, output_size

# initialise the weights and biases randomnly to enable symmetry breaking
def initialise_parameters(input_size, hidden_size, output_size):

    W1 = np.random.randn(hidden_size, input_size) * 0.01
    W2 = np.random.randn(output_size, hidden_size) * 0.01
    b1 = np.zeros((hidden_size, 1))
    b2 = np.zeros((output_size, 1))

    # store in dict for access
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

# sigmoid
def sigmoid(z):
    return 1/(1+np.exp(-z))

# derivative
def d_sigmoid(z):
    return z*(1-z)

# forward prop
def forward(X, parameters):

    # get parameters from dict
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    # calculate the forward using the activation function
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)

    # store the values in a dict
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    
    return A2, cache

# cross entropy loss function
def compute_cost(A2, Y):
    
    cost = (-1/Y.shape[1]) * np.sum(Y*np.log(A2) + (1-Y) * np.log(1-A2))
    cost = float(np.squeeze(cost))
    
    return cost

# backprop
def backprop(parameters, cache, X, Y):

    m = X.shape[1]
    
    # get parameters from dict in cache
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    
    # compute gradients
    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2,A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1,keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1,X.T)
    db1 = (1/m) *  np.sum(dZ1, axis=1,keepdims=True)
    
    # store grads in dict
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    
    return grads

# update the parameters
def update_params(parameters, grads, alpha = 1):

    # fetch grads, weights and biases from dicts
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # update them 
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2

    # store in dict
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters

# define the model
def model(X, Y, hidden_size, epochs = 1000):

    # initiate layer size
    np.random.seed(3)
    input_size = layer_size(X, Y)[0]
    output_size = layer_size(X, Y)[2]

    # intialse the weights
    parameters = initialise_parameters(input_size, hidden_size, output_size)

    # train the model
    for i in range(0, epochs):
         
        A2, cache = forward(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backprop(parameters, cache, X, Y)
        parameters = update_params(parameters, grads)
        
        # Print the cost every 1000 iterations
        print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


# training data
X = np.array([[0,0,1,1],
              [0,1,1,1],
              [1,0,1,1],
               [1,1,1,1]])

# expected output
Y = np.array([[0],[1],[1],[1]])


# apply the model
parameters = model(X, Y, hidden_size = 4, epochs = 1000)


# predict the output
def predict(parameters, X):

    A2, cache = forward(X, parameters) 
    
    return np.round(A2,3)



print('The output is:') 
print(predict(parameters,X))
print('The expected output:')
print(Y)


