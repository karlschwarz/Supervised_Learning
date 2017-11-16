import h5py
import numpy as np
import tensorflow as tf
import math
from tensorflow.python.framework import ops
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    C = tf.constant(C, name = "C")
    one_hot_matrix = tf.one_hot(labels, C, axis = 0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    
    return one_hot

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape = [n_x, None])
    Y = tf.placeholder(tf.float32, shape = [n_y, None])
    
    return X, Y

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)
    for l in range(1, L):
        wl = tf.get_variable('W' + str(l), [layer_dims[l], layer_dims[l - 1]], 
                                                   initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        bl = tf.get_variable('b' + str(l), [layer_dims[l], 1], 
                                                  initializer = tf.zeros_initializer())
        parameters['W' + str(l)] = wl
        parameters['b' + str(l)] = bl
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    
    return Y

def linear_forward(A, W, b):
    Z = tf.add(tf.matmul(W, A), b)
    
    return Z

def linear_activation_forward(A_prew, W, b, activation):
    
    if activation == 'sigmoid':
        Z = linear_forward(A_prew, W, b)
        A = tf.sigmoid(Z)
    elif activation == 'relu':
        Z = linear_forward(A_prew, W, b)
        A = tf.nn.relu(Z)  
        
    return A

def L_model_forward(X, parameters):
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                      activation = 'relu')
    
    ZL = linear_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
    AL = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = 'sigmoid')
    #assert(ZL.shape[1] == X.shape[1])
    
    return ZL, AL

def compute_cost(ZL, Y):
    logits = tf.transpose(ZL)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    
    return cost

def regulization_term(parameters):
    L = len(parameters) // 2
    l2_loss = 0
    for l in range(L):
        WL = parameters['W' + str(l + 1)]
        l2_loss = l2_loss + tf.nn.l2_loss(WL)
    
    return l2_loss

def L_layer_model(X_train, Y_train, X_test, Y_test, layers_dims, learning_rate = 0.0075, lamb = 0, 
                  num_epochs = 1500, minibatch_size = 32, print_cost = False, output_para = False, plot = True):
    
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters_deep(layers_dims)
    ZL, AL = L_model_forward(X, parameters)
    l2_loss = regulization_term(parameters)
    cost = compute_cost(ZL, Y) + lamb * regulization_term(parameters)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for  epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost == True and epoch % 100 == 0:
                print('Cost after epoch %i: %f' % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        if plot == True:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("learning_rate = " + str(learning_rate))
            plt.show()
            
        parameters = sess.run(parameters)
        correct_prediction = tf.equal(tf.argmax(AL), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    
        print('Train Accuracy:', accuracy.eval({X: X_train, Y: Y_train}))
        print('Test Accuracy:', accuracy.eval({X: X_test, Y: Y_test}))
    
        if output_para == True:          
            print('Parameters have been trained!')
            return parameters

def forward_propagation_for_predict(X, parameters):
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A
        A = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                      activation = 'relu')
    
    AL = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = 'sigmoid')
    #assert(ZL.shape[1] == X.shape[1])
    
    return AL

def predict(X, parameters):
    L = len(parameters) // 2
    [n_x, m] = X.shape
    params = {}
    for l in range(L):
        Wl = tf.convert_to_tensor(parameters['W' + str(l + 1)])
        bl = tf.convert_to_tensor(parameters['b' + str(l + 1)])
        params['W' + str(l + 1)] = Wl
        params['b' + str(l + 1)] = bl
    x = tf.placeholder('float', shape = [n_x, m])
    zl = forward_propagation_for_predict(x, params)
    p = tf.argmax(zl)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
    
    return prediction

def pd2ary(X_train_origin, Y_train_origin):
    X_train_ary = np.array(X_train_origin)
    Y_train_ary = np.array([Y_train_origin]).T
    
    return X_train_ary, Y_train_ary

def produce_train_data(X_train_origin, Y_train_origin, n_splits):
    from sklearn.model_selection import KFold
    import numpy as np
    kf = KFold(n_splits = n_splits, shuffle = True)
    X_subs_train, Y_subs_train, X_subs_test, Y_subs_test = [], [], [], []
    X_train_ary, Y_train_ary = pd2ary(X_train_origin, Y_train_origin)
    n = X_train_ary.shape[1]
    
    for train_index, test_index in kf.split(X_train_ary):
        X_sub_train, Y_sub_train = X_train_ary[train_index], Y_train_ary[train_index]
        X_sub_test, Y_sub_test = X_train_ary[test_index], Y_train_ary[test_index]
        X_subs_train.append(X_sub_train)
        Y_subs_train.append(Y_sub_train)
        X_subs_test.append(X_sub_test)
        Y_subs_test.append(Y_sub_test)
        
    for i in range(n_splits):
        X_subs_train[i] = X_subs_train[i].T
        Y_subs_train[i] = one_hot_matrix(((Y_subs_train[i]).T)[-1], 2)
        X_subs_test[i] = X_subs_test[i].T
        Y_subs_test[i] = one_hot_matrix(((Y_subs_test[i]).T)[-1], 2)
    
    assert(X_subs_train[0].shape[0] == n) 
    # assert(Y_subs_train[0].shape[0] == 1y_subs_train[0])
    assert(X_subs_train[0].shape[1] == Y_subs_train[0].shape[1])
    
    return X_subs_train, Y_subs_train, X_subs_test, Y_subs_test

def L_model_validation(X_train, Y_train, X_test, Y_test, layers_dims = [12, 10, 5, 2], num_epochs = 500, learning_rate = 0.0075, lamb = 0, minibatch_size = 32):
 
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters_deep(layers_dims)
    ZL, AL = L_model_forward(X, parameters)
    l2_loss = regulization_term(parameters)
    cost = compute_cost(ZL, Y) + lamb * regulization_term(parameters)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for  epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
        parameters = sess.run(parameters)
        correct_prediction = tf.equal(tf.argmax(AL), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))  
        tr_acy = accuracy.eval({X: X_train, Y: Y_train})
        tst_acy = accuracy.eval({X: X_test, Y: Y_test})
        
        return tr_acy, tst_acy

def Lmodel_kfold(X_train_origin, Y_train_origin, n_splits, paras):
    combs = len(paras)
    results = {}
    for i in range(combs):
        learning_rate = paras[i][0]
        lamb = paras[i][1]
        X_subs_train, Y_subs_train, X_subs_test, Y_subs_test = produce_train_data(X_train_origin, Y_train_origin, n_splits)
        tr_acy_list, tst_acy_list = [], []
        for j in range(n_splits):
            tr_acy, tst_acy = L_model_validation(X_subs_train[j], Y_subs_train[j], X_subs_test[j], Y_subs_test[j],
                                                learning_rate = learning_rate, lamb = lamb)
            tr_acy_list.append(tr_acy)
            tst_acy_list.append(tst_acy)
        results[str(i + 1)] = (np.mean(tr_acy_list), np.mean(tst_acy_list))
    
    return results  

def raw2train(X_train_origin, Y_train_origin, num_features = 2, train_size = 0.8):
    X_train_origin = X_train_origin.as_matrix()
    Y_train = Y_train_origin.as_matrix()
    X_train, X_test, Y_train, Y_test = train_test_split(X_train_origin, Y_train_origin,
                                                    train_size = train_size, test_size = (1 - train_size),
                                                    random_state = 0)
    X_train, X_test = X_train.T, X_test.T
    Y_test = one_hot_matrix(Y_test, 2)
    Y_train = one_hot_matrix(Y_train, 2)
    
    return X_train, X_test, Y_train, Y_test

def L_model_train(X_train, Y_train, layers_dims, learning_rate = 0.0075, lamb = 0, 
                  num_epochs = 1500, minibatch_size = 32, print_cost = True, output_para = True, plot = True):
    
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters_deep(layers_dims)
    ZL, AL = L_model_forward(X, parameters)
    l2_loss = regulization_term(parameters)
    cost = compute_cost(ZL, Y) + lamb * regulization_term(parameters)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for  epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost == True and epoch % 100 == 0:
                print('Cost after epoch %i: %f' % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        if plot == True:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("learning_rate = " + str(learning_rate))
            plt.show()
            
        parameters = sess.run(parameters)
        correct_prediction = tf.equal(tf.argmax(AL), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    
        print('Train Accuracy:', accuracy.eval({X: X_train, Y: Y_train}))
    
        if output_para == True:          
            print('Parameters have been trained!')
            return parameters

def Lmodel_hypersearch(X_train_origin, Y_train_origin, n_splits, num_iters = 1000):
    combs = len(paras)
    results = {}
    tst_acy_prev = 0
    for i in range(num_iters):
        num_exp = -4 * np.random.rand()
        learning_rate = 10 ** num_exp
        lamb = np.random.rand() / 1000
        X_subs_train, Y_subs_train, X_subs_test, Y_subs_test = produce_train_data(X_train_origin, Y_train_origin, n_splits)
        tr_acy_list, tst_acy_list = [], []
        for j in range(n_splits):
            tr_acy, tst_acy = L_model_validation(X_subs_train[j], Y_subs_train[j], X_subs_test[j], Y_subs_test[j],
                                                learning_rate = learning_rate, lamb = lamb)
            tr_acy_list.append(tr_acy)
            tst_acy_list.append(tst_acy)
        tr_acy = np.mean(tr_acy_list)
        tst_acy = np.mean(tst_acy_list)
        if tst_acy > tst_acy_prev:
            tst_acy_prev = tst_acy
            results[str(learning_rate) + ',' + str(lamb)] = (tr_acy, tst_acy_prev)
            
    return results  

def Lmodel_hypersearch_v2(X_train, Y_train, X_test, Y_test, layers_dims = [12, 10, 5, 2], num_iters = 1000, threshold = 0.78, num_epochs = 500):
    results = {}
    tst_acy_prev = threshold
    for i in range(num_iters):
        num_exp = -4 * np.random.rand()
        learning_rate = 10 ** num_exp
        lamb = np.random.rand() / 1000
        tr_acy_list, tst_acy_list = [], []
        tr_acy, tst_acy = L_model_validation(X_train, Y_train, X_test, Y_test, layers_dims = layers_dims, learning_rate = learning_rate, lamb = lamb, num_epochs = num_epochs)
        tr_acy_list.append(tr_acy)
        tst_acy_list.append(tst_acy)
        tr_acy = np.mean(tr_acy_list)
        tst_acy = np.mean(tst_acy_list)
        if tst_acy > tst_acy_prev:
            tst_acy_prev = tst_acy
            results[str(learning_rate) + ', ' + str(lamb)] = (tr_acy, tst_acy)
            
    return results  

def Lmodel_hypersearch_v3(X_train, Y_train, X_test, Y_test, layers_dims = [12, 10, 5, 2], lr = 0, lb = 0, sig1 = 0.0001, sig2 = 0.0001, num_iters = 1000, threshold = 0.82, num_epochs = 500):
    results = {}
    tst_acy_prev = threshold
    for i in range(num_iters):
        num_exp = -4 * np.random.rand()
        learning_rate = sig1 *  np.random.rand() + lr
        lamb = sig2 * np.random.rand() + lb
        tr_acy_list, tst_acy_list = [], []
        tr_acy, tst_acy = L_model_validation(X_train, Y_train, X_test, Y_test, layers_dims = layers_dims, learning_rate = learning_rate, lamb = lamb, num_epochs = num_epochs)
        tr_acy_list.append(tr_acy)
        tst_acy_list.append(tst_acy)
        tr_acy = np.mean(tr_acy_list)
        tst_acy = np.mean(tst_acy_list)
        if tst_acy > tst_acy_prev:
            tst_acy_prev = tst_acy
            results[str(learning_rate) + ', ' + str(lamb)] = (tr_acy, tst_acy)
            
    return results  
