import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    loss = e.T.dot(e) / y.shape[0]
    return loss.item()

#Least squares GD implementation
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    loss = 0
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
    return w, loss

#Least squares SGD implementation
def compute_stoch_gradient(y, tx, w):
    e = y - tx.dot(w)
    gradient = - (tx.T.dot(e)) / y.shape[0]
    return gradient

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    loss = 0
    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            loss = compute_loss(y_batch, tx_batch, w)
            stoch_grad = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * stoch_grad
    return w, loss

#Least squares implementation
def least_squares(y, tx):
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    e = y - tx.dot(w)
    mse = e.T.dot(e) / y.shape[0]
    return w, mse.item()