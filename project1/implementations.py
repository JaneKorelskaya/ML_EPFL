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
#MSE loss
def compute_loss(y, tx, w):
    e = y - tx.dot(w)
    loss = e.T.dot(e) / 2 / y.shape[0]
    return loss.item()

#MSE gradient
def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    gradient = - (tx.T.dot(e)) / y.shape[0]
    return gradient

#Least squares GD implementation
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    w = initial_w
    loss = compute_loss(y, tx, w)

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

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    
    # Define parameters to store w and loss
    w = initial_w
    loss = compute_loss(y, tx, w)
    batch_size = 1
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size):
            loss = compute_loss(y_batch, tx_batch, w)
            stoch_grad = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * stoch_grad
    return w, loss

#Least squares implementation
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    e = y - tx.dot(w)
    mse = e.T.dot(e) / y.shape[0]
    return w, mse.item()

#Ridge regression implementetion
def ridge_regression(y, tx, lambda_):
    """Ridge regression using gradient descent algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        lambda_: scalar

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    lambda_1 = lambda_ * (2 * y.shape[0])
    w = np.linalg.solve(tx.T@tx + lambda_1 * np.identity(tx.T.shape[0]), tx.T@y)
    loss = compute_loss(y, tx, w)
    return w, loss

#Logistic regression GD

def sigmoid(t):
    return 1 / (1 + np.exp(-t))

#Negative log likelihood
def calculate_loss(y, tx, w):
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    p = sigmoid(tx@w)
    loss = - np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) 
    return np.float64(loss)

#Gradient of logloss
def calculate_gradient(y, tx, w):
    gradient = (1 / tx.shape[0]) * tx.T@(sigmoid(tx@w) - y)
    return gradient

#One step of gradient descent using logistic regression
def learning_by_gradient_descent(y, tx, w, gamma):
    loss = calculate_loss(y, tx, w)
    w = w - gamma * calculate_gradient(y, tx, w)
    return loss, w


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    # init parameters
    threshold = 1e-8
    # build tx
    #tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = initial_w
    loss = calculate_loss(y, tx, w)

    losses = []

    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss

#Logistic regression with regularization

#Compute loss and gradient
def penalized_logistic_regression(y, tx, w, lambda_):
    loss = calculate_loss(y, tx, w) + lambda_*np.linalg.norm(w)
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient
#Do one step of gradient descent, using the penalized logistic regression.
def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient
    return loss, w

def reg_logistic_regression(y, tx, lambda_ ,initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        lambda_: scalar
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    
    # init parameters
    threshold = 1e-8
    losses = []

    # build tx
    #tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = initial_w
    loss = calculate_loss(y, tx, w)

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss
