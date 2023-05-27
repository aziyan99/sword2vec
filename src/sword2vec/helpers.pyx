import numpy as np
cimport numpy as np

def softmax(u):
    """
    Computes the softmax of the given vector u.

    Parameters:
    - u: The input vector.

    Returns:
    - softmax_u: The softmax of the input vector.
    """

    exp_u = np.exp(u)
    return exp_u / np.sum(exp_u)


def loss(w_c, u):
    """
    Computes the loss function given the context word vector w_c and the output vector u.

    Parameters:
    - w_c: The context word vector.
    - u: The output vector.

    Returns:
    - loss_value: The computed loss value.
    """

    u_exp = np.exp(u)
    softmax_sum = np.sum(u_exp)
    return -np.sum(w_c) + len(w_c) * np.log(softmax_sum)


def error(out, w_c):
    """
    Computes the error between the predicted output vector and the true context word vector.

    Parameters:
    - out: The predicted output vector.
    - w_c: The true context word vector.

    Returns:
    - error_value: The computed error value.
    """

    return np.sum(np.subtract(out, w_c), axis=0)


def forward_pass(wt, w1, w2):
    """
    Performs the forward pass of the word2vec model given the input word vector wt and the weights w1 and w2.

    Parameters:
    - wt: The input word vector.
    - w1: The weights of the hidden layer.
    - w2: The weights of the output layer.

    Returns:
    - softmax_u: The softmax of the output vector u.
    - h: The hidden layer activations.
    - u: The output vector before softmax.
    """

    wt = np.array(wt)  # Convert to NumPy array
    h = np.dot(wt, w1)
    u = np.dot(h, w2)
    return softmax(u), h, u

def learning(learning_rate, w1, w2, dw1, dw2):
    """
    Performs the weight update step during the learning process.

    Parameters:
    - learning_rate: The learning rate.
    - w1: The weights of the hidden layer.
    - w2: The weights of the output layer.
    - dw1: The gradients of the hidden layer weights.
    - dw2: The gradients of the output layer weights.

    Returns:
    - w1: The updated weights of the hidden layer.
    - w2: The updated weights of the output layer.
    """

    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    return w1, w2

def backpropagation(out, word, h, w2):
    """
    Performs backpropagation to compute the gradients of the weights.

    Parameters:
    - out: The predicted output vector.
    - word: The input word and its corresponding context word vector.
    - h: The hidden layer activations.
    - w2: The weights of the output layer.

    Returns:
    - dw1: The gradients of the hidden layer weights.
    - dw2: The gradients of the output layer weights.
    """

    err = error(out, word[1]).T
    dw2 = np.outer(err, h)
    EH = np.dot(err, w2.T)
    dw1 = np.outer(EH, np.array(word[0]).T)
    return dw1.T, dw2.T

def cosine_similarity(v1, v2):
    """
    Computes the cosine similarity between two vectors.

    Parameters:
    - v1: The first vector.
    - v2: The second vector.

    Returns:
    - similarity: The cosine similarity between the two vectors.
    """
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity