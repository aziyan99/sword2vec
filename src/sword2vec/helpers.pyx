import numpy as np
cimport numpy as np

def softmax(u):
    exp_u = np.exp(u)
    return exp_u / np.sum(exp_u)


def loss(w_c, u):
    u_exp = np.exp(u)
    softmax_sum = np.sum(u_exp)
    return -np.sum(w_c) + len(w_c) * np.log(softmax_sum)


def error(out, w_c):
    return np.sum(np.subtract(out, w_c), axis=0)


def forward_pass(wt, w1, w2):
    wt = np.array(wt)  # Convert to NumPy array
    h = np.dot(wt, w1)
    u = np.dot(h, w2)
    return softmax(u), h, u

def learning(learning_rate, w1, w2, dw1, dw2):
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    return w1, w2

def backpropagation(out, word, h, w2):
    err = error(out, word[1]).T
    dw2 = np.outer(err, h)
    EH = np.dot(err, w2.T)
    dw1 = np.outer(EH, np.array(word[0]).T)
    return dw1.T, dw2.T

def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    similarity = dot_product / (norm_v1 * norm_v2)
    return similarity