import numpy as np

from sentences_to_matrix import sentences_to_matrix
from softmax import softmax


def predict(sentences, Weights, bias, dictionary):
    # Convert sentences to vectors [matrix for multiple sentences]
    X = sentences_to_matrix(sentences, dictionary);

    # Return class 5 (not found) when none of words was recognized
    if np.array_equal(np.zeros((1, 300)), X):
        return [5]

    # Define number of training examples
    m = X.shape[0]

    pred_label = np.arange(X.shape[0])
    for i in range(m):  # Loop over the training examples
        z = np.dot(Weights, X[i]) + bias
        a = softmax(z)
        pred_label[i] = np.argmax(a)

    return pred_label