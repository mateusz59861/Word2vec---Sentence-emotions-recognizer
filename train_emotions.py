import numpy as np
from sklearn.model_selection import train_test_split
import pickle

from sentences_to_matrix import sentences_to_matrix
from softmax import softmax
from predict import predict


# ----FUNCTIONS----

def train_model(train_data_content, train_labels, dictionary, learning_rate=0.01, num_iterations=400):
    np.random.seed(1)

    # Convert sentences to vectors [matrix for multiple sentences]
    X = sentences_to_matrix(train_data_content, dictionary);

    # Convert labels to one-hot vectors
    Y = labels_to_onehot(train_labels, 5)

    # Define number of training examples
    m = Y.shape[0]  # number of training examples
    n_y = 5  # number of classes
    n_h = 300  # dimensions of the Word2vec vectors

    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    pred_label = np.arange(Y.shape[0])

    # Optimization loop
    for t in range(num_iterations):  # Loop over the number of iterations
        for i in range(m):  # Loop over the training examples
            # Forward propagate the avg through the softmax layer
            z = np.dot(W, X[i]) + b
            a = softmax(z)
            pred_label[i] = np.argmax(a)

            # Compute cost using the i'th training label's one hot representation and the output of the softmax
            cost = -np.sum(Y[i] * np.log(a))

            # Compute gradients
            dz = a - Y[i]
            dW = np.dot(dz.reshape(n_y, 1), X[i].reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 100 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))

    return pred_label, W, b


def labels_to_onehot(labels, number_of_classes):
    one_hot = np.zeros((labels.shape[0], number_of_classes))

    for label in range(labels.shape[0]):
        one_hot[label, int(labels[label])] = 1

    return one_hot


# Load Dictionary
dict = pickle.load(open("data/word_dict.p", "rb"))

# Load Labels
labels = np.genfromtxt('data/labels.txt', dtype='int')

# Load Sentences
with open('data/sentences.txt') as f:
    content = f.readlines()
content = [x.strip().lower() for x in content]

# split data
train_data, test_data, train_labels, test_labels = train_test_split(content, labels, test_size=0.1, random_state=42)

# train model
print("========== TRAIN ==========")
pred, W, b = train_model(train_data, train_labels, dict, learning_rate=0.1, num_iterations=400)

# display accuracy - TRAIN
accuracy = 0
for i in range(pred.shape[0]):
    if pred[i] == train_labels[i]:
        accuracy = accuracy + 1
    else:
        print("Overshoot prediction {}: {} => {}".format(i, train_data[i], pred[i]))

print("TRAIN Accuracy: {}".format(accuracy / pred.shape[0]))
print("Correct Labels: : {}/{}".format(accuracy, pred.shape[0]))

# test model
print("========== TEST ==========")
predicted_labels = predict(test_data, W, b, dict)

# display accuracy - TEST
accuracy = 0
for i in range(predicted_labels.shape[0]):
    if predicted_labels[i] == test_labels[i]:
        accuracy = accuracy + 1
    else:
        print("Overshoot prediction {}: {} => {}".format(i, test_data[i], predicted_labels[i]))

print("TEST Accuracy: {}".format(accuracy / predicted_labels.shape[0]))
print("Correct Labels: : {}/{}".format(accuracy, predicted_labels.shape[0]))

# test model - ALL
print("========== TEST ALL ==========")
predicted_labels = predict(content, W, b, dict)

# display accuracy - ALL
accuracy = 0
for i in range(predicted_labels.shape[0]):
    if predicted_labels[i] == labels[i]:
        accuracy = accuracy + 1
    else:
        print("Overshoot prediction {}: {} => {}".format(i, content[i], predicted_labels[i]))

print("ALL Accuracy: {}".format(accuracy / predicted_labels.shape[0]))
print("Correct Labels: : {}/{}".format(accuracy, predicted_labels.shape[0]))

# save training parameters to files
np.save('data/weights.npy', W)
np.save('data/bias.npy', b)
