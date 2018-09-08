import numpy as np

softmax = lambda x: np.exp(x) / np.sum(np.exp(x))