import gensim
import numpy as np
import pickle


# Read input csv file
input_csv = np.genfromtxt('input/Emotify.csv', dtype='str', delimiter=',')

# save sentences to a file
sentences = input_csv[:, :1]
np.savetxt('data/sentences.txt', sentences, delimiter=" ", fmt="%s")

# save labels to a file
labels = input_csv[:, 1]
np.savetxt('data/labels.txt', labels, delimiter=" ", fmt="%s")

# Prepare list of words to dictionary
f = open('data/sentences.txt', 'r')
fq = f.read().lower().split()
words = np.unique(fq)

# Load pretrained word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)

# Prepare dictionary
dict = {}
for x in range(words.shape[0]):
    try:
        dict[words[x]] = model[words[x]]
    except:
        print("Model does not contain word: ", words[x])


# Save dictionary to a file
pickle.dump(dict, open("data/word_dict.p", "wb"))