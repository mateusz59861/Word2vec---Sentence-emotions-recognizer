import numpy as np
import pandas as pd


def sentences_to_matrix(sentences, dictionary):
    input_data = np.zeros((len(sentences), 300))
    for lines in range(len(sentences)):
        list_of_words = np.array(pd.Series(sentences[lines]).str.split().tolist())
        zdanie = np.zeros((300,))
        counter = 0

        for words in range(list_of_words.shape[1]):
            aaa = pd.Series(sentences[lines]).str.split().str[words].values
            slowo = pd.Series(aaa).to_string()

            try:
                zdanie = zdanie + dictionary[slowo[5:]]
                counter = counter + 1
            except:
                # print("Model does not contain word: ", slowo[5:])
                continue

        if counter == 0:
            continue
        else:
            zdanie = zdanie / counter

        input_data[lines][:] = zdanie
    X = np.float32(input_data)

    return X