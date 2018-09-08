# Word2vec---Sentence-emotions-recognizer

Neural network for basic sentences classification and Emoticon assingment using pretrained Google Word2vec model.

Unzip all files to one common folder. Then download Google Word2vec dictionary file <b>GoogleNews-vectors-negative300.bin.gz</b> from
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
to the project root folder.

To train neural network, first create a dictionary and prepare input data by running <i>create_dictionary.py</i> script. Type in Command Prompt:

> python path_to_"create_dictionary.py"_file

Execution of above script may take several minutes. It creates dictionary from word2vec model, but using only words found in input data. That allows to training network without loading whole dictionary into memory. To train neural network, type in Command Prompt:

> python path_to_"train_emotions.py"_file

To test neural network, type in Command Prompt:

> python path_to_"Recognizer.py"_file

Emoticon Recognizer GUI will appear. Type your sentence and <b>Recognize emotion</b>. Now you can test how neural network is working without Word2vec model, basing only on words found in input data. To load Word2vec model, click <b>Load word2vec model</b>.


![alt text](https://datascience-enthusiast.com/figures/image_1.png)
