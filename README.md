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


Algorithm used to train neural network:
![alt text](https://datascience-enthusiast.com/figures/image_1.png)
1. Split sentences into words
2. Convert words to vectors using word2vec model
3. Computr average vector from all vectors in sentence - this will be an input to a network
4. Convert emoticon label to one-hot vector - this will be an output
5. Compute output by just executing dot product of weights and input vector
6. Normalize output using softmax function
7. Compute cost function and weights and bias gradients
8. Update weights and bias

Data used to build neural model consists of 188 sentences. Test data rate was 10%, wchich is 19 sentences, and 169 trainig sentences.

Final results:
- Train data - 100% accuracy
- Test data - 84.2% accuracy - 3 overshoot predictions
- All data - 98.4% accuracy - 3 overshoot predictions

![Alt text](/images/table.PNG?raw=true "Optional Title")
