# Word2vec---Sentence-emotions-recognizer

Neural network for basic sentences classification and Emoticon assingment using pretrained Google Word2vec model.

Unzip all files to one common folder.

Download Google Word2vec dictionary file <b>GoogleNews-vectors-negative300.bin.gz</b> from
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
to the project root folder.

To retrain existing neural network ("net" folder), type in Command Prompt:

> python path_to_"train_model.py"_file

To train new neural network, delete existing "net" folder before execute above command.


To test neural network, type in Command Prompt:

> python "Recognizer.py"_file

Shape Recognizer GUI will appear. Load image and Recognize shape.



GoogleNews-vectors-negative300.bin.gz
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
