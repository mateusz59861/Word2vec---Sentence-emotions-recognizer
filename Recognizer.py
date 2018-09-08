import numpy as np
import tkinter as tk
from tkinter import StringVar
from PIL import Image, ImageTk
import pickle
import gensim

from predict import predict


# ----FUNCTIONS----

def load_image(path_to_image):
    im = Image.open(path_to_image)
    im = im.resize((412, 412))
    im = ImageTk.PhotoImage(im)
    image_label.configure(image=im)
    image_label.image = im


def recognize(event=None):
    msg = sentence.get()
    sentence.set("") # Clears input field.
    text_label.config(text=msg)
    my_input = [msg.strip().lower()]
    try:
        if word2vec_loaded:
            prediction = predict(my_input, W, b, model)
        else:
            prediction = predict(my_input, W, b, dict)
        load_image("images/" + str(prediction[0]) + ".png")
    except:
        load_image("images/5.png")


def load_word2vec():
    # Load pretrained word2vec model
    global model
    model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
    global word2vec_loaded
    word2vec_loaded = True
    load_word2vec_button.config(text='Word2vec model loaded', state='disabled')
    load_image("images/ready.png")
    text_label.config(text='Word2vec model loaded. Type your sentence:')



window = tk.Tk()
window.title('Emotion Recognizer')
window.geometry('420x540')
window.resizable(False, False)

sentence = StringVar()

# ----FRAMES----
image_frame = tk.Frame(window, bg='black', relief='groove', height=420, width=420, padx=2, pady=2)
image_frame.grid(column=0, row=0, sticky="ew")
image_frame.grid_propagate(False)

button_answer_frame = tk.Frame(window, bg='green',  height=100, width=420)
button_answer_frame.grid(column=0, row=1 ,columnspan=3)
#button_answer_frame.grid_propagate(False)

# ----LABELS-----
image_label = tk.Label(image_frame, bg='black')
image_label.grid(column=0, row=0, sticky="ew")

text_label = tk.Label(button_answer_frame, width=60, height=1, text="Type your sentence:", bg="grey")
text_label.grid(row=0, column=0, sticky="ew")

# ----ENTRY TEXT----
recognize_text = tk.Entry(button_answer_frame, width=60, textvariable=sentence)
recognize_text.grid(column=0, row=1, sticky="ew")

# ----BUTTON----
load_word2vec_button = tk.Button(button_answer_frame, text='Load word2vec model (This may take several minutes)', height=2, width=60, command=load_word2vec, bg="grey")
load_word2vec_button.grid(column=0, row=3)

recognize_button = tk.Button(button_answer_frame, text='Recognize emotion', height=2, width=60, command=recognize, bg="grey")
recognize_button.grid(column=0, row=2)
window.bind('<Return>', recognize)

model = {}
word2vec_loaded = False

# Load Dictionary
dict = pickle.load(open("data/word_dict.p", "rb"))

# Load Weights and bias
W = np.load('weights.npy')
b = np.load('bias.npy')

window.mainloop()
