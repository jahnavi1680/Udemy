# https://deeplearningcourses.com/c/deep-learning-advanced-nlp
import sys
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE= 20000

# first, get the pretrained weights
#We have mapped the raw text file to a dictionory with keys as words and values as vectors.

word2vec = {}
with open(os.path.join('../data/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word2vec[word] = coefs
print('Found %s word vectors.' % len(word2vec))
    # or
    # content = f.read()
from tensorflow.keras.preprocessing.text import Tokenizer
# load in the training data
df_train = pd.read_csv("../data/text_class/train.csv")
sequences = df_train['comment_text'].fillna("DUMMY_VALUE").values
print(len(sequences))
# create a tokenizer to vectorize the text
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sequences)
# convert the text to sequences of integers
sequences = tokenizer.texts_to_sequences(sequences)
# create a word index mapping
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#padding the sequences to the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)  # shape - (159571, 100) - hundred is the length of sequence

#prepare the embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i < MAX_VOCAB_SIZE:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all zeros.
            embedding_matrix[i] = embedding_vector
print('Shape of embedding matrix:', embedding_matrix.shape)  # (20000, 100)

#model building starts here
from tensorflow.keras.models import Sequential



'''from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.models import Model
from keras.layers import Input, LSTM, GRU
import numpy as np
import matplotlib.pyplot as plt

try:
  import keras.backend as K
  if len(K.tensorflow_backend._get_available_gpus()) > 0:
    from keras.layers import CuDNNLSTM as LSTM
    from keras.layers import CuDNNGRU as GRU
except:
  pass


T = 8
D = 2
M = 3


X = np.random.randn(1, T, D)


def lstm1():
  input_ = Input(shape=(T, D))
  rnn = LSTM(M, return_state=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  o, h, c = model.predict(X)
  print("o:", o)
  print("h:", h)
  print("c:", c)


def lstm2():
  input_ = Input(shape=(T, D))
  rnn = LSTM(M, return_state=True, return_sequences=True)
  # rnn = GRU(M, return_state=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  o, h, c = model.predict(X)
  print("o:", o)
  print("h:", h)
  print("c:", c)


def gru1():
  input_ = Input(shape=(T, D))
  rnn = GRU(M, return_state=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  o, h = model.predict(X)
  print("o:", o)
  print("h:", h)


def gru2():
  input_ = Input(shape=(T, D))
  rnn = GRU(M, return_state=True, return_sequences=True)
  x = rnn(input_)

  model = Model(inputs=input_, outputs=x)
  o, h = model.predict(X)
  print("o:", o)
  print("h:", h)



print("lstm1:")
lstm1()
print("lstm2:")
lstm2()
print("gru1:")
gru1()
print("gru2:")
gru2()'''