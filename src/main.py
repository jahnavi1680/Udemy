import sys
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE= 20000


# load in pre-trained word vectors
print('Loading word vectors...')
word2vec = {}
with open(os.path.join('../data/glove.6B/glove.6B.%sd.txt' % EMBEDDING_DIM)) as f:
  # is just a space-separated text file in the format:
  # word vec[0] vec[1] vec[2] ...
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec
print('Found %s word vectors.' % len(word2vec))

df_train = pd.read_csv("../data/text_class/train.csv")
print(len(df_train))

sentences = df_train['comment_text'].fillna("DUMMY_VALUE").values
# convert the sentences (strings) into integers
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

# pad the sequences to the same length
from tensorflow.keras.preprocessing.sequence import pad_sequences
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape) #shape - (159571,100) -  hundred is the length of sequence

# get word -> integer mapping
word2idx = tokenizer.word_index
print(word2idx['hello'])

# prepare embedding matrix
print('Filling pre-trained embeddings...')
num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word2idx.items():
  if i < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      # words not found in embedding index will be all zeros.
      embedding_matrix[i] = embedding_vector
print('Shape of embedding matrix:', embedding_matrix.shape)  # (20000, 100)

