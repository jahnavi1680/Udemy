import sys
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


EMBEDDING_DIM = 100
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
tokenizer = Tokenizer(num_words=200000)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
print(len(sequences[0]),len(sequences[1]),len(sequences[2]))



#adding padding
#wgats the max length of the sentences?
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_len = 100
