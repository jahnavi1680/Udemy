# Udemy
udemy RNN course code implementations

# CNN for text classification
In this section, we are going to use a 1D CNN to classify text. 
So how does a 1D CNN classify text and capture context. It has to do with the convolution and how it gets the features of surrounding worrds to effect thr features of target word.

Lets see how we do the feature engineering
oops I forgot. CNN trains itself to get the features too. But lets see how the input is formatted and given.

The convolution is expected to pass through a sentence overlapping n words at a time. Each word here is an embedding (i.e. vector of length e). The CNN will have this covolution run through all the sentences in the training set. that will be covered in training part I suppose. We are in the input stage

## First lets get the pretrained vectors
We are using GloVe. It's downloaded from http://nlp.stanford.edu/data/glove.6B.zip  
For more exploration on how this looks - glove.ipynb  
We have mapped the raw text file to a dictionory with keys as words and values as vectors.

## Now mapping our sentences to these
For that, first we need to convert the sentences into tokens first. For this, we use keras tokeniser. Which also saves vocabulary in a dictionary {'the':1,'of':2}  
Then, we want all the sentences to be of same length. So we add tokens to it as a padding (a part of keras prepocessing).  
Then, convert each sentence into an sequence matrix. shape -  (159571, 100)  

## embeddings
So our glove embeddings are word:embedding_vector. But our sequences are all numbers that tokeniser mapped to words.  

Guess what we do? We don't have to replace the sequence matrix with vectors apparently. We make embedding matrix. With all the words from vocabulary. The shape of the embedding matrix is (20000,100)  

Then theres something called embedding layer. Which I don't understand. So I'll get back to this in a few minutes. -- actually this will be clear in CNN architecture explanation

## CNN architcture
I will just write down the dimensions as the input goes through every step.  

Input - [100,] - max length of sequence
Embedding layer - So this layer does stuff that will return the embeddings of our input sequene.  
A quick detour to the embedding matrix. It contains the glove vectors of all the words in the vocabulary indexed according to their vocab index.. i.e."the" word's embeddings are in the first of the embedding matrix.  
The embedding layer gets the words 

# Simple RNNs (RNN, LSTM, GRU) for text classification
In the previous section, the teacher of the course helped with a lot of code given for text classification. I guess for RNN, we are going to have to figure out lol.  
So the code in the https://github.com/lazyprogrammer/machine_learning_examples/blob/master/nlp_class3/simple_rnn_test.py file was all that give to us. Lets use it and get the classification results by ourselves.

## Data Processing
The data that's given to the models is pretty much the same. So let's replicate from that code. (We understood pretty well).
1. Getting the pretrained weights from Glove
2. 

## 