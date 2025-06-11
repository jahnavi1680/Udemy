# Udemy
udemy RNN course code implementations

# CNN for text classification
In this section, we are going to use a 1D CNN to classify text. 
So how does a 1D CNN classify text and capture context. It has to do with the convolution and how it gets the features of surrounding worrds to effect thr features of target word.

Lets see how we do the feature engineering
oops I forgot. CNN trains itself to get the features too. But lets see how the input is formatted and given.

The convolution is expected to pass through a sentence overlapping n words at a time. Each word here is an embedding (i.e. vector of length e). The CNN will have this covolution run through all the sentences in the training set. that will be covered in training part I suppose. We are in the input stage

## First lets get the pretrained vectors.
We are using GloVe. It's downloaded from http://nlp.stanford.edu/data/glove.6B.zip  
For more exploration on how this looks - glove.ipynb  
We have mapped the raw text file to a dictionory with keys as words and values as vectors.

## Now mapping our sentences to these


