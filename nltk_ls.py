import numpy as np
import nltk
#Snowball stemmer
from nltk.stem.snowball import EnglishStemmer
stemmer = EnglishStemmer()
def tokenize(sentence):
    
    return nltk.word_tokenize(sentence)


def stem(word):
  
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
   
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, n in enumerate(words):
        if n in sentence_words: 
            bag[idx] = 1

    return bag