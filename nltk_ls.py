import numpy as np
import nltk
#Snowball stemmer
from nltk.stem.snowball import EnglishStemmer
stemmer = EnglishStemmer()#snowball includes different language stemmer but we need english stemmer
def tokenize(sentence):#tokenize
    
    return nltk.word_tokenize(sentence) #split the sentence


def stem(word):#analysing root of the word
  
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
   
    # stem each word
    sentence_words = []
    for word in tokenized_sentence:
        sentence_words.append(stem(word))
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, n in enumerate(words):
        if n in sentence_words: 
            bag[idx] = 1

    return bag