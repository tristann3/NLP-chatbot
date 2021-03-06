import nltk 
import numpy as np
# Bottom ssl is workaround for broken script on punkt donwloadm which returns a loading ssl error
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
#End of error workaround

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
#Imports needed from nltk

#Our Tokenizer
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

#Stemming Function
def stem(word):
    return stemmer.stem(word.lower())

#Bag of Words Function
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag


"""
The purpose of tozenizing the text is to brek down the sentance into 'tokens'
so that it is easier for computers to understand. This is broken down into a list
of individual words instead of one sentance string.
"""
#Testing our Tokenizer
test_sentence = "I will not live in peace until I find the Avatar!"
print(tokenize(test_sentence))

#TODO: Test our Stemming function on the below words. 
#TODO CONT: How does stemming affect our data?
words = ["Organize", "organizes", "organizing", "disorganized"]
for word in words:
    print(stem(word))

"""
Stemming cleans word data by identifying similar patterns. This is to combat
nearly-identical words but of which are sightly different due to past-tense, 
present participle, etc. It cleans it by chopping off part of it, making the 
3 "slightly different" words, identical. this affects our data outside of the 
word by helping the Bag of Words function cut down on its scope!
"""


#TODO: Implement the above Bag of Words function on the below sentence and words. 
#TODO (CONTINUED): What does the Bag of Words model do? Why would we use Bag of Words here instead of TF-IDF or Word2Vec?
print("Testing our bag_of_words function")
sentence = ["I", "will", "now", "live", "in", "peace", "until", "I", "find", "the", "Avatar"]
words = ["hi", "hello", "I", "you", "the", "bye", "in", "cool", "wild", "find"]
print(bag_of_words(sentence, words))
print("--------------")

"""
The Bag of Words function transforms the tokenized sentence into a matrix. 
Data shows either a 1 or a 0 in which tht word is present or not. Bag of 
Words is useful to us because it can show how frewuent words are used in our corpus.

We use Bag of Words over TF-IDF and Word2Vec because of the cimplicity of this task. 
We have a very small corpus. TF-IDF is able to ignore filler words, which we have little to none.
Word2Vec is a system to identify similar words with similar meaning.
"""


