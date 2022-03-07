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


#TODO: Test our function with the below sentence to visualize Tokenization. 
#TODO CONT: What is the purpose of tokenizing our text? 
#Testing our Tokenizer
test_sentence = "I will not live in peace until I find the Avatar!"
print(tokenize(test_sentence))

#TODO: Test our Stemming function on the below words. 
#TODO CONT: How does stemming affect our data?
words = ["Organize", "organizes", "organizing", "disorganized"]


#TODO: Implement the above Bag of Words function on the below sentence and words. 
#TODO (CONTINUED): What does the Bag of Words model do? Why would we use Bag of Words here instead of TF-IDF or Word2Vec?
print("Testing our bag_of_words function")
sentence = ["I", "will", "now", "live", "in", "peace", "until", "I", "find", "the", "Avatar"]
words = ["hi", "hello", "I", "you", "the", "bye", "in", "cool", "wild", "find"]
print(bag_of_words(sentence, words))
print("--------------")


