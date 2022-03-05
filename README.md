# Probabilistic A.I Chatbot

> The following is an implementation of a Natural Language Processor in the form of a *ChatBot* which takes in a user's question or comment, and, implementing common Natural Language Processor techniques as well as probability, returns an answer

<h2> Tech Stack </h2>
This program is written in Python and employs several critical libraries and packages, including:

- Pytorch
  - An open sourced Machine Learning library used for natural language processing
- N.L.T.K (Natural Language Toolkit) 
  - A bundle of libraries for English language processing 
  
<h2> How It Works </h2>

**Tokenizing** 
- The first function in preparing our user input for interpretability is a Tokenizer, which breaks a sentence down into it's individual words

**Stemming**
- Next, we apply a stemming function, which further breaks down each word into it's root without affixes or suffixes
  > This allows a better, more consistent interpretation

**Bag-Of-Words**
- Further divide's our stemmed and tokenized words into occurences within a given text. This allows a computer to interpret information processed as language into fathomable, quantifiable vectors and numbers
  > Specifically, for each occurence of a word, a value of 1 is given at the index where the word occurs, otherwise a value of 0 is given

After our text is broken down into it's most basic, interpretable form, we are able to train our model on the text, and apply a probability metric to display specific responses

**Model**
- Our model is a simple neural network consistent of two linear "feed forward" layers

<h2> Deployment </h2>
In order to deploy this project, simply download the code to your local I.D.E. and

```
python3 train.py

```

This will deploy all required training 

```
python3 chat.py
```

This will deploy the chatbot in the terminal with realistic responses within any given context!

<h2> Future </h2>

In the future, we will implement a greater amount of hidden layers, as well as an increased amount of text located within the .json file on which to train and broaden the capability of the chatbot



  
 
