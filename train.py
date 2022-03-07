import json
import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet, AdvancedNeuralNet
from nltk_utils import tokenize, stem, bag_of_words
from transformers import AutoModel, BertTokenizerFast

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
        

print(all_words)
ignore_words = ['?', '!', '.', ',']
#We don't want punctuation marks
all_words = [stem(w) for w in all_words if w not in ignore_words]

print("---------------")
print("All our words after tokenization")
print(all_words)

all_words = sorted(set(all_words))
tags = sorted(set(tags))

#Now we are creating the lists to train our data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

#Convert into a numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)

# Import BERT
bert = AutoModel.from_pretrained('bert-base-uncased')
# Load BERT tozenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

#TODO: How do these hyperparameters affect optimization of our chatbot? 
"""
Batch size determines the number of times the data is propogated (back or forwards) throughout
the model, larger batch sizes require more memory but yield better results. This parameter 
exponentially increases the runtime of our training data as the batch size increases.

The hidden size is the number of nodes in a hidden layer. This is optimized by making the hidden_size
equal or close to the number ofnodes in the input or outpus layer.

The output size is optimized here by making the output_size equal to the number or POS tags we have,
in this case it is 7 and should not be tuned any further.

The learning rate has to do with the step size in the gradient descent. A high learning rate may
cause problems with settling in a local minimum instead of the true minimum, which is why here
we have a very small learning rate to combat that issue

the num_epochs variable is the number of times the model runs. This hyperparameter is conjunctively
used in awareness of the learning rate, we have lots of epochs here because we have a very low learning
rate. If we had a larger learning rate we would decrease the number of epochs.
"""
batch_size = 8
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

input_size = len(X_train[0])
print("Below is the Input Size of our Neural Network")
print(input_size, len(all_words))
print("Below is the output size of our neural network, which should match the amount of tags ")
print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

#The below function helps push to GPU for training if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdvancedNeuralNet(bert, input_size, hidden_size, output_size).to(device)

#Loss and Optimizer

#TODO: Experiment with another optimizer and note any differences in loss of our model. Does the final loss increase or decrease? 
"""
The initial loss of the Adam optimizer was 0.0003, which is very respectable for this model. 

I knew before-hand that the SGD optimizer was outperformed by the Adam optimizer,
however I was curious to know just HOW much better it was. After experimenting
I got a final loss of 1.7631. In comparison, our results will only vary by ~ 5%, 
but it is still worth choosing Adam over SGD.
"""
#TODO CONT: Speculate on why your changed optimizer may increase or decrease final loss

"""
Adam is better comparatively due to the fact that it is essentially SGD with extra steps. 
It is a combination of RMSprop, and SGD with momentum. momentum is the ability that Adam has 
(which SGD doesn't) that helps accelerate gradients vectors in the right directon, thus leading
to faster converging.
"""
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        #Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        #backward and optimizer step 
        optimizer.zero_grad()

        #Calculate the backpropagation
        loss.backward()
        optimizer.step()

    #Print progress of epochs and loss for every 100 epochs
    if (epoch +1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

    #Need to save the data 
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete, file saved to {FILE}')
#Should save our training data to a pytorch file called "data"