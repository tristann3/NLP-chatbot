# Welcome to the A.I. Chatbot Companion Project!

---

This project is a summative assessment of your computer science, deep learning, and natural language processing capabilities.

This file is your **single-source-of-truth** for accessing relevant challenges and tasks required to complete the Chatbot Companion project.

Please read through this document exhaustively and make sure you have a fundamental understanding of the range and type of tasks you're required to complete for successful project completion.

## PROGRESSIVE CHALLENGES

The first major benchmark of successful project performance is being able to run the project successfully and understand the basics of its architecture and hierarchy.

**Navigate to `train.py` and `nltk_utils.py`: answer all relevant _TODOs_ to complete the first part of the project.**

## ANALYTIC OBJECTIVES

Once you've successfully ran and interpreted your project once, you'll perform some additional challenges flexing both your working knowledge of natural language processing as well as your experience in software engineering and computer science.

## OBJECTIVE 1: Designing Attention

The major part of this project is the implementation of a powerful neural network that can capably learn sentiment and patterns across input text.

Now that you have an understanding of higher-level neural networks and attentive models, let's go ahead and evaluate your ability to implement them in code.

Navigate to the `model.py` file and take note of the construction of the `NeuralNet()` object.

Create a new object called `AdvancedNeuralNet()` and redesign the neural network architecture using your knowledge of advanced deep learning. 

More specifically, create an advanced neural network architecture that incorporates **the attention layer** as a major part of the algorithm – you may flexibly use other higher-order neural network architecture choices as well, such as imported pretraining modules, additional optimizers, and other RNN-friendly tools. 

(Remember to update the relevant scripts in `chat.py` and `train.py` such that you make use of your attention networks: simply replace `from model import NeuralNet` with `from model import AdvancedNeuralNet` and update relevant importations/calls as needed.)

While this can and will increase training costs, it will more-than-likely lead to significant improvements in accuracy determination. 

Create a file called `reports.txt`; save your accuracy/loss reports (as produced by your terminal/notebook) for both your default `NeuralNet()` as well as your custom `AdvancedNeuralNet()`. 

If done correctly, you should be left with an additional `reports.txt` file that contains at least two (2) new blocks of text representing accuracy/loss evaluations for two separate model configurations.

## OBJECTIVE 2: Hyperparameter Optimization

At this point in your deep learning and data science journey, you should be no stranger to performing hyperparameter tuning and optimization to improve the performance of a learning algorithm.

Since you should have already addressed a `TODO` challenge on specific type of hyperparameters for your model training, now you'll take the chance to actually do it! 

Navigate to the `train.py` file and perform **at least four (4)** different hyperparameter tunings, changing up discretized values within reason.

Be sure to access relevant documentation for NLTK, PyTorch, and/or TensorFlow to assure you have an understanding of what each hyperparameter represents, or even what other hyperparameters you can include that aren't already tracked/provided.

Create a file called `tuning.txt`; after running each of your tuning sessions, save each of the final accuracy/loss performance reports (as produced by your terminal/notebook) as text within the logging file. 

(Remember to identify which hyperparameters you've chosen for each specific logged case; you may even want to extend your code programmatically to generate text that tracks configured hyperparameters and actually outputs all information to a file without manual oversight.)

If done correctly, you should be left with an additional `tuning.txt` file that contains at least four (4) new blocks of text representing accuracy/loss evaluations for discretized hyperparameter tuning scenarios.

## OBJECTIVE 3: Dataset Augmentation

The data that we use for this project is represented by the structure in `intents.json`. 

This can be an odd data structure to get used to. 

`intents.json` effectively encodes our data as representative nested patterns of acceptable input-output phrases.

To better understand the dataset, let's take the time to extend our dataset and add additional input-output patterns and phrases to beef up our model!

Navigate to `intents.json` and add additional nested data in the form of new **tags**, **patterns**, and **responses**. 

By default, there are seven discrete groups of intents available grouped by `tag` (greetings, goodbyes, thanks, items, payments, deliveries, funny) – your job is to add _at least three (3)_ additional groups of intents. 

(The specific phrase pairings and types of intents are totally up to you!)

If done correctly, you should have an updated `intents.json` file with at least ten (10) total groupings of intents, allowing for more flexile data interpretation from our algorithm.

## OBJECTIVE 4: Companionization

Our chatbot works relatively well as an internal Terminal script - however, chatbots are generally designed to work in platforms such as web clients and software like Slack or Discord.

While that's not a focus of this project, it is important for applied NLP to understand what the process of scaling algorithmic technology looks like!

Create and navigate to a new file called `scaling.txt` – in this file, _write at least one (1) paragraph_ describing what the process would look like to deploy this chatbot to a social media site or software platform of your choice.

Be as specific as you can be without diving into programmatic implementation: feel free to look up tutorials and walkthroughs online to get a better picture of how to go about this process.

If done correctly, you should have a new file called `scaling.txt` that effectively serves as a short quickstart guide to taking this project and deploying/extending it to a live platform for companionized use. 