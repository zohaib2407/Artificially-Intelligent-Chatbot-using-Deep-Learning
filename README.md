# Artificially-Intelligent-Chatbot-using-Deep-Learning

## Objective
We built two different chatbots that will simulate human conversation through texts. We have explored two broader categories of chatbots:

#### Scripted chatbot
The chatbot will be trained on an existing dataset using a neural network model and once deployed, it will respond to the posed questions with the pre-determined scripts that are created and stored in their library. In general, these are simple chatbots that highly depend on user input. If customer queries fall outside the pre-defined rules, these chatbots fall short of recognizing conversation context and won’t be able to identify advanced scenarios.

#### Artificially Intelligent Chatbot
Artificial intelligence chatbots, on the other hand, use natural language processing (NLP) and Deep Neural Networks to understand the intent behind the question and solve the customer’s problem without any human assistance. The chatbot is trained on an existing dataset using seq-to-seq/BERT model and for the posed questions it will generate new answers (that are not there in answer corpus) based on the index of the vocabulary that it is trained on. The bot tries to mimic human-like traits and responses.

## Dataset 
We plan to use the two different datasets based for the two different types of chatbots. 

#### The Yahoo! Question-Answer Dataset
We used an open-source Yahoo’s ‘Question-answers’ dataset for training the Artificially Intelligent chatbot for both the categories explained in previous section. The idea is to first, enable the bot to search for the best answer from the pre-determined script. This will be for the similar set of questions that are present in Yahoo’s dataset. In the second part of the project plan, we will try and train the bot in a way that based on the similarity scores for the questions, it will be able to mimic human-like responses and come up with unique responses. This data we have collected is a subset of the Yahoo! Answers corpus from a 10/25/2007 dump. The data contains 142,627 questions and their answers. In addition to question-and-answer text, the corpus contains a small number of metadata, i.e., which answer was selected as the best answer, and the category and sub-category that was assigned to this question. No personal information is included in the corpus. The size of this dataset is 104 MB.

#### Intents Dataset 
This dataset will be used for training the Scripted Chatbot. The intents dataset is formulated around ‘Question-Answer’ pairs in a JSON format. There are multiple pairs of different questions and answers that are grouped under a single tag. The multiple tags in the dataset are like ‘greeting’, ‘goodbye’, ‘thanks’ etc. Under each tag, we have a set of similar questions and a set of similar answers corresponding to those questions. The model will be trained on different pairs of ‘tags’ and ‘responses. 

## Scripted Chatbot

#### What is a Scripted Chatbot?
Scripted chatbots are retrieval based chatbots that are trained on an existing dataset using a neural network model and once deployed, it will respond to the posed questions with the pre-determined scripts that are created and stored in their library. The retrieval based chatbots uses the search and random pick methodology after being trained for interaction. Once trained, user interacts with the bot using a graphical interface. The question/statement posed by the user is first undergoes preliminary cleaning steps such as tokenization, stemming, indexing etc. After that, the model classifies to which tag does the question belong to and then it randomly picks the response from the responses under the corresponding tag and hence, it is called a retrieval based chatbot.

#### Dataset Info
For training this chatbot we will use our intents dataset. The intents dataset is formulated around ‘Question-Answer’ pairs in a JSON format. There are multiple pairs of different questions and answers that are grouped under a single tag. The multiple tags in the dataset are like ‘greeting’, ‘goodbye’, ‘thanks’ etc. Under each tag, we have a set of similar questions and a set of similar answers corresponding to those questions.

#### Pre- processing 
We performed various pre-processing of the data before our model could be trained on it. A few of the steps that were performed are tokenization, lemmatization, and removing duplicate words from the list. Next, we created our training data in form of input-output pairs. Our inputs are different statement of the ‘patterns’ tag and outputs are the class our input pattern (tags) belongs to. Next, we have created a vocabulary and loaded our query/response pair into memory. Since we are dealing with sequences of words, which do not have an implicit mapping to a discrete numerical space, we have created one by mapping each unique word that we have encountered in our dataset to an index value.

#### Model Architecture and Training
To build a model, we used a specific architecture of the neural network using the Keras API. We used a deep neural network of 3 layers. The first layer has 128 neurons with input size equal to the length of the training dataset and ReLu activation. The second layer has 64 neurons with ReLu activation and the third layer is the output layer which contains number of neurons equal to number of intents to predict output intent with SoftMax operation. There are also two dropout layers in between the layers. 
Next, we used Stochastic Gradient descent for optimizing our model with ‘cross_entropy’ loss. We trained our model for 200 epochs (since we have a small training data) and achieved 100% accuracy.

#### Result and Prediction
The user interacts with the bot using a Graphical User Interface which is built using the ‘tkinter’ package from python. For prediction on user responses, the trained model is loaded and then using the graphical user interface the predicted response from the bot is displayed to the user. The output of the model tells us the class the user query belongs to, so we have implemented some functions which will identify the class and then retrieve us a random response from the list of responses.

![image](https://user-images.githubusercontent.com/35283246/163801216-2e570d22-7be2-4845-b282-c4928f84042a.png)





