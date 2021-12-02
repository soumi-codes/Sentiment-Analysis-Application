# Sentiment-Analysis-Application

This prototype predicts the sentiments of textual comments and feedbacks to deduce the satisfaction of the customers or the experience they have had. Natural Language Processing techniques and Machine Learning algorithm are used to classify the sentimenet of the reviews using python libraries. It can take custom user input and classify whether the given feedback or review in textual context is positive or negative.

#### Dataset used
The data source used for this project is a resturant review dataset in tsv format taken from Kaggle website. The dataset is converted to csv format and then used in the python script.

#### Preprocessing
The python libraries used in the project are pandas, re, nltk and Scikit-learn. Regular expression is used to remove all characters other than alphabets. Then the text is converted to lower case and the stopwords are removed. Lemmatization is performed on each word to convert the different forms of a word into a base form of the word (lemma). After pre-processing, a corpus of processed data is obtained.

#### Creating Word Cloud & BoW Model 
A word cloud then is created to get some insight about the processed data and represent it in a way such that the size of each word will indicate its frequency. The bag of words model is created that converts the text data into vectors and describes the total occurrence of words within a document. Bigram parameter is used to give importance to the combination of words. 

#### Classfication
The corpus is split into training and testing sets in a 75:25 ratio. A Support Vector Machine classifier model is created, trained and evaluated based on the accuracy and predictions. The developed model has an accuracy of about 80% and successfully detects the sentiments of the textual reviews. 

#### Custom Input
The model has been tested with a few custom reviews taken from user and it has detected the sentiments correctly in almost all the cases.

#### Future Work
The model needs to be trained with more data to improve its accuracy. Different classifier models can also be tested. An GUI can be integrated to take user input and display the results for its seamless functioning.
