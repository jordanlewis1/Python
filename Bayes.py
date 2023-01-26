#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 13:09:00 2022

@author: jlewis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
#MNNB needs numeric data so use countVectorizer
import sklearn.feature_extraction.text as CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.naive_bayes as MNNB
#Multinomial naive bayes
import sklearn.pipeline as pipeline
#from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re


#IMPORT
df = pd.read_csv("apiCleaned2.csv")
#print(df.head())
#print(df.shape)
#print(df['Spam'].value_counts())


#spam words
#spamW = ' '.join(list(df[df['Spam'] == 1]['Text']))
#spamWC = WordCloud(width = 300, height = 280).generate(spamW)
#plt.figure(figsize = (10,8), facecolor = 'k')
#plt.imshow(spamWC)
#plt.axis('off')
#plt.tight_layout(pad = 0)
#plt.show()

#randomize data using sample()
data_random = df.sample(frac=1, random_state=1)
#frac=1 return all rows in random order
#random state used for initializing the internal random number generator
#decideds how the data is split into training and testing
#use 60% of testing data 
trainingTest = round(len(data_random) * 0.6)
#x_train, x_test, y_train, y_test = train_test_split(df.Text, df.Spam, test_size = 0.6)



trainS = data_random[:trainingTest].reset_index(drop=True)
testS = data_random[trainingTest:].reset_index(drop=True)

#print(trainS.shape)
#print(testS.shape)

#training_set['Spam'].value_counts(normalize=True)
#gives percentage of spam vs not spam
# clean data



#Goal since no priority is available, to get the most of the actual text.
#find the words to find key words that make a comment spam

#TOKENIZATION
#remove punctuation
trainS['Text'] = trainS['Text'].str.replace('\W', ' ')
#make lowercase
trainS['Text'] = trainS['Text'].str.lower()
#print(training_set.head())

#v = CountVectorizer()
#x_train_count = v.fit_transform[x_train.values]
#x_train_count.toarray()[:3]



#vector.fit(df)
#print(vector)
#punctuation and letter case does not effect how each word is compared

#parse/iterate and separate string into words to analyze each word separately
#append to a list

#split comments 
trainS['Text'] = trainS['Text'].str.split()


vocab = []
for text in trainS['Text']:
    for word in text:
        #append to list
        vocab.append(word)

#allow list to be manipulated as vocab
vocab = list(set(vocab))
#print(vocab)



#Word is each individual word in comment
#initiate list to keep track of individual words in each comment to 0
#curly braces are used to denote value of a dictionary
wordsC = {Word: [0] * len(trainS['Text']) for Word in vocab}

#enumate to save each time a word is found in each comment
#use nested for loop to check each comment
for index, Text in enumerate(trainS['Text']):
   for word in Text:
      wordsC[word][index] += 1

      
wordCS = pd.DataFrame(wordsC)
#print(wordCS.head())

#new dataframe with all parameters (allows each word to be analyzed separately)
#newdf is splits comments into words
newdf = pd.concat([trainS, wordCS], axis=1)
#format(newdf.head())

#cleaned data set
#initialize spam words and non spam words, not comments
spamWords = newdf[newdf['Spam'] == 1]
RealWords = newdf[newdf['Spam'] == 0]

# et probability of spam and not spam words
#use len to return number of spam words rather than the actual words

#Spam probability is spam  words /all words
probSpam = len(spamWords) / len(newdf)
#Real probability is real words / all words
probReal = len(RealWords) / len(newdf)

#Number of spam words in spam messages using sum
NumSpamWords = spamWords['Text'].apply(len)
NumSpam = NumSpamWords.sum()

#Number of Real words 
NumRealWords = RealWords['Text'].apply(len)
#.sum is used to get total
NumReal = NumRealWords.sum()

#number of vocab list
NumVocab = len(vocab)

#Used to tackle the problem of zero probability
alpha = 1

#find Prob of a individual word being spam or real
#conditional probability
ProbSP = {Word:0 for Word in vocab}
ProbWR = {Word:0 for Word in vocab}

#prob of individual words  being spam or not
for word in vocab:
    #Number of words given that it is spam by taking the sum of words only from spam
    NumWordGS = spamWords[word].sum()
    #Probability of a word given that its spam is Num of spam words
    ProbWordGS = (NumWordGS + alpha) / (NumSpam + alpha * NumVocab)
    ProbSP[word] = ProbWordGS
    
    #number of words given real is taking sum of words only from real
    NumWordGR = RealWords[word].sum()
    #Probability of a word given its real is num
    ProbWordGR = (NumWordGR + alpha) / (NumReal + alpha * NumVocab)
    ProbWR[word] = ProbWordGR 

#USE----------
#print(ProbSP)
#print(ProbWR)


#Spam filter
#Filter function is needed to classify the words as either spam or not
#we compare the probability of each individual word
def filter(comment):
    
    Spam = probSpam
    Real = probReal
    #for all words in comment
    for word in comment:
        #if word is in spam or real  individual word probability and multiplies it by
        #total prob of spam or not
        if word in ProbSP:
            Spam *= ProbSP[word]
        if word in ProbWR:
            Real *= ProbWR[word]
    #probabilities        
    print("P(Spam|Comment) is: ", Spam) 
    print("P(Real|Comment) is: ", Real)    
    
    #now the comparison is needed, if spam prob is higher then its spam if not then it is real
    if Spam > Real:
        print("Spam Comment")
    elif Real > Spam:
        print("Real Comment")
    else:
        print("It is equally likely to be a spam or real comment.")

#USE------------       
#filter("stay blessed")
#filter("cute")


#function to analyze test set
def dataFilter(comment):
    Spam = probSpam
    Real = probReal
    #for all words in comment
    for word in comment:
        #if word is in spam or real mult
        if word in ProbSP:
            Spam *= ProbSP[word]
        if word in ProbWR:
            Real *= ProbWR[word] 
    
    #now the comparison is needed, if spam prob is higher then its spam if not then it is real
    #return labels instead create new column stating if a comment is spam or not
    
    if Spam > Real:
        #value of spam in my data set is 1
        return "1"
    elif Real > Spam:
        #value of real in my data set is 0
        return "0"
    else:
        print("It is equally likely to be a spam or real comment.")
 
#prediction of spam values       
testS['Predicted'] = testS['Text'].apply(dataFilter)
#USE-----------
print(testS)

#The metric I used to analyze my model was accuracy
#we need to compare rows Spam (row 6) and new Predicted (row 7)



        
    
        
