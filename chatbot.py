import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import json
from tensorflow import keras
import nltk
from nltk.stem import WordNetLemmatizer

class Chatbot:
    def __init__(self, intent_bot_path, ner_bot_path=None, words_path="words.pkl", classes_path="classes.pkl", intents_path="intents.json"):
        print("loading models")
        self.intent_classifier = keras.models.load_model(intent_bot_path)
        # load the ner bot 
        self.words = pickle.load(open(words_path, "rb"))
        self.classes = pickle.load(open(classes_path, "rb"))
        self.intents = json.load(open(intents_path, "r"))
        self.lemmatizer = WordNetLemmatizer()
        print("models loaded")
    
    def preprocess(self, item):
        tokens = nltk.word_tokenize(item)
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def bag_of_words(self, sentence):
        sentence_words = self.preprocess(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        
        return np.array(bag)

    def predict_intent(self, sentence, tolerance=0.7):
        bow = self.bag_of_words(sentence)
        prediction = self.intent_classifier(np.array([bow]))[0]
        max_value = max(prediction).numpy()
        res = np.where(prediction.numpy() == max_value)[0][0]
        if res >= tolerance:
            return res
        else:
            return -1
