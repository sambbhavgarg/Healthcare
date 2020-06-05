import json
import csv 
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, DocumentSentimentResults, CategoriesOptions, EmotionOptions
import boto3
import botocore
import pandas as pd
from nltk.corpus import words
setofwords = set(words.words())
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from tqdm import tqdm

class visualise:

    def __init__(self, data):
        self.data = data

    def textvis(self):
        md = self.data
        target_cnt = Counter(md.Sentiment)
        plt.figure(figsize=(16,8))
        plt.bar(target_cnt.keys(), target_cnt.values())
        plt.title("Dataset sentiment distribuition")
        plt.show()
    
        target_cnt = Counter(md.Emotion)
        plt.figure(figsize=(16,8))
        plt.bar(target_cnt.keys(), target_cnt.values())
        plt.title("Emotion distribuition")
        plt.show()
    
        keywords = md["Keyword"]
        freq = {} 
        for item in keywords: 
            if (item in freq): 
                freq[item] += 1
            else: 
                freq[item] = 1
        freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
        count = Counter(freq)
        a = dict(count.most_common(7))
        plt.figure(figsize=(20,8))
        plt.bar(a.keys(), a.values())
        plt.title("Top KeyWords")
        plt.show()
    
        cats = md["Category"]
        freq = {} 
        for item in cats: 
            if (item in freq): 
                freq[item] += 1
            else: 
                freq[item] = 1
        freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
        count = Counter(freq)
        a = dict(count.most_common(3))
        plt.figure(figsize=(20,8))
        plt.bar(a.keys(), a.values())
        plt.title("Top Categories")
        plt.show()
    
        return