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

class analyse:

    def __init__(self, data):
        self.md = data
    
    def catanalyse(self, category):
        final = self.md
        subject = '/' + category

        df_cats = pd.DataFrame(final["Category"])
        df_cats = df_cats.drop_duplicates()
        health_based = []
        for cat in df_cats["Category"]:
            if (cat.startswith(subject)):
                health_based.append(cat)
        health_based = list(set(health_based))
        terms = []
        for i in health_based:
            j = len(i) - 1
            ans = ''
            while (i[j] != '/'):
                ans += i[j]
                j = j - 1
            ans = ans[::-1]
            terms.append(ans)

        return terms

    def keyanalyse(self):
        final = self.md
        
        df_keys = pd.DataFrame(final["Keyword"])
        df_keys = df_keys.drop_duplicates()

        keys = []
        for key in df_keys["Keyword"]:
            res = key.split()  
            ans = ''
            for i in res:
                if (i in setofwords):
                    ans = ans + ' ' + i
        keys.append(ans[1:])
        
        keys = list(set(keys))
        return keys