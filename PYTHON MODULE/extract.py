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
import contractions
from symspellpy.symspellpy import SymSpell
import pkg_resources
from symspellpy import SymSpell, Verbosity



'''

1.  actual: The dataframe passed by the user. Will be cleaned down to just one column if multple provided
2. sample_size: How many samples to analyse at one go via IBM Cloud.
3. Chunksize: Chunk of data in one pass.

'''


# IMPORTING AND DOWNLOADING SYMSPELL PACKAFES
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)



class extract:
    
    def __init__(self, actual, sample_size = 10):
        self.actual = actual
        self.skipper = sample_size
# ________________________________________________________________________________________________________________________________________________

        
    # Authenticates the IBM WATSON API Keys using IAM
    def authenticate(self, API_KEY, URL):
        authenticator = IAMAuthenticator(API_KEY)
        service = NaturalLanguageUnderstandingV1(version='2018-03-16', authenticator=authenticator)
        service.set_service_url(URL)
        return service

# ________________________________________________________________________________________________________________________________________________
        
    
    # Using IBM Watson to get analysis
    def __perform(self, df_try, API_KEY, URL):
        data = pd.DataFrame(columns=["Data", "Language", "Sentiment", "Emotion", "Keyword", "Category"])
        try:
        	service = self.authenticate(API_KEY=API_KEY, URL=URL)
        except:
        	print("ERROR IN AUTHENTICATION. Please verify your credentials.")
    
        for tweet in df_try["Data"]:
            try:
                tw = tweet
                response = service.analyze(text=tweet, features = Features(sentiment= DocumentSentimentResults(), emotion=EmotionOptions(), keywords=KeywordsOptions(), categories=CategoriesOptions())).get_result()
            except:
                print("Error in analysing data: ", tw)
                continue
            
            try:
                lan = response["language"]
                sent = response["sentiment"]["document"]["label"]
            except:
                lan = 'en'
                sent = 'neutral'
            
            ans = -1
            place = -1
            emotion = []
            try:
                for i in response["emotion"]["document"]["emotion"]:
                    emotion.append(response["emotion"]["document"]["emotion"][i])
                for j in range(len(emotion)):
                    if emotion[j] > ans:
                        ans  = emotion[j]
                        place = j
                if (place == 0):
                    emot = 'sadness'
                elif (place == 1):
                    emot = 'joy'
                elif (place == 2):
                    emot = 'fear'
                elif (place == 3):
                    emot = 'disgust'
                else:
                    emot = 'anger'
            
            except:
                emot = "neutral"
            
            try:
                word = response["keywords"][0]["text"]
            except:
                word = '----'
            
            try:
                cat = response['categories'][0]['label']
            except:
                cat = 'Unknown'
            
            final = {"Data":tw, "Language":lan, "Sentiment":sent, "Emotion":emot, "Keyword":word, "Category":cat}
            data = data.append(final, ignore_index=True)
        return data

# ________________________________________________________________________________________________________________________________________________

   	# Return a smaller subset for analysis by IBM
    def __get_sample(self, df, nums, skip):
        return df[nums:nums + skip]   

# ________________________________________________________________________________________________________________________________________________


    # Removing short tweets
    def __remove_short(self, df):
        df.drop(df[df['Length'] < 30].index, inplace = True) 
        return df

# ________________________________________________________________________________________________________________________________________________


   # Simple cleaning
    # def __clean_simple(self, x):
    #     a = []
    #     for tw in x:
    #         tw = tw.lower()
    #         tw= re.sub(r'\n',' ', tw) # Remove line breaks
    #         tw = re.sub('\s+', ' ', tw).strip()# Remove leading, trailing, and extra spaces
    #         tw = re.sub(r'https?://\S+', '', tw) # Remove links
    #         tw = re.sub(r'http?://\S+', '', tw)  #Remove links

    #         # SPELL CORRECTION

    #         suggestions = sym_spell.lookup(tw, Verbosity.TOP)
    #         for suggestion in suggestions:
    #             word= suggestion.term
    #             word= contractions.fix(word)
    #             a.append(word)
    #         a.append(' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tw).split()))
    #     return set(a)

    def __clean_simple(self, tw):

        tw = tw.lower()
        tw= re.sub(r'\n',' ', tw) # Remove line breaks
        tw = re.sub('\s+', ' ', tw).strip()# Remove leading, trailing, and extra spaces
        tw = re.sub(r'https?://\S+', '', tw) # Remove links
        tw = re.sub(r'http?://\S+', '', tw)  #Remove links
        tw = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tw)
        list_of_words = tw.split();

        # SPELL CORRECTION
        for i in range(len(list_of_words)):
            word = list_of_words[i]
            suggestions = sym_spell.lookup(word, Verbosity.TOP)
            ans = ''
            for suggestion in suggestions:
                ans = suggestion.term
                ans = contractions.fix(ans)
            if (ans == ''):
                list_of_words[i] = word
            else:
                list_of_words[i] = ans

        tw = ' '.join(list_of_words)
        return tw

# ________________________________________________________________________________________________________________________________________________


    # Simple analytical cleaning
    def __analyse_tweet(self, df):
        df = df.drop_duplicates(subset = "Data")
        df = df.reset_index(drop = True)                                                                             
        # x = set(df["Data"])                            # Doesn't make sense to remove duplicates and then create set. Removing to apply functions directly via Pandas
        df["Data"] = df.Data.apply(self.__clean_simple);
        # a = self.__clean_simple(x)
        # a = list(a)
        # a = a[1:]
        # dff = pd.DataFrame(a, columns=["Data"])
        return df

# ________________________________________________________________________________________________________________________________________________

    # Helper function for perform_analysis
    def __extract_information(self, df, API_KEY, URL):
        df = self.__analyse_tweet(df)
        skip = self.skipper	
        lengths = []
        for i in df["Data"]:
            l = len(i)
            lengths.append(l)
        df["Length"] = lengths
        df = self.__remove_short(df)
        del df["Length"]
        md = pd.DataFrame(columns=["Data", "Language", "Sentiment", "Emotion", "Keyword", "Category"])
        for nums in tqdm(range(0, df.shape[0], skip)):
            df_mini = self.__get_sample(df, nums, skip)
            df_mini = df_mini.reset_index(drop = True)
            df_res = self.__perform (df_mini, API_KEY, URL)
            md = pd.concat([md, df_res], axis=0)
        print()
        return md

# ________________________________________________________________________________________________________________________________________________

    
    # Main
    def extract_features(self, API_KEY, URL, chunksize = 200):
        size = self.actual.shape[0]
        copy = self.actual
        final = pd.DataFrame(columns=["Data", "Language", "Sentiment", "Emotion", "Keyword", "Category"])

        for i in range(0, self.actual.shape[0], chunksize):
            res = self.__extract_information(copy[i:i + chunksize], API_KEY, URL)
            i = i + chunksize
            final = pd.concat([final, res], axis=0)
        final = final.reset_index(drop = True)
        return final

# ________________________________________________________________________________________________________________________________________________

