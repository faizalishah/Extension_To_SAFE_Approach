
# coding: utf-8

# In[29]:


import spacy
import SAFE_Patterns
from Text_Preprocessing import TextProcessing
from Feature_Matching import Merge_Features
import SAFE_Evaluation
import ReadXMLData
from ReadXMLData import XML_REVIEW_DATASET
from SAFE_Evaluation import Evaluate
import Feature_Matching
import Text_Preprocessing
import importlib
import json
from urllib.request import urlopen
import re
import requests
from nltk.stem.snowball import SnowballStemmer
import time
import nltk
import numpy as np
import pickle
from numpy import linalg as LA
import math


# In[30]:


importlib.reload(SAFE_Evaluation)
importlib.reload(SAFE_Patterns)
importlib.reload(Text_Preprocessing)
importlib.reload(Feature_Matching)
importlib.reload(ReadXMLData)


# In[10]:


stemmer = SnowballStemmer("english")


# In[11]:


nlp = spacy.load('en_core_web_sm')


# In[12]:


from enum import Enum
class ENGLISH_APPS(Enum):
    ANGRY_BIRD = 343200656
    DROP_BOX= 327630330
    EVERNOTE = 281796108
    TRIP_ADVISOR = 284876795
    PIC_ART= 587366035
    PINTEREST = 429047995
    WHATSAPP=310633997

class DATASETS(Enum):
    GUZMAN_ORIGINAL=4
    
class ANNOTATORS:
    CODER1 = 1 
    CODER2 = 2 


# In[31]:


class EXTRACTION_MODE(Enum):
    APP_DESCRIPTION = 1
    USER_REVIEWS= 2

class MERGE_MODE(Enum):
    DESCRIPTION=1
    USER_REVIEWS=2
    DESCRIPTION_USER_REVIEWS=3

class EVALUATION_TYPE(Enum):
    EXACT=1
    PARTIAL=2


# In[32]:


class SAFE:
    def __init__(self,appName,review_sents,nlp):
        self.nlp = nlp
        self.data = review_sents
        self.appName= appName
       
    def GetReviewsWithExtractedFeatures(self):
        self.PreprocessData()
        return self.data,self.extracted_app_features_reviews
    
    def ReplaceCommonTypos(self,sent):
        typos=[["im","i m","I m"],['Ill'],["cant","cnt"],["doesnt"],["dont"],["isnt"],["didnt"],["couldnt"],["Cud"],["wasnt"],["wont"],["wouldnt"],["ive"],["Ive"],["theres"]      ,["awsome", "awsum","awsm"],["Its"],["dis","diz"],["plzz","Plz ","plz","pls ","Pls","Pls "],[" U "," u "],["ur"],      ["b"],["r"],["nd ","n","&"],["bt"],["nt"],["coz","cuz"],["jus","jst"],["luv","Luv"],["gud"],["Gud"],["wel"],["gr8","Gr8","Grt","grt"],      ["Gr\\."],["pics"],["pic"],["hav"],["nic"],["nyc ","9ce"],["Nic"],["agn"],["thx","tks","thanx"],["Thx"],["thkq"],      ["Xcellent"],["vry"],["Vry"],["fav","favo"],["soo","sooo","soooo"],["ap"],["b4"],["ez"],["w8"],["msg"],["alot"],["lota"],["kinda"],["omg"],["gota"]]
        replacements=["I'm","i will","can't","doesn't","don't","isn't","didn'tfu","couldn't","Could","wasn't","won't","wouldn't","I have","I have","there's","awesome",             "It's","this","please","you","your","be","are","and","but","not","because","just","love","good","Good","well","great","Great\\.",             "pictures","picture","have","nice","nice","Nice","again","thanks","Thanks","thank you","excellent","very","Very","favorite","so","app","before","easy","wait","message","a lot","lot of","kind of","oh, my god","going to"]

        sent_tokens = nltk.word_tokenize(sent)

        new_sent=""

        for i,token in enumerate(sent_tokens):
            found_type=False
            for index,lst in enumerate(typos):
                if token in lst:
                    new_sent +=  replacements[index]
                    if i<(len(sent_tokens)-1):
                        new_sent += " "
                        found_type=True
                        break

            if found_type == False:
                new_sent += token 
                if i<(len(sent_tokens)-1):
                    new_sent += " "
        
        return(new_sent.strip())
    
    def PreprocessData(self):
        self.reviews_with_sents_n_features={}

        count=0

        for review_id in self.data.keys():
            review_sent_text  = self.data[review_id]['review_sent']
            sents_with_features={}
            reviewSent_wise_features=[]
            textProcessor = TextProcessing(self.appName,review_sent_text)
            unclean_sents = textProcessor.SegmemtintoSentences(sents_already_segmented=False)
            review_clean_sentences = textProcessor.GetCleanSentences()
            SAFE_Patterns_Obj=SAFE_Patterns.SAFE_Patterns(self.appName,review_id,review_clean_sentences,unclean_sents)
            sents_with_features = SAFE_Patterns_Obj.ExtractFeatures_Analyzing_Sent_POSPatterns()

            for sid in sents_with_features.keys():
                reviewSent_wise_features.extend(sents_with_features[sid]['extracted_features'])

            self.reviews_with_sents_n_features[review_id] = sents_with_features
            self.data[review_id]['predicted_features'] = reviewSent_wise_features
        
            count = count + 1
        

        self.extracted_app_features_reviews = self.GetListOfExtractedAppFeatures()
        
    def GetListOfExtractedAppFeatures(self):
        list_extracted_app_features=[]
        for sent_id in self.reviews_with_sents_n_features.keys():
            sents_with_app_features = self.reviews_with_sents_n_features[sent_id]
            for sent_id in sents_with_app_features.keys():
                app_features = sents_with_app_features[sent_id]['extracted_features']
                list_extracted_app_features.extend(app_features)
        
        return(list_extracted_app_features)
    
    def SaveFeatureInFile(self,path,lst_appfeatures):
        out_file = open(path, 'w')
    
        
        for item in lst_appfeatures:
              out_file.write("%s\n" % item)

        out_file.close()


# In[33]:


if __name__ == '__main__':
    
    for app in ENGLISH_APPS:
        
        print('*' * 5, app.name,'*' * 5)
        objXML_DS = XML_REVIEW_DATASET(DATASETS.GUZMAN_ORIGINAL,app)
        reviewSents_with_true_features = objXML_DS.ReadReviewSentsWithAspectTerms()
      
        obj_surf = SAFE(app.name,reviewSents_with_true_features,nlp)
        true_features_dict,extracted_features = obj_surf.GetReviewsWithExtractedFeatures()
        
        obj_Evaluation = Evaluate(true_features_dict,extracted_features)
        eval_results,tp_aspectTerms,fp_aspectTerms = obj_Evaluation.PerformEvaluation()
    
        print('Precision : %.3f , Recall , %.3f , F1-score : %.3f' % (float(eval_results['precision']),float(eval_results['recall']),float(eval_results['fscore'])))
        print("")

