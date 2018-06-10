
# coding: utf-8

# In[1]:


from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from scipy.sparse import hstack
import numpy as np
from nltk.util import ngrams
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics  import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from scipy.sparse import coo_matrix
import csv
import pickle
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import time


# In[2]:


stemmer = SnowballStemmer("english")


# In[3]:


from enum import Enum
class ENGLISH_APPS(Enum):
    ANGRY_BIRD = 343200656
    DROP_BOX= 327630330
    EVERNOTE = 281796108
    TRIP_ADVISOR = 284876795
    PIC_ART= 587366035
    PINTEREST = 429047995
    WHATSAPP=310633997


# In[4]:


class AppFeaturesClassification:
    def __init__(self):
        #self.TestApp = testApp # Use it once the model achieve reasonable performance via k-fold cross validation on other apps
        self.SetTrainDataset()
    
    def SetTrainDataset(self):
        path="../Train_Test_Datasets/"        
       
        self.train_lst_features=[]
        self.train_lst_labels=[]
        
        self.test_lst_features=[]
        self.test_lst_labels=[]
        
        self.list_description_words=[]
        
        pd_AspectTerms = pd.DataFrame()
        
        for app in ENGLISH_APPS:
            #print("Preparing Train Dataset.............")
            #preparing train-set following cross-app validation settings
#             if app.name!=self.TestApp.name:
                #print(app.name)
            file_tp_features_path = path +  app.name + "/" + app.name + "_tp.txt"
            file_fp_features_path = path +  app.name + "/" + app.name + "_fp.txt"

            app_tp_features = self.GetAppFeatures(file_tp_features_path)
            tp_class = [1] * len(app_tp_features)
            app_fp_features = self.GetAppFeatures(file_fp_features_path)
            fp_class = [0] * len(app_fp_features)

            self.train_lst_features.extend(app_tp_features)
            self.train_lst_features.extend(app_fp_features)

            self.train_lst_labels.extend(tp_class)
            self.train_lst_labels.extend(fp_class)

            with open("../" + app.name  + "_desc_words" + ".pkl", "rb") as fp:
                words_desc = pickle.load(fp)

            self.list_description_words.extend(words_desc)
            
#             else:
#                 file_tp_features_path = path + app.name + "/"  +  app.name + "_tp.txt"
#                 file_fp_features_path = path + app.name +  "/" + app.name + "_fp.txt"
                
#                 app_tp_features = self.GetAppFeatures(file_tp_features_path)
#                 tp_class = [1] * len(app_tp_features)
#                 app_fp_features = self.GetAppFeatures(file_fp_features_path)
#                 fp_class = [0] * len(app_fp_features)
                
#                 self.test_lst_features.extend(app_tp_features)
#                 self.test_lst_features.extend(app_fp_features)
                
#                 self.test_lst_labels.extend(tp_class)
#                 self.test_lst_labels.extend(fp_class)
        
        #print(Counter(self.test_lst_labels))
                
    def GetFeatureWordClusterWithScaleDistance(self,feature_word):
        filepath = 'word_clusters_with_scale_distances.pkl'
        
        with open(filepath, 'rb') as f:
            word_clusters_with_scale_distances = pickle.load(f)
        
        for cluster_no in word_clusters_with_scale_distances.keys():
            clustered_words_with_distances = word_clusters_with_scale_distances[cluster_no]
            
            if clustered_words_with_distances.get(feature_word,'na')!='na':
                return cluster_no,clustered_words_with_distances[feature_word]
        
        
        return -1,0
    
    def SentenceFeatures(self,aspectTerm,aspectTerm_wo_Stemmed):
        features = {}
        
        POS_TAGS_feature = '-'.join(tag for word,tag in aspectTerm)
        aspect_words = [word for word,tag in aspectTerm]
            
        features['{}'.format(POS_TAGS_feature)] = True
        
        if len(aspectTerm_wo_Stemmed)<4:
            remaining = 4 - len(aspectTerm_wo_Stemmed) 
            for i in range(0,remaining):
                aspectTerm_wo_Stemmed.append('')
        
        nltk_stop_words =  stopwords.words('english')
        sklearn_stop_words = list(stop_words.ENGLISH_STOP_WORDS)
        stop_words_list = set(nltk_stop_words + sklearn_stop_words)
        
        Contain_Stop_Word = any([w in stop_words_list for w in aspectTerm_wo_Stemmed])
        
        Word_in_App_Description = any([self.CheckWordInAppDesc(word) for word,tag in aspectTerm])
        
        features['{}'.format('CONTAIN_APP_DESCRIPTION_WORD')] = Word_in_App_Description
        
        for index,word in enumerate(aspectTerm_wo_Stemmed):
            word_cluster_label = 'WORD' + str(index+1) + "_CLUSTER"
            word_dist_label = 'WORD' + str(index+1) + "_DIST"
            
            if word!='':
                cluster_no,cluster_scale_dist = self.GetFeatureWordClusterWithScaleDistance(word)
                
            else:
                cluster_no,cluster_scale_dist = -1,0
                
            features['{}'.format(word_cluster_label)] = cluster_no
            features['{}'.format(word_dist_label)] = cluster_scale_dist
        
        
        return features
    
    def GenerateFeatureMatrix(self,Tag_Clean_AspectTerms,clean_AspectTerms_wo_Stemming):
        feature_dicts=[]
         
        for index,aspectTerm in enumerate(Tag_Clean_AspectTerms):
            Aspect_Words_WO_Stemming = clean_AspectTerms_wo_Stemming[index]
            features=self.SentenceFeatures(aspectTerm,Aspect_Words_WO_Stemming)
            feature_dicts.append(features)
        
        assert len(feature_dicts) == len(Tag_Clean_AspectTerms)
        
        return feature_dicts
      
    def CleanAppFeatures(self,lst_app_features):
        clean_aspectTerms=[]
        Tag_Clean_AspectTerms=[]
        clean_AspectTerms_wo_Stemming=[]
        
        for aspectTerm in lst_app_features:
            aspectTermWords = aspectTerm.split()
            tag_AspectWords = nltk.pos_tag(aspectTermWords)
            tag_AspectWords_stemmed = [(stemmer.stem(word.lower()),tag) for word,tag in tag_AspectWords]
            AspectWords_wo_Stemmed = [word for word,tag in tag_AspectWords if tag not in ['IN','DT','PRP$']]
            AspectWordsTagged = [(word,tag) for word,tag in tag_AspectWords_stemmed if tag not in ['IN','DT','PRP$']]
            AspectWords = ' '.join([word for word,tag in tag_AspectWords_stemmed if tag not in ['IN','DT','PRP$']])
            clean_aspectTerms.append(AspectWords)
            Tag_Clean_AspectTerms.append(AspectWordsTagged)
            clean_AspectTerms_wo_Stemming.append(AspectWords_wo_Stemmed)
        
        return clean_aspectTerms,Tag_Clean_AspectTerms,clean_AspectTerms_wo_Stemming
    
    def CheckWordInAppDesc(self,feature_word):     
        return (stemmer.stem(feature_word) in self.list_description_words)
    
    
    def Train_n_Evaluate_Model(self):
        
        model,count_vectorizer,dict_vectorizer = self.Train_Model()
        
        fn_model= 'finalized_model.sav'
        pickle.dump(model, open(fn_model, 'wb'))
        
        fn_countVectorizer = 'Count_Vectorizer.sav'
        pickle.dump(count_vectorizer, open(fn_countVectorizer, 'wb'))
        
        fn_dictVectorizer = 'Dict_Vectorizer.sav'
        pickle.dump(dict_vectorizer, open(fn_dictVectorizer, 'wb'))
        
#         Clean_AspectTerms,Tagged_Clean_AspctTerms,Clean_AspectTerms_Unstemmed = self.CleanAppFeatures(self.test_lst_features)
        
#         X = count_vectorizer.transform(Clean_AspectTerms)
        
#         custom_features_dict = self.GenerateFeatureMatrix(Tagged_Clean_AspctTerms,Clean_AspectTerms_Unstemmed)
#         custom_features_matrix = dict_vectorizer.transform(custom_features_dict)
        
#         X_Test = hstack((X,custom_features_matrix))
#         X_Test = X_Test.tocsr()
    
#         y_test = np.array(self.test_lst_labels)
        
#         pred_y_test = model.predict(X_Test)
        
#         precision,recall,fscore,_ = precision_recall_fscore_support(y_test,pred_y_test,labels=[1])
        
#         print('Precision: %.1f, Recall: %.1f, FScore: %.1f ' % (precision*100, recall*100,fscore*100))
        
#         return precision,recall,fscore

    
    def Train_Model(self):
        
        # bag-of-word model                                
        model = LogisticRegression()
        
        count_vectorizer = CountVectorizer()
                                   
        Clean_AspectTerms,Tagged_Clean_AspctTerms,Clean_AspectTerms_Unstemmed = self.CleanAppFeatures(self.train_lst_features)
        
        X = count_vectorizer.fit_transform(Clean_AspectTerms)
        
        custom_features_dict = self.GenerateFeatureMatrix(Tagged_Clean_AspctTerms,Clean_AspectTerms_Unstemmed)
        
        dict_vectorizer = DictVectorizer()
        custom_features_matrix = dict_vectorizer.fit_transform(custom_features_dict)

        
        X_Train = hstack((X,custom_features_matrix))
        X_Train = X_Train.tocsr()
        
        y = np.array(self.train_lst_labels)
        
        model.fit(X_Train,y)
        
        return model,count_vectorizer,dict_vectorizer
        
                
    def GetAppFeatures(self,path):
        with open(path) as f:
            appFeatures = f.readlines()
        lst_appFeatures = [appFeature.strip() for appFeature in appFeatures] 
        return lst_appFeatures


# In[5]:


if __name__ == '__main__':
#     lst_precision,lst_recall,lst_fscore=[],[],[]
#     for app in ENGLISH_APPS:
#         print('*'*5,app.name, '*' * 5)
    obj=AppFeaturesClassification()
    obj.Train_n_Evaluate_Model()
#         lst_precision.append(precision)
#         lst_recall.append(recall)
#         lst_fscore.append(fscore)

#     average_precision = np.mean(lst_precision)*100
#     average_precision_std = np.std(lst_precision)*100

#     average_recall = np.mean(lst_recall)*100
#     average_recall_std = np.std(lst_recall)*100

#     average_fscore = np.mean(lst_fscore)*100
#     average_fscore_std = np.std(lst_fscore)*100
    
#     print('Precision: %.1f +/- %.3f , Recall: %.1f +/- %.3f, FScore: %.1f +/- %.3f' % (average_precision,average_precision_std,\
#               average_recall,average_recall_std,average_fscore,average_fscore_std))

