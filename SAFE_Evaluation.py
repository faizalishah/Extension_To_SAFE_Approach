
# coding: utf-8

# In[1]:


import nltk
from nltk.stem.snowball import SnowballStemmer
import time
from scipy.sparse import hstack
import numpy as np
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics  import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from imblearn.over_sampling import RandomOverSampler
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from scipy.sparse import coo_matrix
import csv
import pickle
import time


# In[2]:


stemmer = SnowballStemmer("english")


# In[ ]:


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


from enum import Enum

class EVALUATION_TYPE(Enum):
    EXACT=1
    PARTIAL=2


# In[1]:


def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None


# In[33]:


class Evaluate:
    def __init__(self,dict_true_n_extracted_features,extracted_features):
        self.dict_True_n_Extracted_Features=dict_true_n_extracted_features
        self.predicted_features = extracted_features
        
        self.LoadFeatureWordsFromAppDescription()
        #self.predicted_features_cluster = cluster_predicted_features
        #self.evaluation_type = evaluation_type
    
    def LoadFeatureWordsFromAppDescription(self):
        self.list_description_words=[]
        
        for app in ENGLISH_APPS:
            with open("../" + app.name  + "_desc_words" + ".pkl", "rb") as fp:
                words_desc = pickle.load(fp)
            
            self.list_description_words.extend(words_desc)
    
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
    
    def GetFeatureWordClusterWithScaleDistance(self,feature_word):
        filepath = 'word_clusters_with_scale_distances.pkl'
        
        with open(filepath, 'rb') as f:
            word_clusters_with_scale_distances = pickle.load(f)
        
        for cluster_no in word_clusters_with_scale_distances.keys():
            clustered_words_with_distances = word_clusters_with_scale_distances[cluster_no]
            
            if clustered_words_with_distances.get(feature_word,'na')!='na':
                return cluster_no,clustered_words_with_distances[feature_word]
        
        return -1,0
    
    def CheckWordInAppDesc(self,feature_word):
       
    #         with open("../" + self.TestApp.name  + "_desc_words" + ".pkl", "rb") as fp:
    #             words_desc = pickle.load(fp)
            
        return (stemmer.stem(feature_word) in self.list_description_words)
    
    def GenerateFeatureMatrix(self,Tag_Clean_AspectTerms,clean_AspectTerms_wo_Stemming):
        feature_dicts=[]
         
        for index,aspectTerm in enumerate(Tag_Clean_AspectTerms):
            Aspect_Words_WO_Stemming = clean_AspectTerms_wo_Stemming[index]
            features=self.SentenceFeatures(aspectTerm,Aspect_Words_WO_Stemming)
            feature_dicts.append(features)
        
        assert len(feature_dicts) == len(Tag_Clean_AspectTerms)
        
        return feature_dicts

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
        
        features['{}'.format('CONTAIN_STOPWORD')] = Contain_Stop_Word
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
    
    def PerformEvaluation(self):
        
        true_predictions = 0
        false_predictions = 0
        false_negatives = 0
        
        tp_partials = 0
        fp_partials = 0
        fn_partials = 0
        
        self.predicted_aspects_list=[]
        self.true_aspects_list = []
        
        for sent_id in self.dict_True_n_Extracted_Features.keys():
            
            true_features = self.dict_True_n_Extracted_Features[sent_id]['true_features']
            
            predicted_features = self.dict_True_n_Extracted_Features[sent_id]['predicted_features']
            
            if len(true_features)!=0 :
                for i in range(0,len(true_features)):
                    t_aspect = true_features[i].split()
                    if len(t_aspect)>=2 and len(t_aspect)<=4:
                        str_true_aspect = ' '.join(t_aspect)
                        self.true_aspects_list.append(str_true_aspect.lower().strip())

            if len(predicted_features)!=0 :
                for i in range(0,len(predicted_features)):
                    p_aspect = predicted_features[i].split()
                    str_predicted_aspect = ' '.join(p_aspect)
                    self.predicted_aspects_list.append(str_predicted_aspect.lower().strip())
        
        self.predicted_aspects_list = self.CleanFeatures(self.predicted_aspects_list)
        
        Clean_AspectTerms,Tagged_Clean_AspctTerms,Clean_AspectTerms_Unstemmed = self.CleanAppFeatures(self.predicted_aspects_list)
        
        #A binary classification model to categorize predicted app features into 
        #true app features vs. false app featuares
        
        with open(r"Count_Vectorizer.sav", "rb") as input_file:
            count_vectorizer = pickle.load(input_file)
        
        with open(r"Dict_Vectorizer.sav", "rb") as input_file:
            dict_vectorizer = pickle.load(input_file)
        
        with open(r"finalized_model.sav", "rb") as input_file:
            model = pickle.load(input_file)
        
        X = count_vectorizer.transform(Clean_AspectTerms)
        
        custom_features_dict = self.GenerateFeatureMatrix(Tagged_Clean_AspctTerms,Clean_AspectTerms_Unstemmed)
        custom_features_matrix = dict_vectorizer.transform(custom_features_dict)
        
        X_Test = hstack((X,custom_features_matrix))
        X_Test = X_Test.tocsr()
        
        pred_y_test = model.predict(X_Test)
        
        predicted_list_array = np.array(self.predicted_aspects_list)
        
        model_predicted_true_aspect=predicted_list_array[pred_y_test==1]
    
        precision_type_partial,recall_type_partial,fscore_type_partial,tp_aspects, fp_aspects = self.EvaluateAspects_type_based_partial(self.true_aspects_list,model_predicted_true_aspect)       
        
        evaluation_type_partial = {'precision' : '%.3f' % precision_type_partial ,'recall' : '%.3f' % recall_type_partial,'fscore' : '%.3f' % fscore_type_partial}
       
        return evaluation_type_partial,tp_aspects,fp_aspects
        

    def CleanFeatures(self,extracted_app_features):
        #print('Cleaning Feautres Started -> ')
        start_time = time.time()
        list_clean_feaures=[]
        # remove noise
        for feature in extracted_app_features:
            words = feature.split()
            duplicate_words = all(stemmer.stem(x) == stemmer.stem(words[0]) for x in words)
            #duplicate_words = all(x == words[0] for x in words)
            
            if duplicate_words!=True:
                list_clean_feaures.append(feature)
        
        list_clean_features = set(list_clean_feaures)
        
        clean_features_list = list_clean_feaures.copy()
        
        clean_features=[]
        
        for feature_term1 in list_clean_feaures:
            status = any([feature_term1 in f for f in list_clean_feaures if f!=feature_term1])
            if status==True:
                clean_features_list.remove(feature_term1)
        
        
        return clean_features_list
    
    def EvaluateAspects_type_based(self,true_aspects,predicted_aspects):

        stemmer = SnowballStemmer("english")

        stemmed_true_aspects=[]
        stemmed_predicted_aspects=[]
        
        predicted_aspects=[aspect for aspect in predicted_aspects if aspect!='']
        
        for t_aspect in true_aspects:
            t_aspect_words = t_aspect.strip().split()
            t_aspect_words = [stemmer.stem(w) for w in t_aspect_words]
            update_taspect_term = ' '.join(t_aspect_words)
            stemmed_true_aspects.append(update_taspect_term.strip())

        
        for p_aspect in predicted_aspects:
            p_aspect_words = p_aspect.strip().split()
            p_aspect_words = [stemmer.stem(w) for w in p_aspect_words]
            update_paspect_term = ' '.join(p_aspect_words)
            stemmed_predicted_aspects.append(update_paspect_term.strip())
        
        unique_true_aspects = set(stemmed_true_aspects)
        unique_predicted_aspects = set(stemmed_predicted_aspects)
        
        print("# of unique predicted aspects = %d" % (len(unique_predicted_aspects)))

        tp = 0
        fp = 0
        fn = 0
        
        tp_aspects=[]
        
        #print(unique_true_aspects)
        
        if len(unique_true_aspects)>0 and len(unique_predicted_aspects)>0:
        
            for p_aspect in unique_predicted_aspects:
                
                match = False

                for t_aspect in unique_true_aspects:                

                    status = self.MatchFeatureWords(p_aspect,t_aspect)
                    #print(status,"->",p_aspect,"||||",t_aspect)
                    #match_count = match_count + 1

                    if status == True:
                        match = True
                        break;

                if match == False:
                    #print('fp->',p_aspect)
                    fp = fp + 1
                    #print('fp value ->',fp)
                

            # False Negative cases

            for t_aspect in unique_true_aspects:     

                match=False

                for p_aspect in unique_predicted_aspects:
                    status = self.MatchFeatureWords(p_aspect,t_aspect)

                    if status==True:
                        tp_aspects.append(t_aspect)
                        #print('tp->',t_aspect)
                        tp = tp  + 1
                        match = True
                        traverse = True
                        break


                if match == False:
                    fn = fn + 1
        
        try:
            precision = tp/(tp + fp)
        except ZeroDivisionError as err:
            precision=0
        
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError as err:
            recall = 0
        
        try:
            fscore = 2 * (precision * recall)/(precision + recall)
        except ZeroDivisionError as err:
            fscore = 0 
            
        return (tp,fp,fn,precision,recall,fscore,tp_aspects)
    
    def MatchFeatureWords(self,p_aspect,t_aspect):
        
        t_aspect_words = t_aspect.split()
        t_aspect_size = len(t_aspect_words)
        
        p_aspect_words = p_aspect.split()
        p_aspect_size = len(p_aspect_words)        
        
        if p_aspect_size==t_aspect_size:
                status = False
                status = all([p_word in t_aspect_words for p_word in p_aspect_words])

                return(status)
        
                    
        return (False)
    
    def MatchFeatureWords_Partial(self,p_aspect,t_aspect):
        
        t_aspect_words = t_aspect.split()
        t_aspect_size = len(t_aspect_words)
        
        p_aspect_words = p_aspect.split()
        p_aspect_size = len(p_aspect_words)
        
        t_aspect_words_stemmed = [stemmer.stem(w) for w in t_aspect_words]
        p_aspect_words_stemmed = [stemmer.stem(w) for w in p_aspect_words]
        
        greater=None
        diff = 0

        if t_aspect_size > p_aspect_size:
            diff = t_aspect_size - p_aspect_size
            greater='true'
        elif p_aspect_size > t_aspect_size:
            greater = 'predict'
            diff = p_aspect_size - t_aspect_size

        status = False

        if greater=='true':
            status = all([w in t_aspect_words_stemmed for w in p_aspect_words_stemmed])
        elif greater=='predict':
            status = all([w in p_aspect_words_stemmed for w in t_aspect_words_stemmed])
        elif diff==0 and p_aspect_size==t_aspect_size:
            status = all([p_word in t_aspect_words_stemmed for p_word in p_aspect_words_stemmed])

        return (status)
    
    
    def EvaluateAspects_type_based_partial(self,true_aspects,predicted_aspects):
        
        #print('Evaluation started ..............')

        stemmer = SnowballStemmer("english")
        
        start_time = time.time()
        
        predicted_aspects=[aspect for aspect in predicted_aspects if aspect!='']

        tp = 0
        fp = 0
        fn = 0
        
        tp_aspects=[]
        fp_aspects=[]
        
        if len(true_aspects)>0 and len(predicted_aspects)>0:
        
            for p_aspect in predicted_aspects:
                
                match = False

                for t_aspect in true_aspects:                

                    status = self.MatchFeatureWords_Partial(p_aspect,t_aspect)

                    if status == True:
                        match = True
                        break;

                if match == False:
                    fp = fp + 1
                    fp_aspects.append(p_aspect)

            for t_aspect in true_aspects:     

                match=False


                for p_aspect in predicted_aspects:
                    status = self.MatchFeatureWords_Partial(p_aspect,t_aspect)

                    if status==True:
                        tp_aspects.append(t_aspect)
                        tp = tp  + 1
                        match = True
                        break


                if match == False:
                    fn = fn + 1
        try:
            precision = tp/(tp + fp)
        except ZeroDivisionError as err:
            precision=0
        
        try:
            recall = tp / (tp + fn)
        except ZeroDivisionError as err:
            recall = 0
        
        try:
            fscore = 2 * ((precision * recall)/(precision + recall))
        except ZeroDivisionError as err:
            fscore = 0 
        
        
        return (precision,recall,fscore,tp_aspects,fp_aspects)

