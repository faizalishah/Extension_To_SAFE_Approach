
# coding: utf-8

# In[16]:


from xml.etree import ElementTree
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import SubElement
from xml.dom import minidom
import html
import re


# In[20]:


class XML_REVIEW_DATASET(object):
    
    def __init__(self,dataset,app_name):
        self.dataset = dataset
        self.app_name = app_name
    
    def ReadReviewSentsWithAspectTerms(self):
    
        app_data_path='../labeled_datasets/' + self.dataset.name +  "/"  + self.app_name.name + ".xml"


        tree = ElementTree.parse(app_data_path)
        corpus = tree.getroot()
        sentences = corpus.findall('.//sentence')
        
        sents_with_aspect_terms={}
        
        sent_node = Element('sentences')
        
        for sent in sentences:
            #sent_id = sent.attrib['id']
            #review_id = sent.attrib['review-id']
            review_id = sent.attrib['id']
            #sent_class = sent.attrib['class']
            sent_text = re.sub(r'\s+'," ",sent.find('text').text)
            
            aspectTerms = sent.find('aspectTerms')
            
            list_aspect_terms=[]
            
            if aspectTerms is not None:
                aspectTerm = aspectTerms.findall('aspectTerm')
               
                for aspect_term in aspectTerm:
                    app_feature = aspect_term.attrib['term'].strip()
                    list_aspect_terms.append(app_feature)
            
            sents_with_aspect_terms[review_id]={'review_sent':sent_text,'true_features':list_aspect_terms,'predicted_features':[], 'review-id':review_id} #,'sent_class': sent_class}
        
        return(sents_with_aspect_terms)

